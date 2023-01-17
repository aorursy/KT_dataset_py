# General imports

import os

import numpy as np

import pandas as pd

from tqdm.notebook import tqdm

from joblib import Parallel, delayed



# Tensorflow

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint



# Transformers

from transformers import TFAutoModel, TFBertModel, AutoTokenizer
# Detect hardware, return appropriate distribution strategy

try:

    # TPU detection. No parameters necessary if TPU_NAME environment variable is

    # set: this is always the case on Kaggle.

    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

    print('Running on TPU ', tpu.master())

except ValueError:

    tpu = None



if tpu:

    tf.config.experimental_connect_to_cluster(tpu)

    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)

else:

    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.

    strategy = tf.distribute.get_strategy()



print("REPLICAS: ", strategy.num_replicas_in_sync)
# Configuration parameters

AUTO = tf.data.experimental.AUTOTUNE

BATCH_SIZE = 16 * strategy.num_replicas_in_sync

MAX_LEN = 192

MODEL = 'dccuchile/bert-base-spanish-wwm-uncased' # for BERT model replace by e.g. dccuchile/bert-base-spanish-wwm-uncased   # jplu/tf-xlm-roberta-large

LANG = "es" # can be any of es, it, tr in this notebook

CONSTANT_LR = 3e-6 # 3e-6 generally good. Set lower e.g. 1e-6 for more finetuning

BALANCEMENT = [0.8, 0.2] # non-toxic vs. toxic

BERT_MODEL = True # specify if the given model is a BERT model

N_EPOCHS = 3 # 3-5 epochs are usually enough. Set higher e.g. 5 for more finetuning

N_ITER_PER_EPOCH = 10

PREDICT_START_ITER = 10 # start iteration to predict on test. best iterations found around +-20 (2 full epochs)



# Upgrades

STAGE2 = True # resume training with checkpoint of best model

REPEAT_PL = 6 # Upgrade: repeat PL with train (I repeated 6x on my last subs). Default=0 (no pseudolabels)
def regular_encode(texts, max_len):

    """

    Tokenizing the texts into their respective IDs using regular batch encoding

    

    Accepts: * texts: the text to be tokenize

             * max_len: max length of text

    

    Returns: * array of tokenized IDs 

    """

    enc_di = tokenizer.batch_encode_plus(

        texts, 

        return_attention_masks=False, 

        return_token_type_ids=False,

        pad_to_max_length=True,

        max_length=max_len

    )

    

    return np.array(enc_di['input_ids'])
def parallel_encode(texts, max_len):

    """

    Tokenizing the texts into their respective IDs using parallel processing

    

    Accepts: * texts: the text to be tokenized

             * max_len: max length of text

    

    Returns: * array of tokenized IDs + the toxicity label  

    """

    enc_di = tokenizer.encode_plus(

        str(texts[0]),

        return_attention_masks=False, 

        return_token_type_ids=False,

        pad_to_max_length=True,

        max_length=max_len

    )

    return np.array(enc_di['input_ids']), texts[1]
def build_model(transformer, max_len):

    """

    Build the model by using transformer layer and simple CLS token

    

    Accepts: * transformer: transformer layer

             * max_len: max length of text

    

    Returns: * model 

    """

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    sequence_output = transformer(input_word_ids)[0]

    cls_token = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(cls_token)

    model = Model(inputs=input_word_ids, outputs=out)    

    return model
# First load the real tokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL)
train = pd.read_csv(f"/kaggle/input/jigsaw-train-multilingual-coments-google-api/jigsaw-toxic-comment-train-google-{LANG}-cleaned.csv")

valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

test = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv")
print(len(train), len(valid), len(test))
# if REPEAT_PL:

#     sub = pd.read_csv("../input/multilingual-toxic-comments-training-data/test9500.csv") # use one of earlier subs

#     print(sub.head())

#     print(len(sub))

#     sub["comment_text"] = test["content"]

#     sub = sub.loc[test["lang"]==LANG].reset_index(drop=True)

#     sub_repeat = pd.concat([sub]*REPEAT_PL, ignore_index=True) # repeat PL multipe times for training

#     print('\n', sub_repeat.head())

#     print(len(sub_repeat))

#     same_cols = ["comment_text", "toxic"]

#     train = pd.concat([train[same_cols], sub_repeat[same_cols]]).sample(frac=1).reset_index(drop=True)
# Get specific validation and test

valid = valid.loc[valid["lang"]==LANG].reset_index(drop=True)

test = test.loc[test["lang"]==LANG].reset_index(drop=True)
%%time 

# Tokenize train with parallel processing

rows = zip(train['comment_text'].values.tolist(), train.toxic.values.tolist())

x_y_train = Parallel(n_jobs=4, backend='multiprocessing')(delayed(parallel_encode)(row, max_len=MAX_LEN) for row in tqdm(rows))
print(len(x_y_train))

try:

    print(x_y_train.shape)

except:

    pass

print(x_y_train[0])
x_train = np.vstack(np.array(x_y_train)[:,0])



y_train = np.array(x_y_train)[:,1]

y_train = np.asarray(y_train).astype('float32').reshape((-1,1))

print(y_train.shape)
%%time

# Tokenize valid regular processing

x_valid = regular_encode(valid.comment_text.values, max_len=MAX_LEN)



y_valid = valid.toxic.values

y_valid = np.asarray(y_valid).astype('float32').reshape((-1,1)) 
%%time

x_test = regular_encode(test.content.values, max_len=MAX_LEN)
# Train and valid dataset

train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_train, y_train))

    .shuffle(buffer_size=len(x_train), seed = 18)

    .prefetch(AUTO)

)



valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_valid, y_valid))

    .batch(BATCH_SIZE)

    .cache()

    .prefetch(AUTO)

)



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(x_test)

    .batch(BATCH_SIZE)

)
# Balance the train dataset by creating seperate negative and positive datasets. 

# Note: tf.squeeze remove the added dim to labels

# Example taken from https://www.tensorflow.org/guide/data



negative_ds = (

  train_dataset

    .filter(lambda _, y: tf.squeeze(y)==0)

    .repeat())



positive_ds = (

  train_dataset

    .filter(lambda _, y: tf.squeeze(y)==1)

    .repeat())



balanced_ds = tf.data.experimental.sample_from_datasets(

    [negative_ds, positive_ds], BALANCEMENT).batch(BATCH_SIZE) # Around 80%/20% to be expected for 0/1 labels
# distribute the datset according to the strategy

train_dist_ds = strategy.experimental_distribute_dataset(balanced_ds)

valid_dist_ds = strategy.experimental_distribute_dataset(valid_dataset)
# Instantiate metrics

with strategy.scope():

    # Accuracy, AUC, loss train

    train_accuracy = tf.keras.metrics.BinaryAccuracy()

    train_auc = tf.keras.metrics.AUC()

    train_loss = tf.keras.metrics.Sum()

    

    # Accuracy, AUC, loss valid

    valid_accuracy = tf.keras.metrics.BinaryAccuracy()

    valid_auc = tf.keras.metrics.AUC()

    valid_loss = tf.keras.metrics.Sum()

    

    # TP, TN, FN, FP train

    train_TP = tf.keras.metrics.TruePositives()

    train_TN = tf.keras.metrics.TrueNegatives()

    train_FP = tf.keras.metrics.FalsePositives()

    train_FN = tf.keras.metrics.FalseNegatives()

    

    # TP, TN, FN, FP valid

    valid_TP = tf.keras.metrics.TruePositives()

    valid_TN = tf.keras.metrics.TrueNegatives()

    valid_FP = tf.keras.metrics.FalsePositives()

    valid_FN = tf.keras.metrics.FalseNegatives()

    

    # Loss function and optimizer

    loss_fn = lambda a,b: tf.nn.compute_average_loss(tf.keras.losses.binary_crossentropy(a,b), global_batch_size=BATCH_SIZE)

    optimizer = tf.keras.optimizers.Adam(learning_rate=CONSTANT_LR)
@tf.function

def train_step(tokens, labels):

    with tf.GradientTape() as tape:

        probabilities = model(tokens, training=True)

        loss = loss_fn(labels, probabilities)

    grads = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    

    # update metrics

    train_accuracy.update_state(labels, probabilities)

    train_auc.update_state(labels, probabilities)

    train_loss.update_state(loss)

    

    train_TP.update_state(labels, probabilities)

    train_TN.update_state(labels, probabilities)

    train_FP.update_state(labels, probabilities)

    train_FN.update_state(labels, probabilities)

    

@tf.function

def valid_step(tokens, labels):

    probabilities = model(tokens, training=False)

    loss = loss_fn(labels, probabilities)

    

    # update metrics

    valid_accuracy.update_state(labels, probabilities)

    valid_auc.update_state(labels, probabilities)

    valid_loss.update_state(loss)

    

    valid_TP.update_state(labels, probabilities)

    valid_TN.update_state(labels, probabilities)

    valid_FP.update_state(labels, probabilities)

    valid_FN.update_state(labels, probabilities)

    
%%time

with strategy.scope():

    if BERT_MODEL:

        transformer_layer = TFBertModel.from_pretrained(MODEL, from_pt=True)

    else:

        transformer_layer = TFAutoModel.from_pretrained(MODEL)

    model = build_model(transformer_layer, max_len=MAX_LEN)

model.summary()
VALIDATION_STEPS = x_valid.shape[0] // BATCH_SIZE

STEPS_PER_EPOCH = x_train.shape[0] // (BATCH_SIZE*N_ITER_PER_EPOCH)

print("Steps per epoch:", STEPS_PER_EPOCH)

EPOCHS = N_EPOCHS*N_ITER_PER_EPOCH



best_auc = 0

epoch = 0



preds_all = []

for step, (tokens, labels) in enumerate(train_dist_ds):

    # run training step

    strategy.experimental_run_v2(train_step, args=(tokens, labels))

    print('=', end='', flush=True)

    

    # print metrics training

    if ((step+1) // STEPS_PER_EPOCH) > epoch:

        print("\n Epoch:", epoch)

        print('|', end='', flush=True)

        print("TP -  TN  -  FP  -  FN")

        print(train_TP.result().numpy(), train_TN.result().numpy(), train_FP.result().numpy(), train_FN.result().numpy())

        print("train AUC: ",train_auc.result().numpy())

        print("train loss: ", train_loss.result().numpy() / STEPS_PER_EPOCH)

        

        # validation run for es, it, tr and save model

        for tokens, labels in valid_dist_ds:

            strategy.experimental_run_v2(valid_step, args=(tokens, labels))

            print('=', end='', flush=True)



        # compute metrics

        print("\n")

        print("TP -  TN  -  FP  -  FN")

        print(valid_TP.result().numpy(), valid_TN.result().numpy(), valid_FP.result().numpy(), valid_FN.result().numpy())

        print("val AUC: ", valid_auc.result().numpy())

        print("val loss: ", valid_loss.result().numpy() / VALIDATION_STEPS)



        # Save predictions and weights of model

        if (valid_auc.result().numpy() > best_auc) & (epoch>=PREDICT_START_ITER):

            best_auc = valid_auc.result().numpy()

            print("Prediction on test set - snapshot")

            preds = model.predict(test_dataset, verbose = 1)

            preds_all.append(preds)

            model.save_weights('best_model.h5') # keep track of best model

        # set up next epoch

        epoch = (step+1) // STEPS_PER_EPOCH



        train_auc.reset_states()

        valid_auc.reset_states()



        valid_loss.reset_states()

        train_loss.reset_states()

        

        train_TP.reset_states()

        train_TN.reset_states()

        train_FP.reset_states()

        train_FN.reset_states()

        

        valid_TP.reset_states()

        valid_TN.reset_states()

        valid_FP.reset_states()

        valid_FN.reset_states()

        

        if epoch >= EPOCHS:

            break
#Generate averages of predictions: last one, and average of snapshots

test["toxic_best"] = preds_all[-1]

test["toxic_avg"] = sum(preds_all)/len(preds_all)
# Save the predictions

MODEL_NAME = MODEL.replace("/", "-")

test.to_csv(f"test-{LANG}-{MODEL_NAME}-2-1.csv", index=False)
valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

valid = valid.loc[valid["lang"]==LANG].reset_index(drop=True)

test = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv")



if REPEAT_PL:

    sub = pd.read_csv("../input/multilingual-toxic-comments-training-data/test9500.csv") # use one of earlier subs

    print(sub.head())

    print(len(sub))

    sub["comment_text"] = test["content"]

    sub = sub.loc[test["lang"]==LANG].reset_index(drop=True)

    sub_repeat = pd.concat([sub]*REPEAT_PL, ignore_index=True) # repeat PL multipe times for training

    print('\n', sub_repeat.head())

    print(len(sub_repeat))

    same_cols = ["comment_text", "toxic"]

    valid = pd.concat([valid[same_cols], sub_repeat[same_cols]]).sample(frac=1).reset_index(drop=True)



test = test.loc[test["lang"]==LANG].reset_index(drop=True)  



x_valid = regular_encode(valid.comment_text.values, max_len=MAX_LEN)

y_valid = valid.toxic.values

y_valid = np.asarray(y_valid).astype('float32').reshape((-1,1)) 

x_test = regular_encode(test.content.values, max_len=MAX_LEN)



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(x_test)

    .batch(BATCH_SIZE)

)
if STAGE2:

    # the validation set becomes train_dataset

    train_dataset = (

        tf.data.Dataset

        .from_tensor_slices((x_valid, y_valid)) # replaced by x_valid, y_valid!

        .shuffle(buffer_size=len(x_valid), seed = 18)

        .prefetch(AUTO)

        .batch(BATCH_SIZE)

        .repeat()

    )

    

    # distribute the datset according to the strategy

    train_dist_ds = strategy.experimental_distribute_dataset(train_dataset)
if STAGE2:

    model.load_weights("best_model.h5") # best model from stage1
if STAGE2:

    STEPS_PER_EPOCH = round(x_valid.shape[0] / (BATCH_SIZE*N_ITER_PER_EPOCH))

    print("Steps per epoch:", STEPS_PER_EPOCH)

    EPOCHS = N_EPOCHS*N_ITER_PER_EPOCH

    best_auc = 0

    epoch = 0



    preds_all = []

    for step, (tokens, labels) in enumerate(train_dist_ds):

        # run training step

        strategy.experimental_run_v2(train_step, args=(tokens, labels))

        print('=', end='', flush=True)



        # print metrics training

        if ((step+1) // STEPS_PER_EPOCH) > epoch:

            print("\n Epoch:", epoch)

            print('|', end='', flush=True)

            print("TP -  TN  -  FP  -  FN")

            print(train_TP.result().numpy(), train_TN.result().numpy(), train_FP.result().numpy(), train_FN.result().numpy())

            print("train AUC: ",train_auc.result().numpy())

            print("train loss: ", train_loss.result().numpy() / STEPS_PER_EPOCH)



            # Save predictions and weights of model

            if epoch>=PREDICT_START_ITER:

                print("Prediction on test set - snapshot")

                preds = model.predict(test_dataset, verbose = 1)

                preds_all.append(preds)

                

            # set up next epoch

            epoch = (step+1) // STEPS_PER_EPOCH

            

            train_auc.reset_states()

            train_loss.reset_states()



            train_TP.reset_states()

            train_TN.reset_states()

            train_FP.reset_states()

            train_FN.reset_states()

            

            if epoch >= EPOCHS:

                # save model if needed

                model.save_weights('best_model_valid.h5') 

                break
if STAGE2:

    #Generate averages of snapshot

    test["toxic_mean_snap_valid"] = sum(preds_all)/len(preds_all)

    # Save the predictions

    MODEL_NAME = MODEL.replace("/", "-")

    test_pre = pd.read_csv(f"test-{LANG}-{MODEL_NAME}-2-1.csv")

    assert sum(test_pre["content"]==test["content"]) == len(test)

    test["toxic_best"] = test_pre["toxic_best"]

    test["toxic_avg"] = test_pre["toxic_avg"]

    

    test.to_csv(f"test-{LANG}-{MODEL_NAME}-2.csv", index=False)