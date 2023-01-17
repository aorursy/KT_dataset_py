MAX_LEN = 192  #Reduced for quicker execution

LR = 1e-5

BATCH_SIZE = 16 # per TPU core

TOTAL_STEPS_STAGE1 = 300

VALIDATE_EVERY_STAGE1 = 100

TOTAL_STEPS_STAGE2 = 200

VALIDATE_EVERY_STAGE2 = 100



PRETRAINED_MODEL = 'jplu/tf-xlm-roberta-large'

D = '/kaggle/input/toxic-comment-classification/'



import os

import numpy as np

import pandas as pd

from sklearn.metrics import roc_auc_score

import tensorflow as tf

print(tf.__version__)

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.models import Model

import transformers

from transformers import TFAutoModel, AutoTokenizer

import logging

# no extensive logging 

logging.getLogger().setLevel(logging.NOTSET)



AUTO = tf.data.experimental.AUTOTUNE
def connect_to_TPU():

    """Detect hardware, return appropriate distribution strategy"""

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



    global_batch_size = BATCH_SIZE * strategy.num_replicas_in_sync



    return tpu, strategy, global_batch_size





tpu, strategy, global_batch_size = connect_to_TPU()

print("REPLICAS: ", strategy.num_replicas_in_sync)
train_df = pd.read_csv(D+'train.csv')

val_df = pd.read_csv(D+'validation.csv')

test_df = pd.read_csv(D+'test.csv')

# sub_df = pd.read_csv(D+'sample_submission.csv')



# subsample the train dataframe to 50%-50%

train_df = pd.concat([

    train_df.query('toxic==1'),

    train_df.query('toxic==0').sample(sum(train_df.toxic),random_state=42)

])

# shufle it just to make sure

train_df = train_df.sample(frac=1, random_state = 42)
%%time



def regular_encode(texts, tokenizer, maxlen=512):

    enc_di = tokenizer.batch_encode_plus(

        texts, 

        return_attention_masks=False, 

        return_token_type_ids=False,

        pad_to_max_length=True,

        max_length=maxlen

    )

    

    return np.array(enc_di['input_ids'])

    



tokenizer = AutoTokenizer.from_pretrained(PRETRAINED_MODEL)

X_train = regular_encode(train_df.comment_text.values, tokenizer, maxlen=MAX_LEN)

X_val = regular_encode(val_df.comment_text.values, tokenizer, maxlen=MAX_LEN)

X_test = regular_encode(test_df.comment_text.values, tokenizer, maxlen=MAX_LEN)



y_train = train_df.toxic.values.reshape(-1,1)

y_val = val_df.toxic.values.reshape(-1,1)
def create_dist_dataset(X, y=None, training=False):

    dataset = tf.data.Dataset.from_tensor_slices(X)



    ### Add y if present ###

    if y is not None:

        dataset_y = tf.data.Dataset.from_tensor_slices(y)

        dataset = tf.data.Dataset.zip((dataset, dataset_y))

        

    ### Repeat if training ###

    if training:

        dataset = dataset.shuffle(len(X)).repeat()



    dataset = dataset.batch(global_batch_size).prefetch(AUTO)



    ### make it distributed  ###

    dist_dataset = strategy.experimental_distribute_dataset(dataset)



    return dist_dataset

    

    

train_dist_dataset = create_dist_dataset(X_train, y_train, True)

val_dist_dataset   = create_dist_dataset(X_val)

test_dist_dataset  = create_dist_dataset(X_test)
%%time



def create_model_and_optimizer():

    with strategy.scope():

        transformer_layer = TFAutoModel.from_pretrained(PRETRAINED_MODEL)                

        model = build_model(transformer_layer)

        optimizer = tf.keras.optimizers.Adam(learning_rate=LR, epsilon=1e-08)

    return model, optimizer





def build_model(transformer):

    inp = Input(shape=(MAX_LEN,), dtype=tf.int32, name="input_word_ids")

    # Huggingface transformers have multiple outputs, embeddings are the first one

    # let's slice out the first position, the paper says its not worse than pooling

    x = transformer(inp)[0][:, 0, :]  

    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[inp], outputs=[out])

    

    return model





model, optimizer = create_model_and_optimizer()

model.summary()
def define_losses_and_metrics():

    with strategy.scope():

        loss_object = tf.keras.losses.BinaryCrossentropy(

            reduction=tf.keras.losses.Reduction.NONE, from_logits=False)



        def compute_loss(labels, predictions):

            per_example_loss = loss_object(labels, predictions)

            loss = tf.nn.compute_average_loss(

                per_example_loss, global_batch_size = global_batch_size)

            return loss



        train_accuracy_metric = tf.keras.metrics.AUC(name='training_AUC')



    return compute_loss, train_accuracy_metric







def train(train_dist_dataset, val_dist_dataset=None, y_val=None,

          total_steps=5000, validate_every=500):

    step = 0

    ### Training lopp ###

    for tensor in train_dist_dataset:

        distributed_train_step(tensor) 

        step+=1



        if (step % validate_every == 0):   

            ### Print train metrics ###  

            train_metric = train_accuracy_metric.result().numpy()

            print("Step %d, train AUC: %.5f" % (step, train_metric))   

            

            ### Test loop with exact AUC ###

            if val_dist_dataset:

                val_metric = roc_auc_score(y_val, predict(val_dist_dataset))

                print("     validation AUC: %.5f" %  val_metric)   



            ### Reset (train) metrics ###

            train_accuracy_metric.reset_states()

            

        if step  == total_steps:

            break







@tf.function

def distributed_train_step(data):

    strategy.experimental_run_v2(train_step, args=(data,))



def train_step(inputs):

    features, labels = inputs



    with tf.GradientTape() as tape:

        predictions = model(features, training=True)

        loss = compute_loss(labels, predictions)



    gradients = tape.gradient(loss, model.trainable_variables)

    optimizer.apply_gradients(zip(gradients, model.trainable_variables))



    train_accuracy_metric.update_state(labels, predictions)









def predict(dataset):  

    predictions = []

    for tensor in dataset:

        predictions.append(distributed_prediction_step(tensor))

    ### stack replicas and batches

    predictions = np.vstack(list(map(np.vstack,predictions)))

    return predictions



@tf.function

def distributed_prediction_step(data):

    predictions = strategy.experimental_run_v2(prediction_step, args=(data,))

    return strategy.experimental_local_results(predictions)



def prediction_step(inputs):

    features = inputs  # note datasets used in prediction do not have labels

    predictions = model(features, training=False)

    return predictions





compute_loss, train_accuracy_metric = define_losses_and_metrics()
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# filepath = "roberta_model.h5"

# checkpoint = ModelCheckpoint(filepath, monitor='val_auc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')



# early = EarlyStopping(monitor='val_auc', mode='max', patience=5)



# reduce_lr = ReduceLROnPlateau(monitor='val_auc', factor=0.1, patience=2, min_lr=1e-5)

# n_steps = X_train.shape[0] // BATCH_SIZE

# train_history = model.fit(

#     train_dist_dataset,

#     steps_per_epoch=n_steps,

#     validation_data=val_dist_dataset,

#     epochs=10,

#     callbacks=[checkpoint, early, reduce_lr]

# )
%%time

train(train_dist_dataset, val_dist_dataset, y_val,

      TOTAL_STEPS_STAGE1, VALIDATE_EVERY_STAGE1)
%%time

# make a new dataset for training with the validation data 

# with targets, shuffling and repeating

val_dist_dataset_4_training = create_dist_dataset(X_val, y_val, training=True)



# train again

train(val_dist_dataset_4_training,

      total_steps = TOTAL_STEPS_STAGE2, 

      validate_every = VALIDATE_EVERY_STAGE2)  # not validating but printing now
# %%time

# sub_df['toxic'] = predict(test_dist_dataset)[:,0]

# sub_df.to_csv('submission.csv', index=False)
# from sklearn.metrics import accuracy_score

# def roc_auc(predictions,target):

#     fpr, tpr, thresholds = metrics.roc_curve(target, predictions)

#     print(len(fpr),len(tpr))

#     roc_auc = metrics.auc(fpr, tpr)

#     return roc_auc

# scores_model=[]

# scores = bi_model.predict(xvalid_pad)

# score=[int(i>=0.5) for i in scores]

# scores_model.append({'Model': 'Transforer','AUC_Score': roc_auc(scores[:m],yvalid),'Accuracy':accuracy_score(yvalid,score[:m])})

# print("Auc: %.2f%%" % (roc_auc(scores[:m],yvalid)))