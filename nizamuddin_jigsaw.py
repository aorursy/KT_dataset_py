import os



import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization, Activation

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model

from tensorflow.keras.callbacks import ModelCheckpoint

from kaggle_datasets import KaggleDatasets

import matplotlib.pyplot as plt

import seaborn as sns

#import transformers

from transformers import TFAutoModel, AutoTokenizer

from transformers import (

    AdamW, get_linear_schedule_with_warmup, get_constant_schedule, 

    XLMRobertaTokenizer, XLMRobertaModel, XLMRobertaConfig,

)

from tqdm.notebook import tqdm

from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
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
AUTO = tf.data.experimental.AUTOTUNE



# Data access

#GCS_DS_PATH = KaggleDatasets().get_gcs_path()



# Configuration

EPOCHS = 5

BATCH_SIZE = 32 * strategy.num_replicas_in_sync

MAX_LEN = 128  #------------------------------------ changed

publictrain = pd.read_csv("/kaggle/input/jigsaw-public-baseline-train-data/train_data.csv")

#stratified sampling to get good proportion of data: 2*sample_size from each language class

# Why 2?: 1 for each lang + 1 for each of the two toxic classes

sample_size = 20000

df_train = publictrain.groupby(['lang','toxic'], group_keys=False).apply(lambda x: x.sample(min(len(x),sample_size)))

df_train = df_train.sample(frac=1).reset_index(drop=True)#shuffling

df_train.shape
# cols = list(set(train2.columns).intersection(train1.columns))

# #['id', 'comment_text', 'toxic', 'obscene', 'threat','insult']#common

# print("for toxic comment competition 2018")

# print(train1[cols].isna().sum()/len(train1))

# print("-"*20)

# print("for unintended bias competition 2019")

# print(train2[cols].isna().sum()/len(train2))
%%time

# train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv", usecols=["comment_text", "toxic"])

# train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv", usecols=["comment_text", "toxic"])

# train2.toxic = train2.toxic.round().astype(int)

# df_train = pd.concat([

#     train1[['comment_text', 'toxic']],

#     train2[['comment_text', 'toxic']].query('toxic==1'),

#     train2[['comment_text', 'toxic']].query('toxic==0').sample(n=99937, random_state=0),])

# df_train = df_train.sample(frac=1).reset_index(drop=True)#shuffling

# import gc

# del train1, train2

# gc.collect(); gc.collect();

# print(df_train.shape, df_valid.shape)

# gc.collect(); gc.collect(); gc.collect();





df_valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

df_valid = df_valid.sample(frac=1).reset_index(drop=True)#shuffling

test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')

sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
%%time

from joblib import Parallel, delayed

tokenizer = AutoTokenizer.from_pretrained('jplu/tf-xlm-roberta-base')



def regular_encode(row, is_test=False, maxlen=MAX_LEN):

    outp = None

    if is_test:#for test data

        enc_di = tokenizer.encode_plus(

            str(row),#row:text

            return_attention_masks=False, 

            return_token_type_ids=False,

            pad_to_max_length=True,

            max_length=maxlen

        )

        outp = np.array(enc_di['input_ids'])

    else:#for validation/train data

        enc_di = tokenizer.encode_plus(

            str(row[0]),#row:(text,label)

            return_attention_masks=False, 

            return_token_type_ids=False,

            pad_to_max_length=True,

            max_length=maxlen

        )

        outp = np.array(enc_di['input_ids']), row[1]

    return outp

                        

rows = zip(df_train['comment_text'].values.tolist(), df_train.toxic.values.tolist())

train = Parallel(n_jobs=4, backend='multiprocessing')(delayed(regular_encode)(row) for row in tqdm(rows))



rows = zip(df_valid['comment_text'].values.tolist(), df_valid.toxic.values.tolist())

valid = Parallel(n_jobs=4, backend='multiprocessing')(delayed(regular_encode)(row) for row in tqdm(rows))

                        

rows = test.content.values.tolist()

x_test = Parallel(n_jobs=4, backend='multiprocessing')(delayed(regular_encode)(row,is_test=True) for row in tqdm(rows))



x_train = np.vstack(np.array(train)[:,0])

y_train = np.array(train)[:,1].astype(np.int32)

x_valid = np.vstack(np.array(valid)[:,0])

y_valid = np.array(valid)[:,1].astype(np.int32)

x_train.shape,y_train.shape,x_valid.shape,y_valid.shape
# train_dataset = (

#     tf.data.Dataset

#     .from_tensor_slices((x_train,y_train))

#     .repeat()

#     .shuffle(2048)

#     .batch(BATCH_SIZE)

#     .prefetch(AUTO)

# )#tensorflow.python.data.ops.dataset_ops.PrefetchDataset



# valid_dataset = (

#     tf.data.Dataset

#     .from_tensor_slices((x_valid,y_valid))

#     .batch(BATCH_SIZE)

#     .cache()

#     .prefetch(AUTO)

# )#tensorflow.python.data.ops.dataset_ops.PrefetchDataset



test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(x_test)

    .batch(BATCH_SIZE)

)#tensorflow.python.data.ops.dataset_ops.PrefetchDataset
from keras.utils.generic_utils import get_custom_objects

def gelu(x):

    """Gaussian Error Linear Unit.

    This is a smoother version of the RELU.

    Original paper: https://arxiv.org/abs/1606.08415

    Args:

        x: float Tensor to perform activation.

    Returns:

        `x` with the GELU activation applied.

    """

    cdf = 0.5 * (1.0 + tf.tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3)))))

    return x * cdf#`x` with the GELU activation applied

get_custom_objects().update({'gelu': Activation(gelu)})
def build_model(transformer, max_len=512):

    """

    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras

    """

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    sequence_output = transformer(input_word_ids)[0]

    cls_token = sequence_output[:, 0, :]#cls_token is a vector of length 768 marginalised against other 2 dimensions

    x = Dense(16, activation=gelu)(cls_token)

    x = BatchNormalization()(x)

    x = Dropout(0.3)(x)

    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=input_word_ids, outputs=out)

    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy'])

    return model

%%time



with strategy.scope():

    transformer_layer = TFAutoModel.from_pretrained('jplu/tf-xlm-roberta-base')

    model = build_model(transformer_layer, max_len=MAX_LEN)

model.summary()
K = 5
def train(X_train,y_train):

    oof_predictions = []

    from sklearn.model_selection import KFold

    from tensorflow.keras.callbacks import LearningRateScheduler

    import math

    kf = KFold(n_splits=K, random_state=1, shuffle=True)

    lr_schedule = LearningRateScheduler(lambda epoch: 0.001 * math.pow(0.001, math.floor((1+epoch)/3.0)))

    

    for ind, (tr, val) in enumerate(kf.split(X_train)):

        X_tr = X_train[tr]

        y_tr = y_train[tr]

        X_vl = X_train[val]

        y_vl = y_train[val]

        print(X_tr.shape,y_tr.shape,X_vl.shape,y_vl.shape)

        

        train_dataset = (

            tf.data.Dataset

            .from_tensor_slices((X_tr,y_tr))

            .repeat()

            .shuffle(2048)

            .batch(BATCH_SIZE)

            .prefetch(AUTO)

        )



        valid_dataset = (

            tf.data.Dataset

            .from_tensor_slices((X_vl,y_vl))

            .batch(BATCH_SIZE)

            .cache()

            .prefetch(AUTO)

        )

        

        n_steps = X_tr.shape[0] // BATCH_SIZE

        train_history = model.fit(

                        train_dataset,

                        steps_per_epoch=n_steps,

                        validation_data=valid_dataset,

                        epochs=EPOCHS,

                        verbose=True, 

                        callbacks=[lr_schedule]

        )

      

        print("Done training! Now predicting")

        oof_predictions.append(model.predict(test_dataset, verbose=1))

    return oof_predictions



oof_predictions = train(x_train,y_train)

avged = sum(oof_predictions)/float(K)
sub['toxic'] = avged#model.predict(test_dataset, verbose=1)

sub.to_csv('submission.csv', index=False)

sub.toxic.hist(bins=100)
#n_steps = x_train.shape[0] // BATCH_SIZE

# train_history = model.fit(

#                         train_dataset,

#                         steps_per_epoch=n_steps,

#                         validation_data=valid_dataset,

#                         epochs=10

#                 )

#--------------------------------------------------------------------------------

# n_steps = x_valid.shape[0] // BATCH_SIZE  #since generator is used

# train_history_2 = model.fit(

#     valid_dataset.repeat(),

#     steps_per_epoch=n_steps,

#     epochs=10

# )
# """

# ------------------------------------------

# REF: https://github.com/optuna/optuna/blob/master/examples/pruning/tfkeras_integration.py

# -----------------------------------------

# Optuna example that demonstrates a pruner for tf.keras.

# In this example, we optimize the validation accuracy of hand-written digit recognition

# using tf.keras and MNIST, where the architecture of the neural network

# and the parameters of optimizer are optimized.

# Throughout the training of neural networks,

# a pruner observes intermediate results and stops unpromising trials.

# """



# import tensorflow as tf

# import tensorflow_datasets as tfds



# import optuna

# from optuna.integration import TFKerasPruningCallback





# BATCHSIZE = 128

# CLASSES = 10

# EPOCHS = 20

# N_TRAIN_EXAMPLES = 3000

# STEPS_PER_EPOCH = int(N_TRAIN_EXAMPLES / BATCHSIZE / 10)

# VALIDATION_STEPS = 30





# def train_dataset():



#     ds = tfds.load("mnist", split=tfds.Split.TRAIN, shuffle_files=True)

#     ds = ds.map(lambda x: (tf.cast(x["image"], tf.float32) / 255.0, x["label"]))

#     ds = ds.repeat().shuffle(1024).batch(BATCHSIZE)

#     ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

#     return ds





# def eval_dataset():



#     ds = tfds.load("mnist", split=tfds.Split.TEST, shuffle_files=False)

#     ds = ds.map(lambda x: (tf.cast(x["image"], tf.float32) / 255.0, x["label"]))

#     ds = ds.repeat().batch(BATCHSIZE)

#     ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

#     return ds





# def create_model(trial):



#     # Hyperparameters to be tuned by Optuna.

#     lr = trial.suggest_loguniform("lr", 1e-4, 1e-1)

#     momentum = trial.suggest_uniform("momentum", 0.0, 1.0)

#     units = trial.suggest_categorical("units", [32, 64, 128, 256, 512])

#     '''

#     lr: 0.09120307278561411

#     momentum: 0.9124601201218243

#     units: 256

#     '''

#     # Compose neural network with one hidden layer.

#     model = tf.keras.Sequential()

#     model.add(tf.keras.layers.Flatten())

#     model.add(tf.keras.layers.Dense(units=units, activation=tf.nn.relu))

#     model.add(tf.keras.layers.Dense(CLASSES, activation=tf.nn.softmax))



#     # Compile model.

#     model.compile(

#         optimizer=tf.keras.optimizers.SGD(lr=lr, momentum=momentum, nesterov=True),

#         loss="sparse_categorical_crossentropy",

#         metrics=["accuracy"],

#     )



#     return model





# def objective(trial):

#     # Clear clutter from previous TensorFlow graphs.

#     tf.keras.backend.clear_session()



#     # Metrics to be monitored by Optuna.

#     if tf.__version__ >= "2":

#         monitor = "val_accuracy"

#     else:

#         monitor = "val_acc"



#     # Create tf.keras model instance.

#     model = create_model(trial)



#     # Create dataset instance.

#     ds_train = train_dataset()

#     ds_eval = eval_dataset()



#     # Create callbacks for early stopping and pruning.

#     callbacks = [

#         tf.keras.callbacks.EarlyStopping(patience=3),

#         TFKerasPruningCallback(trial, monitor),

#     ]



#     # Train model.

#     history = model.fit(

#         ds_train,

#         epochs=EPOCHS,

#         steps_per_epoch=STEPS_PER_EPOCH,

#         validation_data=ds_eval,

#         validation_steps=VALIDATION_STEPS,

#         callbacks=callbacks,

#     )



#     # TODO(@sfujiwara): Investigate why the logger here is called twice.

#     # tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)

#     # tf.compat.v1.logging.info('hello optuna')



#     return history.history[monitor][-1]





# def show_result(study):



#     pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]

#     complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]



#     print("Study statistics: ")

#     print("  Number of finished trials: ", len(study.trials))

#     print("  Number of pruned trials: ", len(pruned_trials))

#     print("  Number of complete trials: ", len(complete_trials))



#     print("Best trial:")

#     trial = study.best_trial



#     print("  Value: ", trial.value)



#     print("  Params: ")

#     for key, value in trial.params.items():

#         print("    {}: {}".format(key, value))





# def main():



#     study = optuna.create_study(

#         direction="maximize", pruner=optuna.pruners.MedianPruner(n_startup_trials=2)

#     )



#     study.optimize(objective, n_trials=25, timeout=600)



#     show_result(study)





# if __name__ == "__main__":

#     main()
# import json 

# import requests 

# api_key = ''

# url = ('https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze' + '?key=' + api_key)

# data_dict = {

#     'comment': {'text': 'what kind of idiot name is foo?'},

#     'languages': ['en'],

#     'requestedAttributes': {'TOXICITY': {}}

# }

# response = requests.post(url=url, data=json.dumps(data_dict)) 

# response_dict = json.loads(response.content) 

# print(json.dumps(response_dict, indent=2))