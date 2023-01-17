model_name = 't5-large'

tokenizer_name = 't5-large'

freeze = True
import os



import numpy as np

import pandas as pd

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input

from tensorflow.keras.optimizers import Adam

from tensorflow.keras.models import Model, load_model

from tensorflow.keras.callbacks import ModelCheckpoint

from kaggle_datasets import KaggleDatasets

import transformers

from tqdm.notebook import tqdm

from tokenizers import BertWordPieceTokenizer, SentencePieceBPETokenizer
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

GCS_DS_PATH = KaggleDatasets().get_gcs_path('jigsaw-multilingual-toxic-comment-classification')



# Configuration

if model_name == 't5-large':

    EPOCHs = 1

    BATCH_SIZE = 8 * strategy.num_replicas_in_sync

else:

    EPOCHS = 3

    BATCH_SIZE = 8 * strategy.num_replicas_in_sync





MAX_LEN = 128
# First load the real tokenizer

tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name)

if model_name == 't5-large':

    tokenizer.pad_token = tokenizer.eos_token



# Save the loaded tokenizer locally

tokenizer.save_pretrained('.')

# Reload it with the huggingface tokenizers library

# fast_tokenizer = SentencePieceBPETokenizer('vocab.txt')

# fast_tokenizer
train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")

train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")

train2.toxic = train2.toxic.round().astype(int)



valid = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')

sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')

train1.head()
train = pd.concat([

    train1[['comment_text', 'toxic']],

    train2[['comment_text', 'toxic']].query('toxic==1'),

    train2[['comment_text', 'toxic']].query('toxic==0').sample(n=150000, random_state=0)

])

print(train.toxic.value_counts())

print(valid.toxic.value_counts())
olid = pd.read_csv('/kaggle/input/olid2019/olid-training-v1.0.tsv', sep='\t')

olid = olid.rename(columns={"id": "id", "tweet": "comment_text", 'subtask_a': 'toxic'})

olid.toxic = (olid.toxic == 'OFF').astype(int)

# olid['comment_text'] = olid['comment_text'].str.replace('@USER', '')

olid.head()
olid_test = pd.read_csv('/kaggle/input/olid2019/testset-levela.tsv', sep='\t')

olid_test_label = pd.read_csv('/kaggle/input/olid2019/labels-levela.csv', names = ['id', 'label'])
olid_test_label.label.value_counts()
olid_test = olid_test.set_index('id').join(olid_test_label.set_index('id'))
olid_test['toxic'] = (olid_test.label == 'OFF').astype(int).values
olid_test.head()
from sklearn.model_selection import train_test_split

olid_train, olid_valid = train_test_split(olid, test_size=3240, random_state = 2020)

olid_train_1k = olid_train[0:1000]

olid_train_2k = olid_train[0:2000]

olid_train_5k = olid_train[0:5000]

def fast_encode_xlm(texts, tokenizer, chunk_size=256, maxlen=512):

    """

    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras

    """

    all_ids = []

    

    for i in tqdm(range(0, len(texts), chunk_size)):

        text_chunk = texts[i:i+chunk_size].tolist()

        encs = tokenizer.batch_encode_plus(text_chunk, pad_to_max_length = True, max_length = maxlen)

        all_ids.extend(np.array(encs.input_ids))

    

    return np.array(all_ids)
x_train = fast_encode_xlm(train.comment_text.astype(str), tokenizer, maxlen=MAX_LEN)

x_valid = fast_encode_xlm(valid.comment_text.astype(str), tokenizer, maxlen=MAX_LEN)

x_test = fast_encode_xlm(test.content.astype(str), tokenizer, maxlen=MAX_LEN)



y_train = train.toxic.values

y_valid = valid.toxic.values
# olid_encode = fast_encode_xlm(olid.comment_text.astype(str), tokenizer, maxlen=MAX_LEN)

olid_train_encode = fast_encode_xlm(olid_train.comment_text.astype(str), tokenizer, maxlen=MAX_LEN)

olid_train_1k_encode = fast_encode_xlm(olid_train_1k.comment_text.astype(str), tokenizer, maxlen=MAX_LEN)

olid_train_2k_encode = fast_encode_xlm(olid_train_2k.comment_text.astype(str), tokenizer, maxlen=MAX_LEN)

olid_train_5k_encode = fast_encode_xlm(olid_train_5k.comment_text.astype(str), tokenizer, maxlen=MAX_LEN)

olid_valid_encode = fast_encode_xlm(olid_valid.comment_text.astype(str), tokenizer, maxlen=MAX_LEN)

olid_test_encode = fast_encode_xlm(olid_test.tweet.astype(str), tokenizer, maxlen = MAX_LEN)

y_olid_train = olid_train.toxic.values

y_olid_train_1k = olid_train_1k.toxic.values

y_olid_train_2k = olid_train_2k.toxic.values

y_olid_train_5k = olid_train_5k.toxic.values

y_olid_valid = olid_valid.toxic.values

y_olid_test = (olid_test_label.label == 'OFF').astype(int).values
train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((x_train, y_train))

    .repeat()

    .shuffle(2048)

    .batch(BATCH_SIZE)

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
olid_test_dataset = (

    tf.data.Dataset

    .from_tensor_slices(olid_test_encode)

    .batch(BATCH_SIZE)

)

olid_valid_dataset = (

    tf.data.Dataset

    .from_tensor_slices((olid_valid_encode, y_olid_valid))

    .batch(BATCH_SIZE)

)

olid_train_1k_dataset = (

    tf.data.Dataset

    .from_tensor_slices((olid_train_1k_encode, y_olid_train_1k))

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)

olid_train_2k_dataset = (

    tf.data.Dataset

    .from_tensor_slices((olid_train_2k_encode, y_olid_train_2k))

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)

olid_train_5k_dataset = (

    tf.data.Dataset

    .from_tensor_slices((olid_train_5k_encode, y_olid_train_5k))

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)

olid_train_dataset = (

    tf.data.Dataset

    .from_tensor_slices((olid_train_encode, y_olid_train))

    .batch(BATCH_SIZE)

    .prefetch(AUTO)

)

def build_model(transformer, max_len=512):

    """

    https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras

    """

    input_word_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")

    sequence_output = transformer(input_word_ids)[0]

    cls_token = sequence_output[:, 0, :]

    out = Dense(1, activation='sigmoid')(cls_token)

#     run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)



    model = Model(inputs=input_word_ids, outputs=out)

    model.compile(Adam(lr=1e-5), loss='binary_crossentropy', metrics=['accuracy']) #, options = run_opts)

    

    return model
%%time

with strategy.scope():

    transformer_layer = (

        transformers.TFAutoModelWithLMHead.from_pretrained(model_name)

    )

    model = build_model(transformer_layer, max_len=MAX_LEN)

model.summary()

if freeze:

    for layer in model.layers[:-1]:

        layer.trainable = False
n_steps = x_train.shape[0] // BATCH_SIZE

model.fit(

    train_dataset,

    steps_per_epoch=n_steps,

    validation_data=valid_dataset,

    epochs=EPOCHS

)
n_steps = x_valid.shape[0] // BATCH_SIZE

model.fit(

    valid_dataset.repeat(),

    steps_per_epoch=n_steps,

    epochs=EPOCHS

)
sub['toxic'] = model.predict(test_dataset, verbose=0)

sub.to_csv('submission.csv', index=False)
sub.describe()
del train_dataset

del valid_dataset

del test_dataset

del train

del valid

del test

del x_train

del x_valid

del x_test
from sklearn.metrics import roc_auc_score
olid_test['toxic_predict'] = model.predict(olid_test_dataset, verbose=0)

olid_test.to_csv('olid_test_0shot.csv', index=False)



roc_auc_score(y_true = olid_test.toxic, y_score = olid_test.toxic_predict)
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, log_loss, f1_score

import seaborn as sns

import numpy as np

from matplotlib import pyplot as plt



def Find_Optimal_Cutoff(target, predicted):

    """ Find the optimal probability cutoff point for a classification model related to event rate

    Parameters

    ----------

    target : Matrix with dependent or target data, where rows are observations



    predicted : Matrix with predicted data, where rows are observations



    Returns

    -------     

    list type, with optimal cutoff value



    """

    fpr, tpr, threshold = roc_curve(target, predicted)

    i = np.arange(len(tpr)) 

    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})

    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]



    return list(roc_t['threshold']) 



threshold = Find_Optimal_Cutoff(olid_test.toxic, olid_test.toxic_predict)

print("the optimal threshold is " + str(threshold[0]))

olid_test.toxic_predict_binary = [1 if p > threshold[0] else 0 for p in olid_test.toxic_predict]
f1_score(y_true = olid_test.toxic, y_pred = olid_test.toxic_predict_binary)
f1_score(y_true = olid_test.toxic, y_pred = olid_test.toxic_predict_binary, average = 'macro')
f1_score(y_true = olid_test.toxic, y_pred = olid_test.toxic_predict_binary, average = 'weighted')
def plot_matrix(target, predicted_binary, name):

    matrix = confusion_matrix(target, predicted_binary)

    TN, FP, FN, TP = matrix.ravel()

    if (TP + FP > 0) and (TP + FN > 0):

        precision = TP / (TP + FP)

        recall = TP / (TP + FN)

        F =  2 * (precision*recall) / (precision + recall)

    else:

        F = 0

    cm_df = pd.DataFrame(matrix,

                         index = ['Nagative', 'Positive'], 

                         columns = ['Nagative', 'Positive'])

    subtitle = 'Precision ' + str(round(precision, 2)) + ' Recall ' + str(round(recall, 2))

    fig, ax = plt.subplots(figsize=(8,6))

    ax = sns.heatmap(cm_df, annot=True, fmt="d")

    bottom, top = ax.get_ylim()

    ax.set_ylim(bottom + 0.5, top - 0.5)

    plt.title('Confusion Matrix - ' + name + "\n" + subtitle)

    plt.ylabel('True label')

    plt.xlabel('Predicted label')

    plt.show()

plot_matrix(olid_test.toxic, olid_test.toxic_predict_binary, name = 'Zero-Shot')

model.save_weights("/kaggle/working/ckpt.h5")

model.predict(olid_test_dataset, verbose=0)[0:5]
with strategy.scope():

    model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])

    

n_steps = olid_train_1k.shape[0] // BATCH_SIZE

model.fit(

    olid_train_1k_dataset.repeat(),

    steps_per_epoch=n_steps,

    validation_data=olid_valid_dataset,

    callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),

    epochs=10

)



olid_test1k = olid_test

olid_test1k['toxic_predict'] = model.predict(olid_test_dataset, verbose=0)

olid_test1k.to_csv('olid_test_1k.csv', index=False)



print('1k roc is ' + str(roc_auc_score(y_true = olid_test1k.toxic, y_score = olid_test1k.toxic_predict)))



threshold = Find_Optimal_Cutoff(olid_test1k.toxic, olid_test1k.toxic_predict)

print("the optimal threshold is " + str(threshold[0]))

olid_test1k['toxic_predict_binary'] = [1 if p > threshold[0] else 0 for p in olid_test1k.toxic_predict]



print('1k f1-score is ' + str(f1_score(y_true = olid_test1k.toxic, y_pred = olid_test1k.toxic_predict_binary)))

print('1k f1-score weighted is ' + str(f1_score(y_true = olid_test1k.toxic, y_pred = olid_test1k.toxic_predict_binary, average = 'weighted')))

print('1k f1-score macro is ' + str(f1_score(y_true = olid_test1k.toxic, y_pred = olid_test1k.toxic_predict_binary, average = 'macro')))

del olid_train_1k_dataset

del olid_train_1k_encode

del olid_train_1k
tf.tpu.experimental.initialize_tpu_system(tpu)

with strategy.scope():

    model.load_weights("/kaggle/working/ckpt.h5") 

    model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])

model.predict(olid_test_dataset, verbose=0)[0:5]
n_steps = olid_train_2k.shape[0] // BATCH_SIZE

model.fit(

    olid_train_2k_dataset.repeat(),

    steps_per_epoch=n_steps,

    validation_data=olid_valid_dataset,

    callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),

    epochs=10

)



olid_test2k = olid_test

olid_test2k['toxic_predict'] = model.predict(olid_test_dataset, verbose=0)

olid_test2k.to_csv('olid_test_2k.csv', index=False)



print('2k roc is ' + str(roc_auc_score(y_true = olid_test2k.toxic, y_score = olid_test2k.toxic_predict)))



threshold = Find_Optimal_Cutoff(olid_test2k.toxic, olid_test2k.toxic_predict)

print("the optimal threshold is " + str(threshold[0]))

olid_test2k['toxic_predict_binary'] = [1 if p > threshold[0] else 0 for p in olid_test2k.toxic_predict]



print('2k f1-score is ' + str(f1_score(y_true = olid_test2k.toxic, y_pred = olid_test2k.toxic_predict_binary)))

print('2k f1-score weighted is ' + str(f1_score(y_true = olid_test2k.toxic, y_pred = olid_test2k.toxic_predict_binary, average = 'weighted')))

print('2k f1-score macro is ' + str(f1_score(y_true = olid_test2k.toxic, y_pred = olid_test2k.toxic_predict_binary, average = 'macro')))

del olid_train_2k_dataset

del olid_train_2k_encode

del olid_train_2k
tf.tpu.experimental.initialize_tpu_system(tpu)

with strategy.scope():

    model.load_weights("/kaggle/working/ckpt.h5") 

    model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])

model.predict(olid_test_dataset, verbose=0)[0:5]
n_steps = olid_train_5k.shape[0] // BATCH_SIZE 

model.fit(

    olid_train_5k_dataset.repeat(),

    steps_per_epoch=n_steps,

    validation_data=olid_valid_dataset,

    callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),

    epochs=10

)



olid_test5k = olid_test

olid_test5k['toxic_predict'] = model.predict(olid_test_dataset, verbose=0)

olid_test5k.to_csv('olid_test_5k.csv', index=False)



print('5k roc is ' + str(roc_auc_score(y_true = olid_test5k.toxic, y_score = olid_test5k.toxic_predict)))



threshold = Find_Optimal_Cutoff(olid_test5k.toxic, olid_test5k.toxic_predict)

print("the optimal threshold is " + str(threshold[0]))

olid_test5k['toxic_predict_binary'] = [1 if p > threshold[0] else 0 for p in olid_test5k.toxic_predict]



print('5k f1-score is ' + str(f1_score(y_true = olid_test5k.toxic, y_pred = olid_test5k.toxic_predict_binary)))

print('5k f1-score is weighted' + str(f1_score(y_true = olid_test5k.toxic, y_pred = olid_test5k.toxic_predict_binary, average = 'weighted')))

print('5k f1-score is macro' + str(f1_score(y_true = olid_test5k.toxic, y_pred = olid_test5k.toxic_predict_binary, average = 'macro')))
del olid_train_5k_dataset

del olid_train_5k_encode

del olid_train_5k
tf.tpu.experimental.initialize_tpu_system(tpu)



with strategy.scope():

    model.load_weights("/kaggle/working/ckpt.h5") 

    model.compile(Adam(lr=2e-6), loss='binary_crossentropy', metrics=['accuracy'])

    

model.predict(olid_test_dataset, verbose=0)[0:5]
n_steps = olid_train.shape[0] // BATCH_SIZE

model.fit(

    olid_train_dataset.repeat(),

    steps_per_epoch=n_steps,

    validation_data=olid_valid_dataset,

    callbacks = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),

    epochs=10

)



olid_test10k = olid_test

olid_test10k['toxic_predict'] = model.predict(olid_test_dataset, verbose=0)

olid_test10k.to_csv('olid_test_10k.csv', index=False)



print('10k roc is ' + str(roc_auc_score(y_true = olid_test10k.toxic, y_score = olid_test10k.toxic_predict)))



threshold = Find_Optimal_Cutoff(olid_test10k.toxic, olid_test10k.toxic_predict)

print("the optimal threshold is " + str(threshold[0]))

olid_test10k['toxic_predict_binary'] = [1 if p > threshold[0] else 0 for p in olid_test10k.toxic_predict]



print('10k f1-score is ' + str(f1_score(y_true = olid_test10k.toxic, y_pred = olid_test10k.toxic_predict_binary)))

print('10k f1-score weighted is ' + str(f1_score(y_true = olid_test10k.toxic, y_pred = olid_test10k.toxic_predict_binary, average = 'weighted')))

print('10k f1-score is macro' + str(f1_score(y_true = olid_test10k.toxic, y_pred = olid_test10k.toxic_predict_binary, average = 'macro')))

del olid_train_10k_dataset

del olid_train_10k_encode

del olid_train_10k