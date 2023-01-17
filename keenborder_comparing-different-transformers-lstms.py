# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import tensorflow_hub as hub
from tqdm import tqdm
from tqdm import tqdm_notebook
from sklearn.metrics import auc
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import AutoTokenizer,BertTokenizer,TFBertModel,TFOpenAIGPTModel,OpenAIGPTTokenizer,DistilBertTokenizer, TFDistilBertModel,XLMTokenizer, TFXLMModel
from transformers import TFAutoModel, AutoTokenizer
from kaggle_datasets import KaggleDatasets
from sklearn.metrics import roc_curve,confusion_matrix,auc
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import matplotlib as mpl


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.initializers import Constant
import re

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
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

train1 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv")
train2 = pd.read_csv("/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv")
train2.toxic = train2.toxic.round().astype(int)

valid_raw = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')
test_raw = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')
sub = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/sample_submission.csv')
# Combine train1 with a subset of train2
train_raw = pd.concat([
    train1[['comment_text', 'toxic']],
    train2[['comment_text', 'toxic']].query('toxic==1'),
    train2[['comment_text', 'toxic']].query('toxic==0').sample(n=100000, random_state=0)
])
valid_raw['toxic'].value_counts().plot(kind='bar')
train_raw['toxic'].value_counts().plot(kind='bar')
neg, pos = np.bincount(train_raw['toxic'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))
# First load the real tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
EPOCHS=2
LEARNING_RATE=1e-5
early_stopping=early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)
AUTO = tf.data.experimental.AUTOTUNE
GCS_DS_PATH = KaggleDatasets().get_gcs_path()
max_seq_length = 192


def single_encoding_function(text,tokenizer,name='BERT'):
    input_ids=[]
    if name=='BERT':
        tokenizer.pad_token ='[PAD]'
    elif name=='OPENAIGPT2':
        tokenizer.pad_token='<unk>'
    elif name=='Transformer XL':
        print(tokenizer.eos_token)
        tokenizer.pad_token= tokenizer.eos_token
    elif name=='DistilBert':
        tokenizer.pad_token='[PAD]'
    
    for sentence in tqdm(text):
        encoded=tokenizer.encode(sentence,max_length=max_seq_length,pad_to_max_length=True)
        input_ids.append(encoded)
    return input_ids

X_train=np.array(single_encoding_function(train_raw['comment_text'].values.tolist(),tokenizer,name="BERT"))
y_train=np.array(train_raw['toxic'])
X_valid=np.array(single_encoding_function(valid_raw['comment_text'].values.tolist(),tokenizer,name="BERT"))
y_valid=np.array(valid_raw['toxic'])
X_test=np.array(single_encoding_function(test_raw['content'].values.tolist(),tokenizer,name="BERT"))
steps_per_epoch = X_train.shape[0] // BATCH_SIZE
def make_data():
    train = (
        tf.data.Dataset
        .from_tensor_slices((X_train, y_train))
        .repeat()
        .shuffle(2048)
        .batch(BATCH_SIZE)
        .prefetch(AUTO))

    valid = (
        tf.data.Dataset
        .from_tensor_slices((X_valid, y_valid))
        .batch(BATCH_SIZE)
        .cache()
        .prefetch(AUTO)
    )

    test = (
        tf.data.Dataset
        .from_tensor_slices(X_test)
        .batch(BATCH_SIZE)
    )
    return train,valid,test
train,valid,test=make_data()
def build_model(transformer_layer,max_len=max_seq_length):
    input_word_ids = tf.keras.layers.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    sequence_output = transformer_layer(input_word_ids)[0]
    
    cls_token = sequence_output[:, 0, :]
    out = tf.keras.layers.Dense(1, activation='sigmoid')(cls_token)
    
    model = tf.keras.Model(inputs=input_word_ids, outputs=out)
    
    
    return model
mpl.rcParams['figure.figsize'] = (12, 10)
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

def plot_loss(history):
# Use a log scale to show the wide range of values.
    plt.semilogy(history.epoch,  history.history['loss'],
               color='red', label='Train Loss')
    plt.semilogy(history.epoch,  history.history['val_loss'],
          color='green', label='Val Loss',
          linestyle="--")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
  
    plt.legend()
    
    
def plot_metrics(history):
    metrics =  ['loss', 'auc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch,  history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
                 color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
        if metric == 'loss':
            plt.ylim([0, plt.ylim()[1]])
        elif metric == 'auc':
            plt.ylim([0.8,1])
        else:
            plt.ylim([0,1])

        plt.legend()

def plot_cm(y_true, y_pred, title):
    ''''
    input y_true-Ground Truth Labels
          y_pred-Predicted Value of Model
          title-What Title to give to the confusion matrix
    
    Draws a Confusion Matrix for better understanding of how the model is working
    
    return None
    
    '''
    
    figsize=(10,10)
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    plt.title(title)
    sns.heatmap(cm, cmap= "YlGnBu", annot=annot, fmt='', ax=ax)

def roc_curve_plot(fpr,tpr,roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' %roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

        
%%time
def compile_model(name):
    with strategy.scope():
        METRICS = [
          tf.keras.metrics.TruePositives(name='tp'),
          tf.keras.metrics.FalsePositives(name='fp'),
          tf.keras.metrics.TrueNegatives(name='tn'),
          tf.keras.metrics.FalseNegatives(name='fn'), 
          tf.keras.metrics.BinaryAccuracy(name='accuracy'),
          tf.keras.metrics.Precision(name='precision'),
          tf.keras.metrics.Recall(name='recall'),
          tf.keras.metrics.AUC(name='auc')]
        if name=='bert-base-uncased':
            transformer_layer = (
                TFBertModel.from_pretrained(name)
            )
        elif name=='openai-gpt':
            transformer_layer = (
                TFOpenAIGPTModel.from_pretrained(name)
            )
        elif name=='distilbert-base-cased':
            transformer_layer = (
                TFDistilBertModel.from_pretrained(name)
            )
        elif name=='xlm-mlm-en-2048':
            transformer_layer = (
                TFBertModel.from_pretrained(name)
            )
        elif name=='jplu/tf-xlm-roberta-large':
            transformer_layer = (
                TFAutoModel.from_pretrained(name)
            )
        model = build_model(transformer_layer, max_len=max_seq_length)
        model.compile(optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE), loss='binary_crossentropy', metrics=METRICS)
    return model

steps_per_epoch=X_train.shape[0]//BATCH_SIZE

model=compile_model('bert-base-uncased')
print(model.summary())
history=model.fit(
    train,steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,callbacks=[early_stopping], validation_data=valid
)
plot_loss(history)
plot_metrics(history)
y_predict=model.predict(valid, verbose=1)
y_predict[ y_predict> 0.5] = 1
y_predict[y_predict <= 0.5] = 0
plot_cm(y_valid, y_predict, 'BERT-Confusion Matrix')

y_predict_prob=model.predict(valid, verbose=1)
fpr, tpr, _ = roc_curve(y_valid,y_predict_prob)
roc_auc = auc(fpr, tpr)
roc_curve_plot(fpr,tpr,roc_auc)
# # First load the real tokenizer
tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
X_train=np.array(single_encoding_function(train_raw['comment_text'],tokenizer,'OPENAIGPT2'))
y_train=np.array(train_raw['toxic'])
X_valid=np.array(single_encoding_function(valid_raw['comment_text'],tokenizer,'OPENAIGPT2'))
y_valid=np.array(valid_raw['toxic'])
X_test=np.array(single_encoding_function(test_raw['content'],tokenizer,'OPENAIGPT2'))
steps_per_epoch = X_train.shape[0] // BATCH_SIZE
train,valid,test=make_data()

model=compile_model('openai-gpt')
print(model.summary())

history=model.fit(
    train,steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,callbacks=[early_stopping], validation_data=valid
)
plot_loss(history)
plot_metrics(history)
y_predict=model.predict(valid, verbose=1)
y_predict[ y_predict> 0.5] = 1
y_predict[y_predict <= 0.5] = 0
plot_cm(y_valid, y_predict, 'OpenAIGPT-Confusion Matrix')
y_predict_prob=model.predict(valid, verbose=1)
fpr, tpr, _ = roc_curve(y_valid,y_predict_prob)
roc_auc = auc(fpr, tpr)
roc_curve_plot(fpr,tpr,roc_auc)
# # First load the real tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
X_train=np.array(single_encoding_function(train_raw['comment_text'],tokenizer,'DistilBert'))
y_train=np.array(train_raw['toxic'])
X_valid=np.array(single_encoding_function(valid_raw['comment_text'],tokenizer,'DistilBert'))
y_valid=np.array(valid_raw['toxic'])
X_test=np.array(single_encoding_function(test_raw['content'],tokenizer,'DistilBert'))
train,valid,test=make_data()
steps_per_epoch = X_train.shape[0] // BATCH_SIZE

model=compile_model('distilbert-base-cased')
print(model.summary())

history=model.fit(
    train,steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,callbacks=[early_stopping], validation_data=valid
)
plot_loss(history)
plot_metrics(history)
y_predict=model.predict(valid, verbose=1)
y_predict[ y_predict> 0.5] = 1
y_predict[y_predict <= 0.5] = 0
plot_cm(y_valid, y_predict, 'Transformer XL Performance-Confusion Matrix')
y_predict_prob=model.predict(valid, verbose=1)
fpr, tpr, _ = roc_curve(y_valid,y_predict_prob)
roc_auc = auc(fpr, tpr)
roc_curve_plot(fpr,tpr,roc_auc)
# # First load the real tokenizer
tokenizer = XLMTokenizer.from_pretrained('xlm-mlm-en-2048')
X_train=np.array(single_encoding_function(train_raw['comment_text'],tokenizer,'XLM'))
y_train=np.array(train_raw['toxic'])
X_valid=np.array(single_encoding_function(valid_raw['comment_text'],tokenizer,'XLM'))
y_valid=np.array(valid_raw['toxic'])
X_test=np.array(single_encoding_function(test_raw['content'],tokenizer,'XLM'))

train,valid,test=make_data()

model=compile_model('xlm-mlm-en-2048')
print(model.summary())

history=model.fit(
    train,steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,callbacks=[early_stopping], validation_data=valid
)
plot_loss(history)
plot_metrics(history)
y_predict=model.predict(valid, verbose=1)
y_predict[ y_predict> 0.5] = 1
y_predict[y_predict <= 0.5] = 0
plot_cm(y_valid, y_predict, 'XLM-Confusion Matrix')
y_predict_prob=model.predict(valid, verbose=1)
fpr, tpr, _ = roc_curve(y_valid,y_predict_prob)
roc_auc = auc(fpr, tpr)
roc_curve_plot(fpr,tpr,roc_auc)
from IPython.display import YouTubeVideo

YouTubeVideo("Ot6A3UFY72c", width=800, height=300)



def regular_encode(texts, tokenizer, maxlen=512):
    enc_di = tokenizer.batch_encode_plus(
        texts, 
        return_attention_masks=False, 
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen
    )
    
    return np.array(enc_di['input_ids'])
tokenizer = AutoTokenizer.from_pretrained('jplu/tf-xlm-roberta-large')
%%time 

X_train = regular_encode(train_raw.comment_text.values, tokenizer, maxlen=max_seq_length)
X_valid = regular_encode(valid_raw.comment_text.values, tokenizer, maxlen=max_seq_length)
X_test = regular_encode(test_raw.content.values, tokenizer, maxlen=max_seq_length)

y_train = train_raw.toxic.values
y_valid = valid_raw.toxic.values
steps_per_epoch = X_train.shape[0] // BATCH_SIZE
train,valid,test=make_data()

final_model=compile_model('jplu/tf-xlm-roberta-large')
print(final_model.summary())

history=final_model.fit(
    train,steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS,callbacks=[early_stopping], validation_data=valid
)
plot_loss(history)
plot_metrics(history)
y_predict=final_model.predict(valid, verbose=1)
y_predict[ y_predict> 0.5] = 1
y_predict[y_predict <= 0.5] = 0
plot_cm(y_valid, y_predict, 'XLM-Roberta-Confusion Matrix')
y_predict_prob=final_model.predict(valid, verbose=1)
fpr, tpr, _ = roc_curve(y_valid,y_predict_prob)
roc_auc = auc(fpr, tpr)
roc_curve_plot(fpr,tpr,roc_auc)
steps_per_epoch = X_valid.shape[0] // BATCH_SIZE
train_history_2 = final_model.fit(
    valid.repeat(),
    steps_per_epoch=steps_per_epoch,
    epochs=EPOCHS
)
max_seq_length = 512
embedding_dim=300
BATCH_SIZE = 32
EPOCHS=1
LEARNING_RATE=1e-5
early_stopping=early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    verbose=1,
    patience=2,
    mode='max',
    restore_best_weights=True)
AUTO = tf.data.experimental.AUTOTUNE

tokenizer = Tokenizer(split=' ', oov_token='<unw>', filters=' ')
tokenizer.fit_on_texts(train_raw['comment_text'].values)

# this takes our sentences and replaces each word with an integer
X_train = tokenizer.texts_to_sequences(train_raw['comment_text'].values)
X_train=np.array(pad_sequences(X_train, max_seq_length))
y_train=np.array(train_raw['toxic'])

X_valid = tokenizer.texts_to_sequences(valid_raw['comment_text'].values)
X_valid = np.array(pad_sequences(X_valid, max_seq_length))
y_valid=np.array(valid_raw['toxic'])

X_test = tokenizer.texts_to_sequences(test_raw['content'].values)
X_test = np.array(pad_sequences(X_test, max_seq_length))


print('The X_train,X_valid,X_test shape respectively is {}-{}-{}'.format(X_train.shape,X_valid.shape,X_test.shape))
train,valid,test=make_data()
!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip glove*.zip
embeddings_index = {}

f = open(os.path.join(os.getcwd(), 'glove.6B.{}d.txt'.format(str(embedding_dim))))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))
# first create a matrix of zeros, this is our embedding matrix
word_index = tokenizer.word_index
num_words=len(word_index)+1
embedding_matrix = np.zeros((num_words, embedding_dim))

# for each word in out tokenizer lets try to find that work in our w2v model
for word, i in tqdm(word_index.items()):
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # we found the word - add that words vector to the matrix
        embedding_matrix[i] = embedding_vector
    else:
        # doesn't exist, assign a random vector
        embedding_matrix[i] = np.random.randn(embedding_dim)
steps_per_epoch=X_train.shape[0]//BATCH_SIZE
with strategy.scope():
    model = Sequential()

    model.add(Embedding(num_words,
                        embedding_dim,
                        embeddings_initializer=Constant(embedding_matrix),
                        input_length=max_seq_length,
                        trainable=True))
    model.add(SpatialDropout1D(0.2))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(32)))
    model.add(Dropout(0.25))
    model.add(Dense(units=1, activation='sigmoid'))

    METRICS = [
              tf.keras.metrics.TruePositives(name='tp'),
              tf.keras.metrics.FalsePositives(name='fp'),
              tf.keras.metrics.TrueNegatives(name='tn'),
              tf.keras.metrics.FalseNegatives(name='fn'), 
              tf.keras.metrics.BinaryAccuracy(name='accuracy'),
              tf.keras.metrics.Precision(name='precision'),
              tf.keras.metrics.Recall(name='recall'),
              tf.keras.metrics.AUC(name='auc')]
    model.compile(loss ='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE),metrics =METRICS)

history=model.fit(
train,steps_per_epoch=steps_per_epoch,epochs=EPOCHS,callbacks=[early_stopping], validation_data=valid)
plot_loss(history)
plot_metrics(history)
y_predict=model.predict(valid, verbose=1)
y_predict[y_predict> 0.5] = 1
y_predict[y_predict <= 0.5] = 0
plot_cm(y_valid, y_predict, 'LSTM with Glovec-Confusion Matrix')
y_predict_prob=model.predict(valid, verbose=1)
fpr, tpr, _ = roc_curve(y_valid,y_predict_prob)
roc_auc = auc(fpr, tpr)
roc_curve_plot(fpr,tpr,roc_auc)
with strategy.scope():
    # Define an input sequence and process it.
    inputs = Input(shape=(max_seq_length,))
    embedding=Embedding(num_words,
                        embedding_dim,
                        embeddings_initializer=Constant(embedding_matrix),
                        input_length=max_seq_length,
                        trainable=True)(inputs)
    

    # Apply dropout to prevent overfitting
    embedded_inputs = Dropout(0.2)(embedding)
    
    # Apply Bidirectional LSTM over embedded inputs
    lstm_outs =Bidirectional(
        LSTM(300, return_sequences=True)
    )(embedded_inputs)
    
    # Apply dropout to LSTM outputs to prevent overfitting
    lstm_outs = Dropout(0.2)(lstm_outs)
    
    # Attention Mechanism - Generate attention vectors
    attention_vector = TimeDistributed(Dense(1))(lstm_outs)
    attention_vector = Reshape((X_train.shape[1],))(attention_vector)
    attention_vector = Activation('softmax', name='attention_vec')(attention_vector)
    attention_output = Dot(axes=1)([lstm_outs, attention_vector])
    
    # Last layer: fully connected with softmax activation
    fc = Dense(300, activation='relu')(attention_output)
    output = Dense(1, activation='sigmoid')(fc)
    
    model=tf.keras.Model(inputs,output)
    
    
    METRICS = [
              tf.keras.metrics.TruePositives(name='tp'),
              tf.keras.metrics.FalsePositives(name='fp'),
              tf.keras.metrics.TrueNegatives(name='tn'),
              tf.keras.metrics.FalseNegatives(name='fn'), 
              tf.keras.metrics.BinaryAccuracy(name='accuracy'),
              tf.keras.metrics.Precision(name='precision'),
              tf.keras.metrics.Recall(name='recall'),
              tf.keras.metrics.AUC(name='auc')]
    model.compile(loss ='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE),metrics =METRICS)
    print(model.summary())

history=model.fit(
train,steps_per_epoch=steps_per_epoch,epochs=EPOCHS,callbacks=[early_stopping], validation_data=valid)
plot_loss(history)
plot_metrics(history)
y_predict=model.predict(valid, verbose=1)
y_predict[ y_predict> 0.5] = 1
y_predict[y_predict <= 0.5] = 0
plot_cm(y_valid, y_predict, 'LSTM with Attention Mechanism-Confusion Matrix')
fy_predict_prob=model.predict(valid, verbose=1)
fpr, tpr, _ = roc_curve(y_valid,y_predict_prob)
roc_auc = auc(fpr, tpr)
roc_curve_plot(fpr,tpr,roc_auc)
sub['toxic'] = final_model.predict(test, verbose=1)
sub.to_csv('submission.csv', index=False)
