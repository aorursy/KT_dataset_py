#import necessary libraries



import numpy as np

import pandas as pd 

import tensorflow as tf

import re

from nltk.corpus import stopwords



from tensorflow.keras import regularizers, initializers, optimizers, callbacks

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.preprocessing.text import Tokenizer

from keras.utils.np_utils import to_categorical

from tensorflow.keras.layers import *

from tensorflow.keras.models import Model

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# read datasets



trn1 = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-toxic-comment-train.csv')

trn2 = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/jigsaw-unintended-bias-train.csv')





val = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/validation.csv')

test = pd.read_csv('/kaggle/input/jigsaw-multilingual-toxic-comment-classification/test.csv')

project_test = pd.read_csv('/kaggle/input/test-sentences-jigsaw/testtest.csv',header=None,names=['sentence', 'label'])
# data cleaning step: labels in trn2 not represented as binary integers



trn2['toxic'] = trn2.toxic.round().astype(int)

trn2_tox = trn2[trn2['toxic'] == 1]

trn2_ok = trn2[trn2['toxic'] == 0]
# combine into one training set



trn =  pd.concat([trn1[['comment_text', 'toxic']],trn2_tox[['comment_text', 'toxic']],trn2_ok[['comment_text', 'toxic']]])

# condense the training set since there are over a million examples



trn = trn.sample(n=300000)
# to be used for exploratory visualizations



trn_toxic = trn[trn['toxic'] == 1]

trn_nt = trn[trn['toxic'] ==0]
len(trn_toxic)
# remove empty comments



trn = trn.where(trn['comment_text'] != "")
# more standardizing



labels = ['toxic', 'not_toxic']

classes = trn['toxic'].values

trn_comments = trn['comment_text']

trn_comments = list(trn_comments)
# strip invalid characters and remove stopwords



def process_text(text, remove_stopwords = True):

    output = ""

    text = str(text).replace("\n", "")

    text = re.sub(r'[^\w\s]','',text).lower()

    if remove_stopwords:

        text = text.split(" ")

        for word in text:

            if word not in stopwords.words("english"):

                output = output + " " + word

    else:

        output = text

    return str(output.strip())[1:-3].replace("  ", " ")

    

texts = [] 



for line in trn_comments: 

    texts.append(process_text(line))
# make word cloud for toxic comments



from wordcloud import WordCloud

import plotly.express as px



def nonan(x):

    if type(x) == str:

        return x.replace("\n", "")

    else:

        return ""



text = ' '.join([nonan(abstract) for abstract in trn_toxic["comment_text"]])



wordcloud1 = WordCloud(max_font_size=None, background_color='white', collocations=False,width=1200, height=1000).generate(text)

fig1 = px.imshow(wordcloud1)

fig1.update_layout(title_text='Common words in toxic comments')
# make word cloud for non-toxic comments



text = ' '.join([nonan(abstract) for abstract in trn_nt["comment_text"]])



wordcloud2 = WordCloud(max_font_size=None, background_color='white', collocations=False,

                      width=1200, height=1000).generate(text)

fig2 = px.imshow(wordcloud2)

fig2.update_layout(title_text='Common words in non-toxic comments')
# percentage of toxic comments in training set



import seaborn as sns

sns.barplot(x=['Not Toxic', 'Toxic'], y=trn['toxic'].value_counts())
# hyperparams



MAX_NB_WORDS = 100000    

MAX_SEQUENCE_LENGTH = 200 

VALIDATION_SPLIT = 0.2  

EMBEDDING_DIM = 100      

GLOVE_DIR = '/kaggle/input/glove6b100dtxt/glove.6B.100d.txt'
# tokenize text and create dictionary



tokenizer = Tokenizer(num_words=MAX_NB_WORDS)

tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index

# pad sequences to standardize length



data = pad_sequences(sequences, padding = 'post', maxlen = MAX_SEQUENCE_LENGTH)
# split into train-validation



indices = np.arange(data.shape[0])

np.random.shuffle(indices)

data = data[indices]

labels = classes[indices]
# split into train-validation



num_validation_samples = int(VALIDATION_SPLIT*data.shape[0])

x_train = data[: -num_validation_samples]

y_train = labels[: -num_validation_samples]

x_val = data[-num_validation_samples: ]

y_val = labels[-num_validation_samples: ]
# upload pre-trained word embedding and set up embedding matrix for embedding layer



embeddings_index = {}

f = open(GLOVE_DIR)

for line in f:

    values = line.split()

    word = values[0]

    embeddings_index[word] = np.asarray(values[1:], dtype='float32')

f.close()



embedding_matrix = np.random.random((len(word_index) + 1, EMBEDDING_DIM))

for word, i in word_index.items():

    embedding_vector = embeddings_index.get(word)

    if embedding_vector is not None:

        embedding_matrix[i] = embedding_vector
def lstm():

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    embedding_layer = Embedding(len(word_index) + 1,

                               EMBEDDING_DIM,

                               weights = [embedding_matrix],

                               input_length = MAX_SEQUENCE_LENGTH,

                               trainable=False,

                               name = 'embeddings')

    embedded_sequences = embedding_layer(sequence_input)

    x = LSTM(25, return_sequences=True,name='lstm_layer')(embedded_sequences)

    x = GlobalMaxPool1D()(x)

    x = Dropout(0.5)(x)

    x = Dense(20, activation="relu")(x)

    x = Dropout(0.5)(x)

    preds = Dense(1, activation="sigmoid")(x)

    model = Model(sequence_input, preds)

    

    return model
lstm = lstm()

lstm.compile(loss = 'binary_crossentropy',

             optimizer='adam',

             metrics = ['accuracy'])
tf.keras.utils.plot_model(lstm)
print('Training progress:')

lstm_history = lstm.fit(x_train, y_train, epochs = 15, batch_size=32, validation_data=(x_val, y_val))

import matplotlib.pyplot as plt



loss_lstm = lstm_history.history['loss']

val_loss_lstm = lstm_history.history['val_loss']

epochs = range(1, len(loss_lstm)+1)

plt.plot(epochs, loss_lstm, label='Training loss')

plt.plot(epochs, val_loss_lstm, label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
from sklearn.metrics import roc_curve



y_hat_lstm = lstm.predict(x_val).ravel()

fpr_lstm, tpr_lstm, thresholds_lstm = roc_curve(y_val, y_hat_lstm)
from sklearn.metrics import auc

auc_lstm = auc(fpr_lstm, tpr_lstm)
auc_lstm
plt.figure(1)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_lstm, tpr_lstm, label='LSTM'.format(auc_lstm))

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc='best')

plt.show()

y_hat = np.around(y_hat_lstm, decimals=0, out=None)
from sklearn.metrics import classification_report
print(classification_report(y_val, y_hat, digits=2))
# define test sentences for project

X = project_test['sentence']
lstm_predictions_project = lstm.predict(X)
print(lstm_predictions_project)
def cnn():

    

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    embedding_layer = Embedding(len(word_index) + 1,

                               EMBEDDING_DIM,

                               weights = [embedding_matrix],

                               input_length = MAX_SEQUENCE_LENGTH,

                               trainable=False,

                               name = 'embeddings')

    

    embedded_sequences = embedding_layer(sequence_input)

    convs = []

    filter_sizes = [2,3,4]

    for filter_size in filter_sizes:

        l_conv = Conv1D(filters=30, 

                        kernel_size=filter_size, 

                        activation='relu')(embedded_sequences)

        l_pool = GlobalMaxPooling1D()(l_conv)

        convs.append(l_pool)

    l_merge = concatenate(convs, axis=1)

    x = Dropout(0.5)(l_merge)  

    x = Dense(20, activation='relu')(x)

    x = Dropout(0.5)(x)

    preds = Dense(1, activation='sigmoid')(x)

    model = Model(sequence_input, preds)

    model.compile(loss='binary_crossentropy',

                  optimizer='adam',

                  metrics=['acc'])

    model.summary()

    return model
cnn = cnn()

tf.keras.utils.plot_model(cnn)
cnn = cnn()

cnn.compile(loss = 'binary_crossentropy',

             optimizer='adam',

             metrics = ['accuracy'])
cnn_history = cnn.fit(x_train, y_train, epochs = 15, batch_size=32, validation_data=(x_val, y_val))
import matplotlib.pyplot as plt



loss_cnn = cnn_history.history['loss']

val_loss_cnn = cnn_history.history['val_loss']

epochs = range(1, len(loss_cnn)+1)

plt.plot(epochs, loss_cnn, label='Training loss')

plt.plot(epochs, val_loss_cnn, label='Validation loss')

plt.title('Training and validation loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend()

plt.show()
from sklearn.metrics import roc_curve



y_hat_cnn = cnn.predict(x_val).ravel()

fpr_cnn, tpr_cnn, thresholds_cnn = roc_curve(y_val, y_hat_cnn)



from sklearn.metrics import auc

auc_cnn = auc(fpr_cnn, tpr_cnn)



plt.figure(1)

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr_cnn, tpr_cnn, label='CNN'.format(auc_cnn))

plt.xlabel('False positive rate')

plt.ylabel('True positive rate')

plt.title('ROC curve')

plt.legend(loc='best')

plt.show()
auc_cnn
y_hat_cnn = np.around(y_hat_cnn, decimals=0, out=None)
print(classification_report(y_val, y_hat_cnn, digits=2))
cnn_predictions_project = cnn.predict(X)
print(cnn_predictions_project)