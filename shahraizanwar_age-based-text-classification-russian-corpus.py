# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import random as rn



import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

import tensorflow.keras.layers as L

from tensorflow.keras.losses import SparseCategoricalCrossentropy

from tensorflow.keras.optimizers import Adam



import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objs as go

import matplotlib.pyplot as plt



from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error



import nltk

from nltk.corpus import stopwords



import seaborn as sns

import re
DESC_FILE = "../input/fiction-corpus-for-agebased-text-classification/description.csv"

COLS = ['file_name', 'book_title', 'author', 'age_rating', 'genres']



desc_df = pd.read_csv(DESC_FILE, encoding="utf-8", delimiter=';', names=COLS)

desc_df.head()
TRAIN_DIR = "../input/fiction-corpus-for-agebased-text-classification/train/"

TEST_DIR = "../input/fiction-corpus-for-agebased-text-classification/test/"



def get_text(row):

    

    ## Find a file in train directory if not 

    ## found go to test directory

    

    try:

        f = open(TRAIN_DIR + row['file_name'], "r")

        row['file_name'] = ' '.join(f.read().split())

        row['data'] = 'Train'

        return row

    

    except FileNotFoundError:

        

        f = open(TEST_DIR + row['file_name'], "r")

        row['file_name'] = ' '.join(f.read().split())

        row['data'] = 'Test'

        return row

    



desc_df['data'] = 'All'

data = desc_df[['file_name','age_rating', 'data']].apply(get_text, axis=1)

data.columns = ['text', 'age', 'data']



data.head()
train_df = data.query("data == 'Train'").reset_index(drop=True)

train_df.head()
test_df = data.query("data == 'Test'").reset_index(drop=True)

test_df.head()
data_dist = data['data'].value_counts()

fig = go.Figure(go.Bar(x = data_dist.index, y = data_dist.values))



fig.update_layout(title_text="Data Distribution")

fig.show()
age_dist_train = train_df['age'].value_counts()

age_dist_test = test_df['age'].value_counts()



fig = make_subplots(

    rows=1, cols=2,

    subplot_titles=("Train","Test")

)



fig.add_trace(

    go.Bar(x = age_dist_train.index, y = age_dist_train.values),

    col=1, row=1

)



fig.add_trace(

    go.Bar(x = age_dist_test.index, y = age_dist_test.values),

    col=2, row=1

)



fig.update_layout(title_text="Age Distribution")
bins = [0, 6, 12, 18]

labels = ['0-6','7-12','13-18']

binned_age_train = pd.cut(train_df['age'], bins=bins, labels=labels).value_counts()

binned_age_test = pd.cut(test_df['age'], bins=bins, labels=labels).value_counts()



fig = make_subplots(

    rows=1, cols=2,

    subplot_titles=("Train","Test")

)



fig.add_trace(

    go.Bar(x = binned_age_train.index, y = binned_age_train.values),

    col=1, row=1

)



fig.add_trace(

    go.Bar(x = binned_age_test.index, y = binned_age_test.values),

    col=2, row=1

)



fig.update_layout(title_text="Binned Age Distribution")
X_train = train_df['text'].copy()

y_train = pd.cut(train_df['age'], bins=bins, labels=labels).copy()



X_test = test_df['text'].copy()

y_test = pd.cut(test_df['age'], bins=bins, labels=labels).copy()
%%time

#from nltk.stem.snowball import SnowballStemmer

from nltk.tokenize import word_tokenize

#stemmer = SnowballStemmer("russian")



def data_cleaner(text):

    

    text = re.sub(" \d+", " ", text)

    

    

    #removing stop words

    text = text.lower().split()

    text = " ".join([word for word in text if not word in stop_words])

    

    #Stemming

    #text = " ".join([stemmer.stem(w) for w in text])

    

    return text



#ps = PorterStemmer() 

stop_words = set(stopwords.words('russian'))



X_train_cleaned = X_train.apply(data_cleaner)

X_test_cleaned = X_test.apply(data_cleaner)



X_train_cleaned.head()
length_dist = [len(x.split(" ")) for x in X_train]

plt.hist(length_dist)

plt.title("Sequence length distribution")

plt.show()
%%time



vocab_size = 80000



tokenizer = Tokenizer(lower=False, num_words=vocab_size)

tokenizer.fit_on_texts(X_train)



X_train_enc = tokenizer.texts_to_sequences(X_train)

X_test_enc = tokenizer.texts_to_sequences(X_test)



#vocab_size = len(tokenizer.word_index)+1

#exp_sen = 1



print("Vocabulary size: {}".format(vocab_size))

#print("max length of sentence: {}".format(max_length))

#print("\nExample:\n")

#print("Sentence:\n{}".format(X_train[exp_sen]))

#print("\nAfter tokenizing :\n{}".format(X_train_enc[exp_sen]))
max_len = 10000



X_train_pd = pad_sequences(X_train_enc, padding='post', maxlen=max_len)

X_test_pd = pad_sequences(X_test_enc, padding='post', maxlen=max_len)
encoding = {'0-6': 0,

            '7-12': 1,

            '13-18': 2}



labels = ['0-6', '7-12', '13-18']



y_train_enc = y_train.copy()

y_test_enc = y_test.copy()



y_train_enc.replace(encoding, inplace=True)

y_test_enc.replace(encoding, inplace=True)
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()

tf.config.experimental_connect_to_cluster(tpu)

tf.tpu.experimental.initialize_tpu_system(tpu)

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)
seed_value = 1337

np.random.seed(seed_value)

tf.random.set_seed(seed_value)

rn.seed(seed_value)







# hyper parameters

EPOCHS = 5

BATCH_SIZE = 256

embedding_dim = 16



with tpu_strategy.scope():

    model = tf.keras.Sequential([

        L.Embedding(vocab_size, embedding_dim, input_length=X_train_pd.shape[1]),

        L.Bidirectional(L.LSTM(64,return_sequences=True)),

        L.Conv1D(64,8),

        L.MaxPool1D(),

        L.Bidirectional(L.LSTM(64,return_sequences=True)),

        L.Conv1D(64,6),

        L.MaxPool1D(),

        L.Bidirectional(L.LSTM(64,return_sequences=True)),

        L.Conv1D(64,3),

        L.MaxPool1D(),

        #L.LSTM(64,return_sequences=True),

        #L.Conv1D(64,2),

        #L.GlobalMaxPooling1D(),

        L.Flatten(),

        L.Dropout(0.5),

        L.Dense(128, activation="relu"),

        L.Dropout(0.5),

        L.Dense(64, activation="relu"),

        L.Dropout(0.5),

        L.Dense(3, activation="softmax")

    ])





    model.compile(loss=SparseCategoricalCrossentropy(),

                  optimizer='adam',metrics=['accuracy']

                 )



model.summary()
history = model.fit(X_train_pd, y_train_enc, epochs=EPOCHS, validation_split=0.12, batch_size=BATCH_SIZE)
fig = px.line(

    history.history, y=['loss', 'val_loss'],

    labels={'index': 'epoch', 'value': 'loss'}

)



fig.show()
fig = px.line(

    history.history, y=['accuracy', 'val_accuracy'],

    labels={'index': 'epoch', 'value': 'accuracy'}

)



fig.show()
pred = model.predict_classes(X_test_pd[0:1000], batch_size=8)
print('Accuracy: {}'.format(accuracy_score(pred, y_test_enc)))
conf = confusion_matrix(y_test_enc, pred)



cm = pd.DataFrame(

    conf, index = [i for i in labels],

    columns = [i for i in labels]

)



plt.figure(figsize = (12,7))

sns.heatmap(cm, annot=True, fmt="d")

plt.show()
print(classification_report(y_test_enc, pred, target_names=labels))