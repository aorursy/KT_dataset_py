# Importing the libraries

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import warnings

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer





from sklearn.pipeline import make_pipeline

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectKBest

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectFromModel

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

from keras.wrappers.scikit_learn import KerasClassifier

# MLP for Pima Indians Dataset with 10-fold cross validation via sklearn

from keras.models import Sequential

from keras.layers import Dense, Dropout

from sklearn import preprocessing

from sklearn import svm





from tqdm import tqdm

from sklearn.model_selection import  KFold

from sklearn import preprocessing

from keras.models import Sequential, Model



from keras.layers import multiply, LeakyReLU, Dense, GlobalAveragePooling1D, Dropout,Input,Conv2D,MaxPooling2D,Input, add, Conv1D, concatenate, BatchNormalization, GlobalMaxPooling1D ,SpatialDropout1D, LSTM, GlobalMaxPool1D, Flatten, GlobalAveragePooling2D, GRU, CuDNNGRU, Embedding, CuDNNLSTM, Bidirectional

from keras.callbacks import EarlyStopping, ModelCheckpoint



from sklearn.model_selection import  KFold



from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

from keras.layers import  Embedding, LSTM, SpatialDropout1D

from sklearn.model_selection import train_test_split

from keras.utils.np_utils import to_categorical

from keras import optimizers



import tensorflow as tf

import keras.backend as K

from keras.engine.topology import Layer

from keras import initializers, regularizers, constraints, optimizers





import pickle

import gc



from nltk.stem import WordNetLemmatizer

from nltk.corpus import stopwords

from string import punctuation

from nltk.stem.snowball import SnowballStemmer

from nltk.stem import PorterStemmer

from keras.utils import np_utils

import unidecode



from sklearn.model_selection import train_test_split



from gensim.models import phrases, word2vec



import random

import re



import lightgbm as lgb

warnings.filterwarnings('ignore')

%matplotlib inline
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/mllibre/ml/train.csv')

data = data.drop_duplicates(subset='title').reset_index(drop=True)
pd.set_option('display.max_row', 20)
pd.DataFrame(data.category.unique())
data_test = pd.read_csv('/kaggle/input/mllibre/ml/test.csv')

data_test.head(10)
data.shape
language_encoder = LabelEncoder()

data['language'] = language_encoder.fit_transform(data['language'])

data_test['language'] = language_encoder.transform(data_test['language'])
category_encoder = LabelEncoder()

data['category'] = category_encoder.fit_transform(data['category'])

data.groupby('category').count()[['title']].sort_values(['title'], ascending=False)
aug_cats = data.groupby('category',as_index=False).count().sort_values(['title'], ascending=False)

aug_cats = aug_cats[aug_cats["title"] <= 5000]

aug_cats.head()
aug_cats_in = aug_cats["category"].values

aug_cats_in
Y = data["category"].values

L = data["language"].values
documents = data["title"].values

documents_test = data_test["title"].values

data.drop('title', axis=1, inplace=True)

data_test.drop('title', axis=1, inplace=True)
documents.shape


def pre_text(corpus):

    



    for sen in tqdm(range(0, len(corpus))):



        document = re.sub(r'\W', ' ', str(corpus[sen]))

        document = unidecode.unidecode(document)

        corpus[sen] = document.lower()



pre_text(documents)

pre_text(documents_test)
categories = pd.DataFrame(data.category.unique())
categories_v = categories.values



for c in tqdm(categories_v):

    c = c[0]

    cat_data = data.index[data['category'] == c].tolist()

    docs = documents[cat_data]

    cv=CountVectorizer(min_df=0.01,lowercase=False)

    X=cv.fit_transform(docs)

    

    documents[cat_data] = [' '.join(doc) for doc in cv.inverse_transform(X)]



del X

del cv

del docs

gc.collect()

x = np.isin(Y, aug_cats_in)

x = np.argwhere(x).flatten()

len(x)
extra_data = {"x":[],"y":[],"l":[]}

extra=True

if extra:

    for j in tqdm(x):



        y = Y[j]

        l = L[j]

        doc = documents[j]

        current = aug_cats[aug_cats["category"] == y]["title"].values[0]



        iters = (5000-current)//current



        for k  in range(iters):

            s = doc.split()

            random.shuffle(s)

            s = ' '.join(s)



            extra_data["x"].append(s)

            extra_data["y"].append(y)

            extra_data["l"].append(l)

        

extra_data = pd.DataFrame(extra_data)

extra_data = extra_data.drop_duplicates(subset='x')





extra_data.describe()
documents = np.concatenate((documents,extra_data["x"].values),axis=0)

Y = np.concatenate((Y,extra_data["y"].values),axis=0)

L = np.concatenate((L,extra_data["l"].values),axis=0)



del extra_data



gc.collect()
pd.DataFrame(documents[:10]).head(10)
pd.DataFrame(documents_test[:10]).head(10)
documents = np.array(documents)




max_fatures = 100000

seq_l = 15

tokenizer = Tokenizer(num_words=max_fatures, split=' ',filters='',lower=False)

tokenizer.fit_on_texts(documents)

train_x = np.array(tokenizer.texts_to_sequences(documents))

#train_x = pad_sequences(train_x,maxlen=seq_l)



test_x = tokenizer.texts_to_sequences(documents_test)

test_x = pad_sequences(test_x,maxlen=seq_l)



word_index = tokenizer.word_index

print('Found %s unique tokens.' % len(word_index))

num_words = min(max_fatures, len(word_index)) + 1
filename = 'tokenizer.pickle'

outfile = open(filename,'wb')

pickle.dump(tokenizer,outfile)

outfile.close()
filename = 'train_x_es.pickle'

outfile = open(filename,'wb')

pickle.dump(train_x,outfile)

outfile.close()



filename = 'test_x_es.pickle'

outfile = open(filename,'wb')

pickle.dump(test_x,outfile)

outfile.close()





filename = 'category_encoder_es.pickle'

outfile = open(filename,'wb')

pickle.dump(category_encoder,outfile)

outfile.close()



filename = 'language_encoder_es.pickle'

outfile = open(filename,'wb')

pickle.dump(language_encoder,outfile)

outfile.close()
def batch_generator(lines, y, l, batch_size = 32):

  

  while True:

 

    batch_paths = np.random.randint(0,lines.shape[0], batch_size)

  

    linesX = lines[batch_paths]

    linesY = y[batch_paths]

    linesL = l[batch_paths]

    

    train_ba = []

    for x in linesX:

        doc = x

        random.shuffle(doc)

        train_ba.append(doc)

    

    train_ba = pad_sequences(train_ba,maxlen=seq_l)

    target = linesY # np_utils.to_categorical(linesY,num_classes=1588)



        



  

    yield ([linesL,train_ba], target)
embed_dim = 128



def create_model():





    inputs1 = Input(shape=(1,))

    fe3 = Dense(1, activation='relu')(inputs1)

    inputs2 = Input(shape=(seq_l,))

    s0 = Embedding(num_words, embed_dim)(inputs2)

    #s0 = SpatialDropout1D(0.2)(s0)

    #s01 = Conv1D(32, 3, activation='relu')(s0)

    s02 = GlobalAveragePooling1D()(s0)

    s03 = GlobalMaxPooling1D()(s0)

    

    #s0 = BatchNormalization()(s0)

    #s0 = BatchNormalization()(s0)

    #se3 = Dense(1024, activation='relu')(s0)

    decoder1 = concatenate([fe3, s02,s03])



    outputs = Dense(1588, activation='softmax')(decoder1)

    # merge the two input models

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)

    

    model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=['accuracy'])



    return model
train_X, val_X, train_y, val_y, train_l, val_l  = train_test_split(train_x, Y, L, test_size=0.05, random_state=42)

    

callbacks = [ModelCheckpoint(filepath='weights_{}.h5'.format(1), monitor='val_accuracy', save_best_only=True,verbose=1)]

train_gen = batch_generator(train_X,train_y, train_l, batch_size=1024)

val_gen = batch_generator(val_X,val_y,val_l, batch_size=1024)



model = create_model()

#model.load_weights('/kaggle/input/preml2/bn.h5')

model.fit_generator(

    

generator=train_gen,

epochs=2,

verbose=2,

steps_per_epoch=len(train_X) // 1024,

validation_data=val_gen,

validation_steps=len(val_X) // 1024,

callbacks=callbacks)







   
sample_submission = pd.read_csv('/kaggle/input/mllibre/ml/sample_submission.csv')

sample_submission.head(10)
for i in tqdm(range(len(test_x))):

    languaje = data_test.iloc[i]["language"]

    input_data = [[languaje],[test_x[i]]]

    p = np.argmax(model.predict(input_data))

    cat = category_encoder.inverse_transform([p])[0]

    sample_submission.at[i,'category']=cat

    
sample_submission.head()
sample_submission.to_csv("submission.csv",index=False)