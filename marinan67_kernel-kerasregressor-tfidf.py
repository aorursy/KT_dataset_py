import pandas as pd

import numpy as np

import os

from keras.optimizers import SGD

from keras.optimizers import rmsprop

from keras.utils import np_utils

from sklearn.model_selection import train_test_split

from keras.callbacks import EarlyStopping

from keras.models import Sequential

from keras.layers import Activation, Dropout, Flatten, Dense,Conv1D,MaxPooling1D,AveragePooling1D,BatchNormalization

from sklearn.metrics import confusion_matrix,classification_report,roc_auc_score
print(os.listdir("../input"))
train = pd.read_csv('../input/train.csv')

train.head()
train.shape
print(train['comment_text'][1804800])
import gensim

from gensim.utils import simple_preprocess

from gensim.parsing.preprocessing import STOPWORDS

from nltk.stem import WordNetLemmatizer, SnowballStemmer

from nltk.stem.porter import *

import numpy as np

np.random.seed(2018)

import nltk

nltk.download('wordnet')
import re

def obr_text(text):

    text = text.lower().replace("ё", "е")

    text = re.sub(r"\d+", "", text, flags=re.UNICODE)

    return text.strip()

train['comment_text'] = [obr_text(t) for t in train['comment_text']]
englishStemmer=SnowballStemmer("english")

russianStemmer=SnowballStemmer("russian")

def lemmatize_stemming(text):

    return englishStemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):

    result = []

    for token in gensim.utils.simple_preprocess(text):

        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) >= 3:

            result.append(lemmatize_stemming(token))

    return result
train['comment_text'] = [preprocess(t) for t in train['comment_text']]
print(train['comment_text'][1804800])
train['comment_text'] = [' '.join(t) for t in train['comment_text']]
print(train['comment_text'][1804800])
results = set()

my_df = train['comment_text']

my_df.apply(results.update)

max_word = len(results)

print (len(results))
from collections import Counter

ls =[]

for i in train['comment_text']:

     ls.append(len(str(i).split()))   

c = Counter(ls)
%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()

plt.hist(ls)
train['target'] = round(train['target'], 1)
from sklearn.utils import resample

df_majority = train[train.target==0]

df_minority1 = train.loc[(train['target'] >0) & (train['target'] <= 0.1)]

df_minority2 = train.loc[(train['target'] >0.1) & (train['target'] <= 0.2)]

df_minority3 = train.loc[(train['target'] >0.2) & (train['target'] <= 0.3)]

df_minority4 = train.loc[(train['target'] >0.3) & (train['target'] <= 0.4)]

df_minority5 = train.loc[(train['target'] >0.4) & (train['target'] <= 0.5)]

df_minority6 = train.loc[(train['target'] >0.5) & (train['target'] <= 0.6)]

df_minority7 = train.loc[(train['target']>0.6) & (train['target'] <= 0.7)]

df_minority8 = train.loc[(train['target'] >0.7) & (train['target'] <= 0.8)]

df_minority9 = train.loc[(train['target'] >0.8) & (train['target'] <= 0.9)]

df_minority10 = train.loc[(train['target'] >0.9) & (train['target'] <= 1.0)]

 

# Downsample majority class

df_majority_downsampled = resample(df_majority, 

                                 replace=False,    # sample without replacement

                                 n_samples=len(df_majority)//3,     # to match minority class

                                 random_state=100) # reproducible results

df_minority_1 = resample(df_minority1, replace=True, n_samples=len(df_majority)//4, random_state=100)

df_minority_2 = resample(df_minority2, replace=True, n_samples=len(df_majority)//4, random_state=100)

df_minority_3 = resample(df_minority3, replace=True, n_samples=len(df_majority)//4, random_state=100)

df_minority_4 = resample(df_minority4, replace=True, n_samples=len(df_majority)//4, random_state=100)

df_minority_5 = resample(df_minority5, replace=True, n_samples=len(df_majority)//4, random_state=100)

df_minority_6 = resample(df_minority6, replace=True, n_samples=len(df_majority)//4, random_state=100)

df_minority_7 = resample(df_minority7, replace=True, n_samples=len(df_majority)//4, random_state=100)

df_minority_8 = resample(df_minority8, replace=True, n_samples=len(df_majority)//4, random_state=100)

df_minority_9 = resample(df_minority9, replace=True, n_samples=len(df_majority)//4, random_state=100)

df_minority_10 = resample(df_minority10, replace=True, n_samples=len(df_majority)//4, random_state=100)

 

# Combine minority class with downsampled majority class

#df_downsampled = pd.concat([df_majority_downsampled, df_minority])

df_downsampled = pd.concat([df_majority_downsampled, df_minority_1, df_minority_2,df_minority_3,df_minority_4,

                            df_minority_5,df_minority_6,df_minority_7,df_minority_8,df_minority_9,df_minority_10])
# Display new class counts

print(len(df_downsampled))

bins = 50

plt.hist(df_downsampled['target'], bins, alpha=0.5, color='b',label='Сбалансированный датасет')

plt.hist(train['target'], bins, alpha=0.5, color='r',label='Заданный датасет')

plt.legend(loc='upper right')

plt.show()
x_raw = df_downsampled['comment_text'].values
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer(max_word)

tokenizer.fit_on_texts(x_raw) 

vocab_size = len(tokenizer.word_index) + 1

print (vocab_size)
print(x_raw.shape, x_raw[1804800])
from sklearn.feature_extraction.text import TfidfVectorizer

tv = TfidfVectorizer(max_features=1000,ngram_range=(1,2),analyzer= 'word',

                        token_pattern=r"\b\w[\w']+\b", min_df=0.01, max_df=0.8, 

                        stop_words = ('english', 'russian'))
x_train = tv.fit_transform(x_raw)

#vocab = tv.get_feature_names()
print (x_train[1804800].toarray())
print(x_train.shape)
y_train = np.array(df_downsampled['target']).astype("float32")
from sklearn.model_selection  import  train_test_split 

x_train1, x_train2,y_train1,y_train2 = train_test_split(x_train,y_train, test_size=0.3, shuffle = True)
print(x_train1.shape, len(y_train1))

print(x_train2.shape, len(y_train2))
early_stopping_callback = EarlyStopping(monitor='acc', patience=2, restore_best_weights=True)
from keras.wrappers.scikit_learn import KerasRegressor

def create_model(optimizer='adam',

                 kernel_initializer='glorot_uniform', 

                 dropout=0.2):

    model = Sequential()

    model.add(Dense(512,activation='relu',kernel_initializer=kernel_initializer, input_shape=(x_train.shape[1],)))

    model.add(Dropout(dropout))

    model.add(Dense(256,activation='relu',kernel_initializer=kernel_initializer))

    model.add(Dropout(dropout))

    model.add(Dense(1,activation='sigmoid',kernel_initializer=kernel_initializer))



    model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])



    return model



# wrap the model using the function you created

clf = KerasRegressor(build_fn=create_model,epochs=55, batch_size=300,callbacks=[early_stopping_callback],verbose=0)
clf.fit(x_train1, y_train1)

print("Обучение остановлено на эпохе", early_stopping_callback.stopped_epoch)
y_pred=clf.predict(x_train2)

from sklearn.metrics import r2_score

r2_score(y_train2, y_pred)
test = pd.read_csv('../input/test.csv')

test.head()
test['comment_text'] = [obr_text(t) for t in test['comment_text']]
test['comment_text'] = [preprocess(t) for t in test['comment_text']]
test['comment_text'] = [' '.join(t) for t in test['comment_text']]
x_raw_t = test['comment_text'].values
tokenizer.fit_on_texts(x_raw_t) 

x_test = tv.transform(x_raw_t)
print(x_test.shape)
y_test=clf.predict(x_test)

print(y_test[:])
sample_submission= pd.read_csv('../input/sample_submission.csv')

sample_submission.head()
sample_submission['prediction']=y_test

sample_submission.head()

sample_submission.to_csv('submission.csv', sep=',',index = False)