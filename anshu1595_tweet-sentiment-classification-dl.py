# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('/kaggle/input/train_data.csv')
data.head()
data.isnull().sum()
data=data[['sentiment','text']]
data.head()
data.isnull().sum()
data=data.dropna()
data.isnull().sum()
y=data['sentiment']
x=data['text']
x.head()
y = pd.Categorical(y)
y[:10]
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer
import spacy
from spacy import displacy
import string
import re
punct = string.punctuation
punct
nlp = spacy.load('en_core_web_sm')
len(punct)
def text_cleaning(sen):
    sen = re.sub("http?\S+","",sen)
    sen = sen.translate(str.maketrans(punct,32*" "))
    doc = nlp(sen)
    tokens = []
    for token in doc:
        if token.lemma_ != "-PRON-":
            temp = token.lemma_.lower().strip()
        else:
            temp = token.lower_.strip()
        tokens.append(temp)
        
    tokens = [token for token in tokens if token not in punct and token not in stopwords.words('english')]
    return tokens
text_cleaning("#BREAKING: Lockdown in containment zones extended till June-30,2020")
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report,confusion_matrix, roc_auc_score
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
X_train,X_test,y_train,y_test = train_test_split(x, y, test_size=0.20, stratify=y, random_state=1111)
print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)
tfidf = TfidfVectorizer(tokenizer=text_cleaning, ngram_range=(1,3),max_df=0.5, max_features=10000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
model = keras.models.Sequential()
model.add(keras.layers.Dense(1000,activation='relu',input_shape=(10000,)))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(1000, activation='relu'))
model.add(keras.layers.Dense(500, activation='relu'))
model.add(keras.layers.Dropout(0.25))
model.add(keras.layers.Dense(500, activation='relu'))
model.add(keras.layers.Dense(3,activation='softmax'))
model.compile(optimizer='sgd',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.summary()
le = LabelEncoder()
y_train_label = le.fit_transform(y_train)
y_test_label = le.transform(y_test)
early_s = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10,restore_best_weights=True)
history = model.fit(X_train_tfidf.toarray(), y_train_label, epochs=30, validation_data=(X_test_tfidf.toarray(),y_test_label), 
                     callbacks=[early_s])
history_df = pd.DataFrame(history.history)
history_df.head()
history_df[['accuracy','val_accuracy']].plot(figsize=(8,6))
history_df[['loss','val_loss']].plot(figsize=(8,6))
history_df[['accuracy','val_accuracy']].sort_values(by='val_accuracy',ascending=False).head()
model.evaluate(X_test_tfidf.toarray(),y_test_label)