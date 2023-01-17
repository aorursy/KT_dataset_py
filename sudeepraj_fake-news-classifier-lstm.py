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
df = pd.read_csv('/kaggle/input/fake-news/train.csv')
df.head()
df.shape
df.isnull().sum()[df.isnull().sum() != 0]
df = df.dropna()
X = df.drop('label', axis=1)
y = df['label']
print('Shape of X is {} and y is {}'.format(X.shape, y.shape))
y.value_counts()
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
voc_size = 5000  #Vocablury size

#One hot representation
messages = X.copy()
messages['title'][1]
messages.reset_index(inplace=True)
import nltk
import re
from nltk.corpus import stopwords

nltk.download('stopwords')
#Data preprocessing
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('^a-zA-Z', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
corpus
onehot_repr = [one_hot(words,voc_size)for words in corpus]
onehot_repr
#Embedding representation
sent_length = 20
embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
print(embedded_docs)
embedded_docs[0]
#Bidirectional LSTM
embedding_vector_features = 40
model = Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(Bidirectional(LSTM(100)))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())
len(embedded_docs)
X_final = np.array(embedded_docs)
y_final = np.array(y)
print('Shape of X_final {} and y_final {}'.format(X_final.shape, y_final.shape))
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=42)
#Model training
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)
#performance 
y_pred = model.predict_classes(X_test)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
cm = confusion_matrix(y_test, y_pred)
asc = accuracy_score(y_test, y_pred)
print('Confusion matrix {} and having accuracy score {}'.format(cm, asc))
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))



