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
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout,Bidirectional,Embedding,Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score,confusion_matrix
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
data= pd.read_csv('/kaggle/input/tweet-sentiment-extraction/train.csv')
test=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
def extraction(column):
    corpus=[]
    for i in range(0,len(column)):
        review = re.sub('[^a-zA-Z0-9]',' ', str(column[i]))
        review = review.lower()
        review = review.split()
        stemmer=WordNetLemmatizer()
        review = [stemmer.lemmatize(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        corpus.append(review)
    return corpus
def embedding(voc_size,max_review_length,corpus):
    one_hot_rep=[one_hot(i,voc_size) for i in corpus]
    sen=sequence.pad_sequences(one_hot_rep, maxlen=max_review_length)
    return sen
corpus=extraction(data.selected_text)
X=embedding(5000,500,corpus)
y=data.sentiment
encoder=LabelEncoder()
y=encoder.fit_transform(y)
y=to_categorical(y)
from sklearn.model_selection import train_test_split
x_train,x_val,y_train,y_val=train_test_split(X,y,test_size=0.2,random_state=101)
def model(voc_size,embedding_vecor_length,input_length):

    model = Sequential()
    model.add(Embedding(5000, embedding_vecor_length, input_length=500))
    model.add(Dropout(0.2))
    model.add(Bidirectional(LSTM(100)))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model
model=model(5000,32,500)
early=EarlyStopping(patience=3)
model.fit(x=x_train,y=y_train,validation_data=(x_val,y_val),batch_size=256,epochs=10)
test_x=extraction(test.text)
x_test=embedding(5000,500,test_x)
y_test=test.sentiment
encoder=LabelEncoder()
y_test=encoder.fit_transform(y_test)
y_test=to_categorical(y_test)
prediction=model.predict(x_test)
model.evaluate(x_test,y_test,verbose=0)
sub=pd.read_csv('/kaggle/input/tweet-sentiment-extraction/sample_submission.csv')
selected=pd.DataFrame(test_x)
dataset=pd.concat([sub['textID'],selected],axis=1)
dataset.columns=['textID','selected_text']
dataset.to_csv('sample_submission.csv',index=False)