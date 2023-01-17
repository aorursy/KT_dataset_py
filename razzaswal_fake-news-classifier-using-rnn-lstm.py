import pandas as pd
data = pd.read_csv('../input/train/train.csv')
data.head()
data.tail()
data = data.dropna()
x = data.drop('label', axis=1)
y = data['label']
x.shape
y.shape
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
voc_size = 5000
messages = x.copy()
messages['title'][1]
messages.reset_index(inplace=True)
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
#dataset preprocessing
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    # print(i)
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
        
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
corpus
onehot_repr = [one_hot(words, voc_size)for words in corpus]
onehot_repr
sent_length = 20
embedded_docs =pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
embedded_docs
embedded_docs[0]
embedding_vector_features = 40
model = Sequential()
model.add(Embedding(voc_size,embedding_vector_features, input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
len(embedded_docs),y.shape
import numpy as np
x_final = np.array(embedded_docs)
y_final = np.array(y)
x_final.shape, y_final.shape
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_final, y_final, test_size=0.33, random_state=42)
model.fit(x_train,y_train,validation_data=(x_test,y_test), epochs=10, batch_size=64)
from tensorflow.keras.layers import Dropout

embedding_vector_features=40
model_1=Sequential()
model_1.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model_1.add(Dropout(0.3))
model_1.add(LSTM(100))
model_1.add(Dropout(0.3))
model_1.add(Dense(1,activation='sigmoid'))
model_1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model_1.fit(x_train,y_train,validation_data=(x_test,y_test), epochs=10, batch_size=64)
y_pred=model_1.predict_classes(x_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)