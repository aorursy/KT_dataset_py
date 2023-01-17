# Importing the libraries
import numpy as np 
import pandas as pd 
df = pd.read_csv('../input/fake-news/train.csv')
df.head()
# Shape of the data
df.shape
# Creating the feature matrix
X = df.drop('label', axis = 1)
X.head()
# Get the dependent variable
y = df['label']
y.head()
# Removing the NaN values
df = df.dropna()
# Shape of the data after removing Nan values 
df.shape
# Importing the essential libraries
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
# Vocabulary size
vocab_size = 5000
messages = df.copy()
# Resetting the indices after the removal of NaN values
messages.reset_index(inplace = True)
# Example text from our data 
print(messages['title'][5])
print(messages['text'][5])
# Again importing the essential libraries
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
corpus
onehot_rep = [one_hot(words,vocab_size) for words in corpus]
onehot_rep
sent_len = 20
embedded = pad_sequences(onehot_rep, padding = 'pre', maxlen = sent_len)
print(embedded)
len(embedded)
embedded[0]
embedding_vector_features = 40
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_features, input_length = sent_len))
model.add(LSTM(100))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())
y = messages['label']
import numpy as np
X_final = np.array(embedded)
y_final = np.array(y)
X_final.shape,y_final.shape
# Train-Test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.33, random_state=30)
# Train the model
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)
y_pred = (model.predict_classes(X_test))
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
confusion_matrix(y_test,y_pred)
accuracy_score(y_test,y_pred)
from tensorflow.keras.layers import Dropout
# Creating the model
embedding_vector_features = 50
model = Sequential()
model.add(Embedding(vocab_size, embedding_vector_features, input_length = sent_len))
#model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.2))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())
X_final = np.array(embedded)
y_final = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.3, random_state=0)

# Train the model
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=64)
y_pred = (model.predict_classes(X_test))
print('\n Confusion Matrix:- \n',confusion_matrix(y_test,y_pred))
print('\n Accuracy:',accuracy_score(y_test,y_pred)* 100,'%')
# Creating model
embedding_vector_features=40
model1=Sequential()
model1.add(Embedding(vocab_size, embedding_vector_features, input_length=sent_len))
model1.add(Bidirectional(LSTM(100)))
# Adding the dropout rate of 0.3
model1.add(Dropout(0.3))
model1.add(Dense(1,activation='sigmoid'))
model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model1.summary())
len(embedded),y.shape
X_final=np.array(embedded)
y_final=np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.3, random_state = 1)

model1.fit(X_train,y_train,validation_data=(X_test,y_test), epochs=10, batch_size=64)
y_pred = (model.predict_classes(X_test))
print('\n Confusion Matrix:- \n',confusion_matrix(y_test,y_pred))
print('\n Accuracy:',accuracy_score(y_test,y_pred)* 100,'%')
