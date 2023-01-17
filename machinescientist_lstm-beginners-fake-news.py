import pandas as pd 
import numpy as np
df = pd.read_csv("../input/fake-news/train.csv")
df.head()
df = df.dropna()
# get independent feature
x = df.drop("label", axis=1)

# get dependent feature
y = df["label"]
print(x.shape)
print(y.shape)
import tensorflow as tf
tf.__version__
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential 
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM, Dense, Bidirectional, Dropout
'''Vocabulary size'''
vocab_size = 6000
msg = x.copy()
'''we are reseting the index as we have droped some values.'''

msg.reset_index(inplace=True)
import nltk
import re
from nltk.corpus import stopwords
'''data preprocessing removing all the things accept letters and doing it lower case'''
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(msg)):
    review = re.sub('[^a-zA-Z]', ' ', msg['title'][i])
    review = review.lower()
    review = review.split()
    ## removing stopwords
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
## observing some example
corpus[:5]
onehot_repr=[one_hot(words,vocab_size)for words in corpus] 
onehot_repr[:5]
sent_length=30
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)
'''CREATING LSTM MODEL''' 
embedding_vector_features=40
model=Sequential()
model.add(Embedding(vocab_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(128))
model.add(Dense(1,activation='relu'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())
'''CREATING BIDIRECTONAL LSTM MODEL'''
embedding_vector_features=40
model1=Sequential()
model1.add(Embedding(vocab_size,embedding_vector_features,input_length=sent_length))
model1.add(Bidirectional(LSTM(100)))
model1.add(Dropout(0.3))
model1.add(Dense(1,activation='sigmoid'))
model1.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model1.summary())
x_final = np.array(embedded_docs)
y_final = np.array(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_final, y_final, test_size=0.33, random_state=42)
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=32)
y_pred = model.predict_classes(X_test)
model1.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=64)
y_pred_bid_lstm = model1.predict_classes(X_test)
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
confusion_matrix(y_test,y_pred)
## For LSTM model
cnf_matrix_log = confusion_matrix(y_test,y_pred)

sns.heatmap(pd.DataFrame(cnf_matrix_log), annot=True,cmap="Reds" , fmt='g')
plt.tight_layout()
plt.title('Confusion matrix\n', y=1.1)
## For Bidirectonal LSTM
cnf_matrix_log = confusion_matrix(y_test,y_pred_bid_lstm)

sns.heatmap(pd.DataFrame(cnf_matrix_log), annot=True,cmap="Reds" , fmt='g')
plt.tight_layout()
plt.title('Confusion matrix\n', y=1.1)