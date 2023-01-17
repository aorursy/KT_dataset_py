import pandas as pd
df=pd.read_csv('/kaggle/input/fake-news-data/train.csv')
df.head()
df=df.dropna()
X=df.drop('label',axis=1)
## Get the Dependent features
y=df['label']
X.shape,y.shape
import tensorflow as tf
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
voc_size=5000
messages=X.copy()
messages['title'][1]
messages.reset_index(inplace=True)
import nltk
import re
from nltk.corpus import stopwords
nltk.download('stopwords')
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
onehot_repr=[one_hot(words,voc_size) for words in corpus]
sent_length=20
embedded_docs=pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
print(embedded_docs)
embedded_docs[0]
embedding_vector_features=40
model=Sequential()
model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(100))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())
len(embedded_docs),y.shape
import numpy as np
X_final=np.array(embedded_docs)
y_final=np.array(y)
X_final.shape,y_final.shape
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.2, random_state=42)
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=50,batch_size=64)
y_pred=model.predict_classes(X_test)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,y_pred)
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
df_test=pd.read_csv('/kaggle/input/fake-news-data/test.csv')
df_X_test=df_test.iloc[:,:]
df_X_test.head()
df_X_testt=df_X_test.dropna()
df_X_testt.shape
test_messages=df_X_testt.copy()
test_messages.reset_index(inplace=True)
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus1 = []
for i in range(0, len(test_messages)):
    
    review = re.sub('[^a-zA-Z]', ' ', test_messages['title'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus1.append(review)
onehot_repr_test=[one_hot(words,voc_size)for words in corpus1]
sent_length_test=20
embedded_docs_test=pad_sequences(onehot_repr_test,padding='pre',maxlen=sent_length_test)
print(embedded_docs_test)
X_test_final=np.array(embedded_docs_test)
y_pred_test1=model.predict(X_test_final)
d=pd.DataFrame(y_pred_test1,columns=['PredictedValue'])
d.head(20)
submit=d.to_csv("submission.csv")