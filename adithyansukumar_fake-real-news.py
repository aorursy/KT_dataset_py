import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df_fake=pd.read_csv('../input/fake-and-real-news-dataset/Fake.csv')
df_true=pd.read_csv('../input/fake-and-real-news-dataset/True.csv')
df_fake.head()
df_true.head()
df_fake.info()
df_true.info()
plt.figure(figsize=(12,9))
sns.countplot(df_fake['subject'])
plt.title('Fake News')
sns.countplot(df_true['subject'])
plt.title('Real News')
df_fake['label']=0
df_true['label']=1
df=pd.concat([df_fake,df_true]).sample(frac=1)
df.tail()
sns.countplot(df['label'])
df.drop(['text','subject','date'],axis=1,inplace=True)
df.columns
from nltk.corpus import stopwords
import string
def text_process(news):
    no_punc=[char for char in news if char not in string.punctuation]
    no_punc=''.join(no_punc)
    word_seq=[word for word in no_punc.split() if word.lower not in stopwords.words('english')]
    return word_seq
news=df['title'].apply(text_process)
news
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
max_len=100
max_words=25000
tokenizer=Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(news)
sequences=tokenizer.texts_to_sequences(news)
news_data=pad_sequences(sequences,maxlen=max_len)
y=df['label']
len(news_data)
len(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(news_data,y,test_size=0.3)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Bidirectional,Dropout,Dense,Embedding
from tensorflow.keras.callbacks import EarlyStopping
model=Sequential()
model.add(Embedding(max_words,64,input_length=max_len))
model.add(Bidirectional(LSTM(100)))
model.add(Dropout(0.5))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
es=EarlyStopping(monitor='val_loss')

predictions=model.fit(x_train,y_train,epochs=30,validation_data=(x_test,y_test),callbacks=[es])
losses=pd.DataFrame(model.history.history)
losses.plot()
