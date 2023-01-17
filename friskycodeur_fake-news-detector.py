import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set_style('darkgrid')

import nltk
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import STOPWORDS,WordCloud
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata


from keras.preprocessing import text,sequence
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
from nltk import pos_tag
from nltk.corpus import wordnet
import keras
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout,Embedding
from keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
real_news=pd.read_csv('../input/fake-and-real-news-dataset/True.csv')
fake_news=pd.read_csv('../input/fake-and-real-news-dataset/Fake.csv')
real_news.head()
fake_news.head()
real_news['Isfake']=0
fake_news['Isfake']=1
df=pd.concat([real_news,fake_news])
df.sample(5)
df.isnull().sum()
sns.countplot(df.Isfake)
df.title.count()
df.subject.value_counts()
plt.figure(figsize=(10,10))
chart=sns.countplot(x='subject',hue='Isfake',data=df,palette='muted')
chart.set_xticklabels(chart.get_xticklabels(),rotation=90,fontsize=10)
df['text']= df['text']+ " " + df['title']
del df['title']
del df['subject']
del df['date']
stop_words=set(stopwords.words('english'))
punctuation=list(string.punctuation)
stop_words.update(punctuation)
def string_html(text):
    soup=BeautifulSoup(text,"html.parser")
    return soup.get_text()

def remove_square_brackets(text):
    return re.sub('\[[^]]*\]','',text)

def remove_URL(text):
    return re.sub(r'http\S+','',text)

def remove_stopwords(text):
    final_text=[]
    for i in text.split():
        if i.strip().lower() not in stop_words:
            final_text.append(i.strip())
    return " ".join(final_text)

def clean_text_data(text):
    text=string_html(text)
    text=remove_square_brackets(text)
    text=remove_stopwords(text)
    text=remove_URL(text)
    return text
df['text']=df['text'].apply(clean_text_data)
plt.figure(figsize=(20,20))
wordcloud=WordCloud(stopwords=STOPWORDS,height=600,width=1200).generate(" ".join(df[df.Isfake==1].text))
plt.imshow(wordcloud,interpolation='bilinear')
plt.figure(figsize=(20,20))
wordcloud=WordCloud(stopwords=STOPWORDS,height=600,width=1200).generate(" ".join(df[df.Isfake==0].text))
plt.imshow(wordcloud,interpolation='bilinear')
X_train,X_test,y_train,y_test=train_test_split(df.text,df.Isfake,random_state=0)
max_features=10000
max_len=300
tokenizer=text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(X_train)
tokenizer_train=tokenizer.texts_to_sequences(X_train)
X_train=sequence.pad_sequences(tokenizer_train,maxlen=max_len)
tokenizer_test=tokenizer.texts_to_sequences(X_test)
X_test=sequence.pad_sequences(tokenizer_test,maxlen=max_len)
glove_file='../input/glove-twitter/glove.twitter.27B.100d.txt'
def get_coefs(word, *arr):
    return word, np.asarray(arr,dtype='float32')
embeddings_index=dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(glove_file,encoding="utf8"))
all_embs=np.stack(embeddings_index.values())
emb_mean,emb_std=all_embs.mean(),all_embs.std()
emb_size=all_embs.shape[1]

word_index=tokenizer.word_index
nb_words=min(max_features,len(word_index))

embedding_matrix = np.random.normal(emb_mean,emb_std,(nb_words,emb_size))

for word,i in word_index.items():
    if i>=max_features: continue
    embedding_vector=embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i]=embedding_vector
batch_size=256
epochs=10
emb_size=100
leaning_rate_reduction=ReduceLROnPlateau(monitor='val_accuracy',patience=2,verbose=10,factor=0.5,min_lr=0.00001)
model=Sequential()
model.add(Embedding(max_features,output_dim=emb_size,weights=[embedding_matrix],input_length=max_len,trainable=False))
model.add(LSTM(units=256,return_sequences=True,recurrent_dropout=0.25,dropout=0.25))
model.add(LSTM(units=128,return_sequences=True,recurrent_dropout=0.25,dropout=0.25))
model.add(LSTM(units=64,recurrent_dropout=0.1,dropout=0.1))
model.add(Dense(units=32,activation='relu'))
model.add(Dense(1,'sigmoid'))
model.compile(optimizer=keras.optimizers.Adam(lr=0.01),loss='binary_crossentropy',metrics=['accuracy'])
model.summary()
history=model.fit(X_train,y_train,batch_size=batch_size,validation_data=(X_test,y_test),epochs=epochs,callbacks=[leaning_rate_reduction])
pred = model.predict_classes(X_test)
pred[5:10]
epochs = [i for i in range(10)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['accuracy']
train_loss = history.history['loss']
val_acc = history.history['val_accuracy']
val_loss = history.history['val_loss']
fig.set_size_inches(20,10)

ax[0].plot(epochs,train_acc,'go-',label='Training Accuracy')
ax[0].plot(epochs,val_acc,'ro-',label='Validation Accuracy')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')
ax[0].legend()

ax[1].plot(epochs,train_loss,'go-',label='Training Loss')
ax[1].plot(epochs,val_loss,'ro-',label='Validation Loss')
ax[1].set_xlabel('Loss')
ax[1].set_ylabel('Accuracy')
ax[1].legend()

plt.show()
cm=confusion_matrix(y_test,pred)
cm=pd.DataFrame(cm,index=['Fake','Not Fake'],columns=['Fake','Not Fake'])
cm
plt.figure(figsize=(10,10))
sns.heatmap(cm,cmap="Blues",linecolor='black',linewidth=1,annot=True,fmt='',xticklabels=['Fake','Not Fake'],yticklabels=['Fake','Not Fake'])
plt.xlabel('Actual')
plt.ylabel('Predicted')
print(f'Accuracy of the model on Training Data is - { model.evaluate(X_train,y_train)[1]*100:.2f}')
print(f'Accuracy of the model on Testing Data is -  {model.evaluate(X_test,y_test)[1]*100:.2f}')