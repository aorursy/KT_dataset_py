import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
#load the dataset
fake_news = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/Fake.csv')
true_news = pd.read_csv('/kaggle/input/fake-and-real-news-dataset/True.csv')
fake_news.head()
true_news.head()
fake_news['label'] = 1
true_news['label'] = 0
news_data = pd.concat([fake_news,true_news])
print('No. of examples:',len(news_data))
news_data.head()
print(news_data['label'].value_counts())
sns.countplot(news_data['label'])
#drop the subjects and date column
news_data.drop('subject',axis=1,inplace=True)
news_data.drop('date',axis=1,inplace=True)
news_data.head()
#merge the title and text columns into one
news_data['news'] = news_data['title']+" "+news_data['text']
news_data.head()
#title and text columns no longer needed
news_data.drop('title',axis=1,inplace=True)
news_data.drop('text',axis=1,inplace=True)
news_data.head()
def remove_urls(text):
  return re.sub('https?:\S+','',text)
def remove_punctuation(text):
  return text.translate(str.maketrans('','',string.punctuation))
def remove_tags(text):
  return re.sub('<.*?>'," ",text)
def remove_numbers(text):
  return re.sub('[0-9]+','',text)
#remove urls from text
news_data['news'] = news_data['news'].apply(remove_urls)
#remove any tags present in the text
news_data['news'] = news_data['news'].apply(remove_tags)
#remove punctuation from text
news_data['news'] = news_data['news'].apply(remove_punctuation)
#remove numbers from the text
news_data['news'] = news_data['news'].apply(remove_numbers)
import nltk
from nltk.corpus import stopwords
stops = stopwords.words('english')
def remove_stopwords(text):
  cleaned = []
  for word in text.split():
    if word not in stops:
      cleaned.append(word)
  return " ".join(cleaned)
news_data['news'] = news_data['news'].apply(remove_stopwords)
news_data.head(10)
#convert words to lower case
news_data['news'] = news_data['news'].apply(lambda word : word.lower())
news_data.head()
#stemming/lemmatization
from nltk.stem import WordNetLemmatizer
#nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

def lemmatize_words(text):
  lemmas = []
  for word in text.split():
    lemmas.append(lemmatizer.lemmatize(word))
  return " ".join(lemmas)
news_data['lemmatized_news'] = news_data['news'].apply(lemmatize_words)
#split into X and y sets
#shuffle the dataset
news_data = news_data.sample(frac=1).reset_index(drop=True)
news_x = news_data['lemmatized_news'].values
news_y = news_data['label'].values
#tokenization,padding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer() 
tokenizer.fit_on_texts(news_x)
word_to_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(news_x)
vocab_size = len(word_to_index)
max_length = 200
embedding_dim = 100
padded_sequences = pad_sequences(sequences,maxlen=max_length,padding='post',truncating='post')
embeddings_index = {};
with open('/kaggle/working/glove.6B.100d.txt') as f:
    for line in f:
        values = line.split();
        word = values[0];
        coefs = np.asarray(values[1:], dtype='float32');
        embeddings_index[word] = coefs;

embeddings_matrix = np.zeros((vocab_size+1, embedding_dim));
for word, i in word_to_index.items():
    embedding_vector = embeddings_index.get(word);
    if embedding_vector is not None:
        embeddings_matrix[i] = embedding_vector;
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Embedding,SpatialDropout1D,LSTM,Bidirectional,Dropout
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
model = Sequential([
    Embedding(vocab_size+1, embedding_dim, input_length=max_length, weights=[embeddings_matrix], trainable=False),
    SpatialDropout1D(0.2),
    Bidirectional(LSTM(128,return_sequences=True)),
    Dropout(0.2),
    LSTM(64),
    Dense(32,activation='relu'),
    Dense(1,activation='sigmoid')
])
optimizer = Adam(learning_rate=0.01)
callbacks = ReduceLROnPlateau(monitor='val_accuracy',patience=2,factor=0.5,min_lr=0.00001)
model.compile(loss='binary_crossentropy',optimizer=optimizer,metrics=['accuracy'])
model.summary()
#split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(padded_sequences,news_y,test_size=0.15,random_state=1)
print('No. of training samples:',len(X_train))
print('No. of testing samples:',len(X_test))
epochs = 10
history = model.fit(X_train,y_train,epochs=epochs,validation_data=(X_test,y_test),batch_size=64,callbacks=[callbacks])
#plot the losses and accuracy
fig,axes = plt.subplots(1,2)
fig.set_size_inches(30,10)
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = list(range(10))
axes[0].plot(epochs,acc,label='training accuracy')
axes[0].plot(epochs,val_acc,label='validation accuracy')
axes[0].set_xlabel('epoch no.')
axes[0].set_ylabel('accuracy')
axes[0].legend()
axes[1].plot(epochs,loss,label='training loss')
axes[1].plot(epochs,val_loss,label='validation loss')
axes[1].set_xlabel('epoch no.')
axes[1].set_ylabel('loss')
axes[1].legend()
#model evaluation
train_stats = model.evaluate(X_train,y_train)
test_stats = model.evaluate(X_test,y_test)
print('training accuracy:',train_stats[1]*100)
print('testing accuracy:',test_stats[1]*100)
#classification report
from sklearn.metrics import classification_report,confusion_matrix
y_pred = model.predict_classes(X_test)
print(classification_report(y_test,y_pred))
print('Confusion matix:\n',confusion_matrix(y_test,y_pred))
