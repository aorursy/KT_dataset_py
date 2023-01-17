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

#now read the json file using pandas

df = pd.read_json("../input/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json", lines=True)
df.head()
#lets find if there is any NaN valus, because NaN values give wrong visualization
df.isna().sum()
del df['article_link']
import seaborn as sns

sns. set_style("dark")
sns.countplot(df.is_sarcastic)
#first we import all necessary libs
import matplotlib.pyplot as plt
import nltk
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud,STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize,sent_tokenize
from bs4 import BeautifulSoup
import re,string,unicodedata
from keras.preprocessing import text, sequence
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from string import punctuation
import keras
from keras.models import Sequential
from keras.layers import Dense,Embedding,LSTM,Dropout,Bidirectional,GRU
import tensorflow as tf
# set and define stop word

stopwd = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stopwd.update(punctuation)
#use beautifulsoup library to extract text from html data
def clean_html(text):
    soup=BeautifulSoup(text, "html.parser")
    return soup.get_text()

#remove [], */ etc html tags from text using "re" lib

def remove_betwn_square_brackets(text):
    return re.sub('\[[^]]*\]', '', text)

#remove urls from data

def remove_betwn_square_brackets(text):
    return re.sub(r'http\S+', '', text)

#lets remove stopwords we counted before
def remove_stopwords(text):
    final_text = []
    for i in text.split():
        if i.strip().lower() not in stopwd:
            final_text.append(i.strip())
    return " ".join(final_text)

#finally we remove all noisy data
def remove_noisy_text(text):
    text = clean_html(text)
    text = remove_betwn_square_brackets(text)
    text = remove_stopwords(text)
    return text



#lets clean the dataset and view

df['headline']=df['headline'].apply(remove_noisy_text)
df['headline']
plt.figure(figsize = (15,15)) # non-sarcastic words wordcloud
wordcld = WordCloud(max_words = 3000 , width = 1400 , height = 600).generate(" ".join(df[df.is_sarcastic == 0].headline))
plt.imshow(wordcld , interpolation = 'bilinear')
plt.figure(figsize = (15,15)) # non-sarcastic words wordcloud
wordcld = WordCloud(max_words = 3000 , width = 1400 , height = 600).generate(" ".join(df[df.is_sarcastic == 1].headline))
plt.imshow(wordcld , interpolation = 'bilinear')
#number of words
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
text_len=df[df['is_sarcastic']==1]['headline'].str.split().map(lambda x: len(x))
ax1.hist(text_len,color='red')
ax1.set_title('Sarcastic text')
text_len=df[df['is_sarcastic']==0]['headline'].str.split().map(lambda x: len(x))
ax2.hist(text_len,color='green')
ax2.set_title('Not Sarcastic text')
fig.suptitle('Words in texts')
plt.show()

#average word length
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,5))
word=df[df['is_sarcastic']==1]['headline'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax1,color='red')
ax1.set_title('Sarcastic text')
word=df[df['is_sarcastic']==0]['headline'].str.split().apply(lambda x : [len(i) for i in x])
sns.distplot(word.map(lambda x: np.mean(x)),ax=ax2,color='green')
ax2.set_title('Not Sarcastic text')
fig.suptitle('Average word length in each text')

#lets convert our text data into more acceptable format

#split words from a sentence and keep is sentence in the list which will help us for tokenization
words = []
for i in df.headline.values:
    words.append(i.split())
print("splitted words:",words[:5])

# use genism lib for word2vec wordembedding
import gensim
#Dim for max embedding
EMBEDDING_DIM = 200

#lets create word2vec model
w2v_model = gensim.models.Word2Vec(sentences = words , size=EMBEDDING_DIM , window = 5 , min_count = 1)
# import keras.preprocessing lib for token
tokenizer = text.Tokenizer(num_words=38071)
tokenizer.fit_on_texts(words)
tokenized_traindata = tokenizer.texts_to_sequences(words)
x = sequence.pad_sequences(tokenized_traindata, maxlen = 20)
print("before tokenization:",len(w2v_model.wv.vocab))
#  vocab size increases by 1
vocab_size = len(tokenizer.word_index) + 1
print("after tokenization, vocab_size:", vocab_size)
#generate weightmatrix using numpy zeros 
weight_matrix=np.zeros((vocab_size, EMBEDDING_DIM))

#lets fill each zeros with value model
for word, k in tokenizer.word_index.items():
        weight_matrix[k] = w2v_model[word]

#lets split dataset into train and test

x_train, x_test, y_train, y_test = train_test_split(x, df.is_sarcastic , test_size = 0.3 , random_state = 0)

#define dnn model
model = Sequential()
#adding embeddidng layers using bidirectional LSTM
model.add(Embedding(vocab_size, output_dim=EMBEDDING_DIM, weights=[weight_matrix], input_length=20, trainable=True))

model.add(Bidirectional(LSTM(units=128 , recurrent_dropout = 0.2 , dropout = 0.2,return_sequences = True)))
model.add(Bidirectional(GRU(units=64 , recurrent_dropout = 0.1 , dropout = 0.1)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=keras.optimizers.Adam(lr = 0.04), loss='binary_crossentropy', metrics=['acc'])

del weight_matrix
model.summary()

history = model.fit(x_train, y_train, batch_size = 128 , validation_data = (x_test,y_test) , epochs = 5)
epochs = [i for i in range(5)]
fig , ax = plt.subplots(1,2)
train_acc = history.history['acc']
train_loss = history.history['loss']
val_acc = history.history['val_acc']
val_loss = history.history['val_loss']
fig.set_size_inches(20,10)

ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
ax[0].plot(epochs , val_acc , 'ro-' , label = 'Testing Accuracy')
ax[0].set_title('Training & Testing Accuracy')
ax[0].legend()
ax[0].set_xlabel("Epochs")
ax[0].set_ylabel("Accuracy")

ax[1].plot(epochs , train_loss , 'go-' , label = 'Training Loss')
ax[1].plot(epochs , val_loss , 'ro-' , label = 'Testing Loss')
ax[1].set_title('Training & Testing Loss')
ax[1].legend()
ax[1].set_xlabel("Epochs")
ax[1].set_ylabel("Loss")
plt.show()
predict = model.predict_classes(x_test)
predict[:10]
#lets use confusion matrix to check on predictions
conmat=confusion_matrix(y_test,predict)
conmat = pd.DataFrame(conmat , index = ['Not Sarcastic','Sarcastic'] , columns = ['Not Sarcastic','Sarcastic'])
plt.figure(figsize = (5,5))
sns.heatmap(conmat,cmap= "Blues", linecolor = 'black' , linewidth = 1 , annot = True, fmt='' , xticklabels = ['Not Sarcastic','Sarcastic'] , yticklabels = ['Not Sarcastic','Sarcastic'])
print(classification_report(y_test, predict, target_names = ['Not Sarcastic','Sarcastic']))
