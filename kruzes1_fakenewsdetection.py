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
!pip install --upgrade tensorflow
import nltk
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud,STOPWORDS
import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from tensorflow.keras.preprocessing.text import one_hot,Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, Flatten,Embedding,LSTM,Conv1D,Input,MaxPool1D,Bidirectional
#!pip install jupyterthemes
from jupyterthemes import jtplot
jtplot.style(theme = "monokai", context = "notebook", ticks = True, grid = False)
df_true = pd.read_csv("../input/fakenews1/True.csv")
df_fake = pd.read_csv("../input/fakenews1/Fake.csv")
df_true.info()
df_fake.info()
df_fake["isfake"] = 0
df_fake
df_true["isfake"] = 1
df_true
df = pd.concat([df_true, df_fake]).reset_index(drop=True)
df
df
df = df.drop("date", axis =1)
df
df["Original"] = df["title"]+" "+df["text"]
df.head()
df.Original[0]
nltk.download("stopwords")
from nltk.corpus import stopwords
stop_words = stopwords.words("english")
stop_words.extend(["from", "subject", "re", "edu", "use"])
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3 and token not in stop_words:
            result.append(token)
            
    return result
df["clean"] = df["Original"].apply(preprocess)
df["Original"][0]
df["clean"][0]
df
list_of_words = []
for i in df.clean:
    for j in i:
        list_of_words.append(j)
len(list_of_words)
total_words = len(list(set(list_of_words)))
total_words
df["clean_joined"] = df["clean"].apply(lambda x: " ".join(x))
df
plt.figure(figsize = (8,8))
sns.countplot(y = "subject", data = df)
plt.figure(figsize = (8,8))
sns.countplot(y = "isfake", data = df)
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 2000, width = 1600, height = 800, stopwords = stop_words).generate(" ".join(df[df.isfake==1].clean_joined))
plt.imshow(wc,interpolation = "bilinear")
plt.figure(figsize = (20,20))
wc = WordCloud(max_words = 2000, width = 1600, height = 800, stopwords = stop_words).generate(" ".join(df[df.isfake==0].clean_joined))
plt.imshow(wc,interpolation = "bilinear")
maxlen = -1
for doc in df.clean_joined:
    tokens = nltk.word_tokenize(doc)
    if(maxlen<len(tokens)):
        maxlen = len(tokens)
        
print("Max words in doc is:",maxlen)
import plotly.express as px
fig = px.histogram(x = [len(nltk.word_tokenize(x)) for x in df.clean_joined], nbins = 100)
fig.show()
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df.clean_joined , df.isfake , test_size =0.2)
from nltk import word_tokenize
tokenizer = Tokenizer(num_words = total_words)
tokenizer.fit_on_texts(x_train)
train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(x_test)
print("Encoding for:\n", df.clean_joined[0], "\nis:\n", train_sequences[0])
padded_train = pad_sequences(train_sequences, maxlen = 40, padding="post", truncating="post")
padded_test = pad_sequences(test_sequences, maxlen = 40, truncating="post")
for i,doc in enumerate(padded_train[:2]):
    print("Padded encoding for doc",i+1," is:",doc)
model = Sequential()

model.add(Embedding(total_words,output_dim = 128))

# Bi-Directional RNN and LSTM
model.add(Bidirectional(LSTM(128)))


model.add(Dense(128, activation ="relu"))
model.add(Dense(1,activation="sigmoid"))
model.compile(optimizer = "adam",loss = "binary_crossentropy" , metrics = ["acc"])
model.summary()
total_words
y_train = np.asarray(y_train)
model.fit(padded_train, y_train, batch_size = 64, validation_split = 0.1, epochs = 2)
pred = model.predict(padded_test)
prediction = []
for i in range(len(pred)):
    if pred[i].item() > 0.5:
        prediction.append(1)
    else:
        prediction.append(0)
        
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(list(y_test),prediction)

print("Model Accuracy:", accuracy)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(list(y_test),prediction)
plt.figure(figsize = (25,25))
sns.heatmap(cm, annot = True)
category = {0:"Fake News", 1:"Real News"}
