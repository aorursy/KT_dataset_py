import pandas as pd
train = pd.read_csv('../input/fake-news/train.csv')
test = pd.read_csv('../input/fake-news/test.csv')
train.head()
print(train.shape)
print(test.shape)
len(train['title'][0])
len(train['text'][0])
data = train.append(test)
data.shape
data.isna().sum()
data = data.dropna()
data.isna().sum()
data.shape
import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(5,5))
sns.countplot(data=data,x='label')
X = data.drop('label',axis=1)
y = data['label']
import tensorflow 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Dense,LSTM,Dropout
from tensorflow.keras.preprocessing.text import one_hot
messages = X.copy()
messages.reset_index(inplace=True)
import nltk
import re
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
lem = WordNetLemmatizer()
corpus = []

for i in range(len(messages)):
    text = re.sub('[^a-zA-Z]',' ',messages['title'][i])
    text = text.lower()
    text = text.split()
    
    text = [lem.lemmatize(word) for word in text if not word in nltk.corpus.stopwords.words('english')]
    text = ' '.join(text)
    corpus.append(text)
corpus[:5]
word2count = {}

for sentence in corpus:
    words = nltk.word_tokenize(sentence)
    
    for word in words:
        if word not in word2count.keys():
            word2count[word] = 1
        else:
            word2count[word] += 1
len(word2count)
import heapq
word_freq = heapq.nlargest(5000,word2count,key=word2count.get)
word_freq
vocab_size = len(word_freq)
onehot_corpus = []
for sentences in corpus:
    Z = one_hot(sentences,vocab_size)
    onehot_corpus.append(Z)
onehot_corpus
length = 20
embedded_sents = pad_sequences(onehot_corpus,padding='pre',maxlen=length)
print(embedded_sents)
embedded_sents[0]
embedding_feature_vectors = 40
model = Sequential()
model.add(Embedding(vocab_size,embedding_feature_vectors,input_length=length))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dropout(0.3))
model.add(Dense(1,activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
import numpy as np
X_final = np.asarray(embedded_sents)
y_final = np.asarray(y)
X_final
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_final,y_final,test_size=0.2)
history = model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=20,batch_size=128)
plt.style.use('dark_background')
fig, ax = plt.subplots(2, 2, figsize=(10, 10))
sns.lineplot(x=np.arange(1, 21), y=history.history.get('loss'), ax=ax[0, 0])
sns.lineplot(x=np.arange(1, 21), y=history.history.get('accuracy'), ax=ax[0, 1])
sns.lineplot(x=np.arange(1, 21), y=history.history.get('val_loss'), ax=ax[1, 0])
sns.lineplot(x=np.arange(1, 21), y=history.history.get('val_accuracy'), ax=ax[1, 1])
ax[0, 0].set_title('Training Loss vs Epochs')
ax[0, 1].set_title('Training Accuracy vs Epochs')
ax[1, 0].set_title('Validation Loss vs Epochs')
ax[1, 1].set_title('Validation Accuracy vs Epochs')
plt.show()
y_pred = model.predict_classes(X_test)
y_pred
from sklearn.metrics import accuracy_score,confusion_matrix
score = accuracy_score(y_test,y_pred)
print(score)
sm = confusion_matrix(y_test,y_pred)
sm
df = pd.DataFrame(sm,columns=np.unique(y_test), index = np.unique(y_test))
df
df.index.name = 'Actual'
df.columns.name = 'Predicted'
plt.figure(figsize = (5,5))

sns.set(font_scale=1.4)
sns.heatmap(df, cmap="Greens",annot=True,annot_kws={"size": 16})
