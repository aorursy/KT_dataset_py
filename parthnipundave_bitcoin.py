import pandas as pd
import numpy as np
import re
data = pd.read_csv('../input/bitcoin-tweets/bitcointweets.csv',header=None)
data = data.loc[:,[1,7]]
data.columns=['tweets','label']
data.head()
data['label'] =data['label'].str.replace(r'([\[\]\'])','')
data.head()
import nltk
import string
from nltk import PorterStemmer
stem = PorterStemmer()
from nltk.corpus import stopwords
def clean_data(text):
    text  = text.split()
    noUser = [word for word in text if word not in (re.findall(r'(^@.+)',str(word)))  ]
    
    clean = [word for word in noUser if word not in string.punctuation]
    clean = [word for word in clean if word not in re.findall(r'(\(\))',str(word))]
    clean = ' '.join(clean)
    
#     clean = clean.lower()
    clean = [word.lower() for word in clean if clean not in stopwords.words('english') ]
    clean = ''.join(clean)
    
    return clean
data['tweets']=data['tweets'].apply(clean_data)

def removeLinks(text):
    text = text.split()
    clean = [word for word in text if word not in (re.findall(r'^http.+',str(word)))]
    return clean
data['tweets']=data['tweets'].apply(removeLinks)
!pip install --upgrade gensim
from gensim.models import Word2Vec
model = Word2Vec(data['tweets'],size=300,window=5,workers=3,sg=1)
model.wv.most_similar('bitcoin')
words = list(model.wv.vocab)
print('Vocabulory size ',len(words))
file = 'word2vec.txt'
model.wv.save_word2vec_format(file,binary=False)
import os
embeddings_index = {}
f = open(os.path.join('','word2vec.txt'),encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    embeddings_index[word]=coefs
f.close()

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
token_obj = Tokenizer()
token_obj.fit_on_texts(data['tweets'])
sequences = token_obj.texts_to_sequences(data['tweets'])
word_index = token_obj.word_index
print('Found ',len(word_index),' unique tokens')
seq_pad = pad_sequences(sequences,maxlen=len(data['tweets'].max()))
print('shape of tweets',seq_pad.shape)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
labels = ohe.fit_transform(data['label'].to_numpy().reshape(-1,1)).toarray()

num_words= len(word_index)+1
embedding_matrix = np.zeros((num_words,300))
for word,i in word_index.items():
    if i> num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
        

print(num_words)
from keras.models import Sequential
from keras.layers import Conv1D,Dense,Embedding,LSTM,MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
model = Sequential()
embedding_layer = Embedding(num_words,300,embeddings_initializer=Constant(embedding_matrix),input_length=20,trainable=False)
model.add(embedding_layer)
model.add(Conv1D(64,5, padding='valid',activation='relu',strides=1))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(128,dropout=0.2))
model.add(Dense(3,activation='sigmoid'))
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()
train_x = seq_pad[:-100]
test_x = seq_pad[-100:]
train_y =labels[:-100:]
test_y = labels[-100:]
model.fit(train_x,train_y,epochs=50,batch_size=32)
predict = model.predict(test_x)
from sklearn.metrics import confusion_matrix,accuracy_score
acc =[]
for i in range(len(predict)):
    if predict[i].argmax()==test_y[i].argmax():
        acc.append(i)
print('Test Accuracy ',len(acc)/len(predict)*100,'%')
