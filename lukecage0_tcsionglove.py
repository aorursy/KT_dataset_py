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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from sklearn.model_selection import train_test_split
from sklearn import metrics
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
import tensorflow as tf
from keras.layers import Embedding,Bidirectional,Dense,Conv1D,Flatten,LSTM,GlobalMaxPooling1D,Dropout
tweets = pd.read_csv('../input/review-battery/train.csv')
tweets.head()

train=tweets.loc[:,["rating","lemmatized"]]
train_k=tweets.loc[:,["rating","lemmatized"]]
train.head()
x_axis= [1,2,3,4,5]
print(x_axis)
y_axis = train.groupby("rating").count()
plt.bar(x_axis,y_axis["lemmatized"])
plt.ylabel("count")
plt.xlabel("rating")
plt.show()
from sklearn.preprocessing import LabelEncoder  
le = LabelEncoder() 
y= le.fit_transform(train['rating']) 
from keras.utils import to_categorical
Y = to_categorical(y)
print(Y.shape)
print(Y)

for i in range(0,len(train["lemmatized"])):
    words = word_tokenize(train["lemmatized"][i])
    words = [word for word in words if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    porter = PorterStemmer()
    stemmed_words= [ porter.stem(word) for word in words]
    train["lemmatized"][i]=stemmed_words
train["lemmatized"][1]
#making dictionary
dictionary = [word  for subli in train["lemmatized"] for word in subli ]
dictionary = list(set(dictionary))
dictionary = sorted(dictionary)
print(dictionary)
tokenizer = Tokenizer(num_words = 4000)
tokenizer.fit_on_texts(train_k["lemmatized"])
sequence = tokenizer.texts_to_sequences(train_k["lemmatized"])
max_seq_len = 1000
padded_seq = pad_sequences(sequence , maxlen = max_seq_len )
padded_seq
X_train,X_test,Y_train,Y_test = train_test_split(padded_seq,Y ,train_size = 0.80,random_state= 37)
print(len(X_train))
print(len(Y_train))
print(len(X_test))
print(len(Y_test))
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)


index_of_words = tokenizer.word_index
print(len(index_of_words))
f = open('/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.100d.txt')
embedd_index = {}
for line in f:
    val = line.split()
    word = val[0]
    coff = np.asarray(val[1:],dtype = 'float')
    embedd_index[word] = coff

f.close()
print('Found %s word vectors.' % len(embedd_index))
embed_num_dims = 100
embedd_index['good']
embedding_matrix = np.zeros((len(index_of_words) + 1, embed_num_dims))

tokens = []
labels = []

for word,i in index_of_words.items():
    temp = embedd_index.get(word)
    if temp is not None:
        embedding_matrix[i] = temp
        
#for plotting
        tokens.append(embedding_matrix[i])
        labels.append(word)
embedding_matrix.shape
from sklearn.manifold import TSNE
#TSNE algorithm used to visualize word embeddings having huge amount (100) dimensions

def tsne():
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens[:200])
    print(new_values.shape)
    
    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16,16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

    
    
    
    
tsne()
#Embedding layer before the actaul BLSTM 
embedd_layer = Embedding(len(index_of_words) + 1 , embed_num_dims , input_length = max_seq_len , weights = [embedding_matrix])
model = Sequential()
model.add(embedd_layer)
model.add(Bidirectional(LSTM( 60, return_sequences = True , dropout = 0.2 , recurrent_dropout = 0.2)))
model.add(GlobalMaxPooling1D())
model.add(Dense(9,activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(5,activation = 'sigmoid'))
model.summary()
from keras.optimizers import Adam
add = Adam(learning_rate=0.01,
    beta_1=0.99,
    beta_2=0.998,
    epsilon=1e-06,
    amsgrad=False,)
model.compile(loss = 'categorical_crossentropy' , optimizer = add , metrics = ['accuracy'])

hist = model.fit(X_train,Y_train,epochs = 60, batch_size = 512, validation_data = (X_test,Y_test))

