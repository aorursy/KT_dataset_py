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
from gensim.models import word2vec
tweets = pd.read_csv('../input/review-battery/train.csv')
tweets.head()
train=tweets.loc[:,["rating","lemmatized"]]
train_k=tweets.loc[:,["rating","lemmatized"]]
x_axis= list((set(train["rating"])))
print(x_axis)
y_axis = train.groupby("rating").count()
print(y_axis)
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
   # sentences = sent_tokenize(train["text"][i])
    words = word_tokenize(train["lemmatized"][i])
    words = [word for word in words if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    porter = PorterStemmer()
    stemmed_words= [ porter.stem(word) for word in words]
    print(stemmed_words)
    train["lemmatized"][i]=stemmed_words
li=list(train["lemmatized"])
li
embedded_index = word2vec.Word2Vec(li, min_count=1,size= 100,workers=3, window =2, sg = 0)
embedded_index ["batteri"]
                                                       
embedded_index.similarity("batteri","good")
model=embedded_index
from sklearn.manifold import TSNE
def display_closestwords_tsnescatterplot(model, word, size):
    
    arr = np.empty((0,size), dtype='f')
    word_labels = [word]
    close_words = model.similar_by_word(word)
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
            wrd_vector = model[wrd_score[0]]
            word_labels.append(wrd_score[0])
            arr = np.append(arr, np.array([wrd_vector]), axis=0)

    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
        plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
        plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()
display_closestwords_tsnescatterplot(model,"cell", 100)
tokenizer = Tokenizer(num_words = 4000)
tokenizer.fit_on_texts(train_k["lemmatized"])
sequence = tokenizer.texts_to_sequences(train_k["lemmatized"])
max_seq_len = 1000
padded_seq = pad_sequences(sequence , maxlen = max_seq_len )
X_train,X_test,Y_train,Y_test = train_test_split(padded_seq,Y ,train_size = 0.80,random_state= 37)

Y_train.shape
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
embed_num_dims = 100
embedded_index['good']
embedding_matrix = np.zeros((len(index_of_words) + 1, embed_num_dims))

tokens = []
labels = []

for word,i in index_of_words.items():
    temp = embedded_index["word"]
    if temp is not None:
        embedding_matrix[i] = temp
        
#for plotting
        tokens.append(embedding_matrix[i])
        labels.append(word)
embedding_matrix.shape
len(index_of_words)
embed_num_dims
max_seq_len
embedd_layer = Embedding(len(index_of_words) + 1 , embed_num_dims , input_length = max_seq_len , weights = [embedding_matrix])
model = Sequential()
model.add(embedd_layer)
model.add(Bidirectional(LSTM(100 , return_sequences = True , dropout = 0.5 , recurrent_dropout = 0.5)))
model.add(GlobalMaxPooling1D())
model.add(Dense(5,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(5,activation = 'sigmoid'))
model.summary()
from keras.optimizers import Adam
add = Adam(learning_rate=0.1,
    beta_1=0.99,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=True,)
model.compile(loss = 'categorical_crossentropy' , optimizer = add , metrics = ['accuracy'])
hist = model.fit(X_train,Y_train,epochs = 30, batch_size = 250, validation_data = (X_test,Y_test))

