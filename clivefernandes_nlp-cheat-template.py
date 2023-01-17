import warnings
warnings.filterwarnings('ignore')

import warnings
warnings.filterwarnings('ignore')

# Modules for data manipulation
import numpy as np
import pandas as pd
import re

# Modules for visualization
import matplotlib.pyplot as plt
import seaborn as sb

# Tools for preprocessing input data
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
# Tools for creating ngrams and vectorizing input data
import gensim

from gensim.models import Word2Vec, Phrases



# Tools for building a model
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing import text
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Tools for assessing the quality of model prediction
from sklearn.metrics import accuracy_score, confusion_matrix


import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
dataset_path = '../input/imdb-dataset-of-50k-movie-reviews/IMDB Dataset.csv'
dataset= pd.read_csv(dataset_path)
dataset.head()
dataset.replace({'positive':1,'negative':0},inplace = True)
dataset.head()

#helper functions for lemmatizations
def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
              'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n' 

def clean_text(x):
    
    
    
    # remove html tags
    regex = re.compile('<.*?>')
    input =  re.sub(regex, '', x)

    #remove punctuations, numbers.
    input = re.sub('[!@#$%^&*()\n_:><?\-.{}|+-,;""``~`—]|[0-9]|/|=|\[\]|\[\[\]\]',' ',input)
    input = re.sub('[“’\']','',input)   
    
    
    #lemmatise 
    ls = list(wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) for word, tag in pos_tag(word_tokenize(input)))
    
    
    #remove stopwords
    return_str = ''
    for word in ls:
       #if word its a long word with single character eg.aaaaaa remove it 
        if word not in stop_dict and len(set(word)) > 2:
            return_str +=word.lower() + " "
       

    
    #lemmatize the text.
    

    return return_str


wnl = WordNetLemmatizer()

stop_dict = stopwords.words('english')

tmp_sent  = "AAAAAA <html> <h1> run <i>running</i> ban banned dancing dance 1 2 3  4   5 5  5 !@#$%^&*(){{:><<< MMM<>?PLOKIU}} </h1> </html>"


clean_text(tmp_sent)



dataset['review'] = dataset['review'].map(clean_text)
x  = dataset['review']
y = dataset['sentiment']
y.shape

tokenizer = text.Tokenizer(num_words=1000)
tokenizer.fit_on_texts(['sample text'])

metrix = tokenizer.texts_to_matrix(['sample text'])



tokenizer = text.Tokenizer(num_words=1000)
tokenizer.fit_on_texts(['sample text'])

metrix = tokenizer.texts_to_matrix(['sample text'],mode = 'tfidf')


tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(x)
sequences = tokenizer.texts_to_sequences(x)

tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(x)

sequences = tokenizer.texts_to_sequences(x)


# !wget  http://nlp.stanford.edu/data/glove.6B.zip
# !unzip glove.6B.zip    
import numpy as np
glove_embedding  = 'glove.6B.100d.txt'

embeddings_index = {}

file =  open(glove_embedding,'r')
    
    
for line in file:
    
    word,embd = line.split(maxsplit = 1)
  
    embd = np.fromstring(embd,'f',sep = ' ')
    
    embeddings_index[word] = embd
    
    
    
file.close()    
    


embedding_matrix = np.zeros((max_word_size,100)) 

print(embedding_matrix.shape)




for index,word in tokenizer.index_word.items():
    
    embd =  embeddings_index.get(word)
    
    if embd is not None:
        embedding_matrix[index] = embd
        
        
# embedding_layer = Embedding(vocab_size, 150, weights=[embedding_vectors], input_length=370, trainable=True)        

bigrams = Phrases(data)
trigrams = Phrases(data)
word2vec_model = Word2Vec(
    sentences = trigrams[data],
    size = 300,
    min_count=3, window=5, workers=4)
sequences = []
for i in tqdm.tqdm(data):
    sent = []
    for word in i:
        if word in word2vec_model.wv.vocab:
            sent.append(word2vec_model.wv.vocab[word].index)
    sequences.append(sent)    
    



#replace your_sequences with the embedding matrix you choose form above.
vocab_size = len(tokenizer.word_index) + 1
X_pad =  pad_sequences(sequences,maxlen = 1000,padding = 'post',value = vocab_size - 1)

X_train, X_test, y_train, y_test = train_test_split(
    X_pad,
    y,
    test_size=0.05,
    shuffle=True,
    random_state=42)
model = Sequential()
model.add(Embedding(input_dim=vocab_size,output_dim=128,input_length = 1000))
model.add(Flatten())
model.add(Dense(50,activation = 'relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss="binary_crossentropy",
    optimizer= 'adam',
    metrics=['accuracy'])

model.summary()

print(model.input_shape)

model.fit(X_train,y_train,validation_data = (X_test,y_test),epochs = 1)


model_lstm = Sequential()
model_lstm.add(Embedding(input_dim=vocab_size,output_dim=128,input_length = 1000))
model_lstm.add(LSTM(60, return_sequences = True))
model_lstm.add(GlobalMaxPool1D())
# model_lstm.add(Flatten())
model_lstm.add(Dropout(0.1))
model_lstm.add(Dense(50,activation = 'relu'))
model_lstm.add(Dropout(0.1))
model_lstm.add(Dense(1, activation='sigmoid'))

model_lstm.compile(
    loss="binary_crossentropy",
    optimizer= 'adam',
    metrics=['accuracy'])

model_lstm.summary()

print(model_lstm.input_shape)

model_lstm.fit(X_train,y_train,validation_data = (X_test,y_test),epochs = 1)


model_cnn = Sequential([
    
    Embedding(input_dim=vocab_size,output_dim=128,input_length = 1000),
    Conv1D(16,8,activation = 'relu'),
    Dropout(0.5),
    MaxPool1D(2),
    Flatten(),
    Dropout(0.5),
    Dense(64,activation = 'relu'),
    Dense(1,activation = 'sigmoid')
    ])


model_cnn.compile(
    loss="binary_crossentropy",
    optimizer= 'adam',
    metrics=['accuracy'])

model_cnn.summary()

print(model_cnn.input_shape)

model_cnn.fit(X_train,y_train,validation_data = (X_test,y_test),epochs = 1)


model_cnn_lstm =  Sequential([
    Embedding(input_dim=vocab_size,output_dim=128,input_length = 1000),
    Conv1D(16,5,activation = 'relu',padding = 'same',strides = 1),
    MaxPool1D(2),
    LSTM(64,name = 'lstm_1'),
    Dropout(0.7),
    Dense(1,activation = 'sigmoid')
    ])


model_cnn_lstm.compile(
    loss="binary_crossentropy",
    optimizer= 'adam',
    metrics=['accuracy'])


# model.fit()
model_cnn_lstm.summary()
# model_cnn_lstm.summary()
# print(model_cnn_lstm.get_layer('lstm_1').input_shape )


model_cnn_lstm.fit(X_train,y_train,validation_data = (X_test,y_test),epochs = 1)


import tensorflow as tf

model_lstm = Sequential()
model_lstm.add(Embedding(input_dim=vocab_size,output_dim=128,input_length = 1000))
model_lstm.add(tf.compat.v1.keras.layers.CuDNNLSTM(60, return_sequences = True))
model_lstm.add(GlobalMaxPool1D())
# model_lstm.add(Flatten())
model_lstm.add(Dropout(0.1))
model_lstm.add(Dense(50,activation = 'relu'))
model_lstm.add(Dropout(0.1))
model_lstm.add(Dense(1, activation='sigmoid'))

model_lstm.compile(
    loss="binary_crossentropy",
    optimizer= 'adam',
    metrics=['accuracy'])

model_lstm.summary()

print(model_lstm.input_shape)

model_lstm.fit(X_train,y_train,validation_data = (X_test,y_test),epochs = 1)


# def plot_confusion_matrix(y_true, y_pred, ax, class_names, vmax=None,
#                           normed=True, title='Confusion matrix'):
#     matrix = confusion_matrix(y_true,y_pred)
#     if normed:
#         matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
#     sb.heatmap(matrix, vmax=vmax, annot=True, square=True, ax=ax,
#                cmap=plt.cm.Blues_r, cbar=False, linecolor='black',
#                linewidths=1, xticklabels=class_names)
#     ax.set_title(title, y=1.20, fontsize=16)
#     #ax.set_ylabel('True labels', fontsize=12)
#     ax.set_xlabel('Predicted labels', y=1.10, fontsize=12)
#     ax.set_yticklabels(class_names, rotation=0)
# fig, axis1 = plt.subplots(nrows=1, ncols=1)
# plot_confusion_matrix([1,1,1,1,0,0], [1,1,0,1,0,0], ax=axis1,
#                       title='Confusion matrix (train data)',
#                       class_names=['Positive', 'Negative'])