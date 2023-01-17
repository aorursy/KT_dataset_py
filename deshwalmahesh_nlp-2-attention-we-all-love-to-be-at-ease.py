# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import nltk 
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from string import punctuation

from keras import backend as K
from keras import initializers, regularizers, constraints

from keras.layers import Dense, Input, LSTM, Bidirectional,Dropout, Embedding, BatchNormalization, Layer
from keras.models import Model
from keras.callbacks import ModelCheckpoint,EarlyStopping, ReduceLROnPlateau
from keras.initializers import Constant
from keras.layers.merge import add
from keras.optimizers import Adam

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import np_utils

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Any results you write to the current directory are saved as output.
seed = 13 # reproducible results

np.random.seed(seed) # numpy seed

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

signs = list(punctuation) # special characters 
stop_words = list(stopwords.words('english')) # stop words like `a,an,the,or,at` etc

wordnet_lemmatizer = WordNetLemmatizer() # make the word cooking,cooks,cooked -> cook (in ideal case)
df = pd.read_csv('/kaggle/input/bbc-fulltext-and-category/bbc-text.csv')
df.head()
df['category'].value_counts().plot(kind='pie',autopct='%.2f%%') # almost equally distributed
def clean_articles(df):
    '''
    Method to clean the existing DataFrame
    '''

    # remove all the new lines and spaces if there are any : - \r and \n
    print('Step 1: Replacing...')
    
    df['text'] = df['text'].str.replace("\r", " ")
    df['text'] = df['text'].str.replace("\n", " ")
    df['text'] = df['text'].str.replace("    ", " ")
    df['text'] = df['text'].str.replace('"', '') # remove double quotes
    df['text'] = df['text'].str.lower()  # make all the words as lower case
    
    for sign in signs:
        df['text'] = df['text'].str.replace(sign, '') # remove any special punctuations
        
    # remove Deshwal's the trailing s as it does not add any information in classification
    df['text'] = df['text'].str.replace("'s", "")


    print('Step 2: Lemmatizing......')
    
    nrows = len(df)
    lemmatized_text_list = []

    for row in range(nrows):
        lemmatized_list = [] # Create an empty list containing lemmatized words
        text = df.loc[row]['text'] # Save the text and its words into an object
        text_words = text.split(" ")

        for word in text_words:  # Iterate through every word to lemmatize
            lemmatized_list.append(wordnet_lemmatizer.lemmatize(word, pos="v"))

        lemmatized_text = " ".join(lemmatized_list)  # Join the list to get a string
        lemmatized_text_list.append(lemmatized_text) # Append to the list containing the texts

    df['text'] = lemmatized_text_list
    
    
    print('Step 3: Removing Stop Words....')
    
    for stop_word in stop_words:
        re_sw = r"\b" + stop_word + r"\b"
        df['text'] = df['text'].str.replace(re_sw, '')

clean_articles(df)
def tokenize_articles(df):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df.text)
    df['words'] = tokenizer.texts_to_sequences(df.text)
    
    return tokenizer
tokenizer = tokenize_articles(df)

df['article_length'] = df.words.apply(lambda i: len(i)) # if article length is less than 10, drop it
df = df[df['article_length']>=10]
df.article_length.describe()
maxlen = 275 # 75 percentile is 273
X = list(sequence.pad_sequences(df.words, maxlen=maxlen)) # add padding to the short length articles

df['encoded_cat'] = LabelEncoder().fit_transform(df['category']) # convert category as 0,1,2,3,4
word_index = tokenizer.word_index

EMBEDDING_DIM = 100

embeddings_index = {}
with open('/kaggle/input/glove-global-vectors-for-word-representation/glove.6B.100d.txt') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print(f'Unique tokens: {len(word_index)}')
print(f'Total Word Vectors: {len(embeddings_index)}')
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
X = np.array(X)
Y = np_utils.to_categorical(df['encoded_cat'].tolist())

x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=seed)
class Attention(Layer):
    def __init__(self,**kwargs):
        super(Attention,self).__init__(**kwargs)

    def build(self,input_shape):
        self.W=self.add_weight(name="att_weight",shape=(input_shape[-1],1),initializer="normal")
        self.b=self.add_weight(name="att_bias",shape=(input_shape[1],1),initializer="zeros")        
        super(Attention, self).build(input_shape)

    def call(self,x):
        et=K.squeeze(K.tanh(K.dot(x,self.W)+self.b),axis=-1)
        at=K.softmax(et)
        at=K.expand_dims(at,axis=-1)
        output=x*at
        return K.sum(output,axis=1)

    def compute_output_shape(self,input_shape):
        return (input_shape[0],input_shape[-1])

    def get_config(self):
        return super(Attention,self).get_config()
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
embedding_layer = Embedding(len(word_index)+1,EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),input_length=maxlen,
                            trainable=False) # do not train as these are GloVe pre trained
inp = Input(shape=(maxlen,), dtype='int32')
embedding= embedding_layer(inp)
attention_out = Attention()(embedding)
flat = Dense(254, activation='relu')(attention_out)
flat = Dropout(0.47)(flat)
flat = BatchNormalization()(flat)
out = Dense(5,activation='softmax')(flat)

AttentionModel = Model(inputs=inp, outputs=out)
AttentionModel.compile(loss='categorical_crossentropy', optimizer='adam', 
                       metrics=['acc',recall_m,precision_m,f1_m])

AttentionModel.summary()
mcp = ModelCheckpoint(filepath='/kaggle/working/best_weights.h5',verbose=1,save_best_only=True,
                      save_weights_only=True)
es = EarlyStopping(min_delta=0.01,patience=2,verbose=1)
rlp = ReduceLROnPlateau(factor=0.005,patience=1,verbose=1,min_delta=0.001,min_lr=1e-6)

callbacks = [mcp,es,rlp]
training_history = AttentionModel.fit(x_train,y_train,batch_size=32,epochs=10,
                                      validation_data=(x_val, y_val),callbacks=callbacks)
inp = Input(shape=(maxlen,), dtype='int32')
embedding= embedding_layer(inp)
lstm_out = LSTM(254, dropout=0.27, recurrent_dropout=0.25, return_sequences=True)(embedding)
x = Dropout(0.49)(lstm_out)
attention_out = Attention()(x)
flat = Dense(512, activation='relu')(attention_out)
flat = Dropout(0.69)(flat)
flat = BatchNormalization()(flat)
out = Dense(5,activation='softmax')(flat)

WithLSTM = Model(inputs=inp, outputs=out)
WithLSTM.compile(loss='categorical_crossentropy', optimizer='adam', 
                       metrics=['acc',recall_m,precision_m,f1_m])

WithLSTM.summary()
training_history2 = WithLSTM.fit(x_train,y_train,batch_size=64,epochs=5,
                                      validation_data=(x_val, y_val),callbacks=callbacks)