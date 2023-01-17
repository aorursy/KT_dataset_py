# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd



df = pd.read_csv('../input/lyrics-generation/train_lyrics_1000.csv')

df1 =pd.read_csv('../input/lyrics-generation/valid_lyrics_200.csv')

print(df1.head())

print(df1['lyrics'][0])



df.head()
from sklearn.preprocessing import LabelEncoder

from joblib import dump, load

import numpy as np



X_train = df['lyrics'].values 



y_train = df['mood'].values



X_valid = df['lyrics'].values 



y_valid = df['mood'].values



print('before embeding training data: %s ...' %y_train[:5])

print('before encoding validation data: %s ...' %y_valid[:5])



le = LabelEncoder()

le.fit(y_train)

le.fit(y_valid)

y_train = le.transform(y_train)

y_valid =le.transform(y_valid)



print('after encoding train data: %s ...' %y_train[:5])

print('after encoding validation data: %s ...' %y_valid[:5])
dump(le, 'label_encoder.joblib') 
from tensorflow.python.keras.preprocessing.text import Tokenizer

from tensorflow.python.keras.preprocessing.sequence import pad_sequences

tokenizer_obj = Tokenizer()

total_lyrics =X_train + X_valid

tokenizer_obj.fit_on_texts(total_lyrics)
#pad sequences

max_length = max([len(s.split()) for s in total_lyrics])
#define vocabulary size

vocab_size = len(tokenizer_obj.word_index) + 1
X_train_tokens =tokenizer_obj.texts_to_sequences(X_train)

X_valid_tokens =tokenizer_obj.texts_to_sequences(X_valid)

X_train_pad =pad_sequences(X_train_tokens,maxlen = max_length,padding ="post")

X_valid_pad =pad_sequences(X_valid_tokens,maxlen = max_length,padding ="post")
from keras.models import Sequential

from keras.layers import  Dense, Embedding, LSTM,GRU

from keras.layers.embeddings import Embedding  
EMBEDDING_DIM = 100

print("Build model.............")

model = Sequential( ) 

model.add(Embedding(vocab_size,EMBEDDING_DIM,input_length=max_length)) 

model.add(GRU(units=32, dropout=0.2,recurrent_dropout =0.2))

model.add(Dense(1,activation='sigmoid'))

#using different optimizers and different optimizer configs 

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary
print("trainning the model....")

model.fit(X_train_pad,y_train, batch_size= 128, epochs=25, validation_data =(X_valid_pad,y_valid),verbose=2)
# validating on test data and obtaning the sentiment of lyrics

test_samples =[df1['lyrics'][0],df1['lyrics'][1],df1['lyrics'][2],df1['lyrics'][3],df1['lyrics'][4],df1['lyrics'][5]]

test_samples_tokens = tokenizer_obj.texts_to_sequences(test_samples)

test_samples_tokens_pad =pad_sequences(test_samples_tokens,maxlen =max_length)

model.predict(test_samples_tokens_pad)
# now trainning our model on word to vec embedding as feature for context



import string 

from nltk.tokenize import word_tokenize 

from nltk.corpus import stopwords 

review_lines = list() 

lines =df['lyrics'].values.tolist() 

for line in lines: 

    tokens = word_tokenize(line) # convert to Lower case 

    tokens = [w. lower() for w in tokens] # remove punctuation from each word

    table=str.maketrans('','',string.punctuation )

    stripped =[w.translate(table) for w in tokens]

    

    words =[word for word in stripped if word.isalpha()] 

    stop_words = set(stopwords.words( 'english') ) 

    words = [w for w in words if not w in stop_words] 

    review_lines.append(words) 
len(review_lines)
import gensim # train word2vec model 

model = gensim.models.Word2Vec(sentences=review_lines,size =EMBEDDING_DIM, window=5,workers=4,min_count=1) 

# vocab size '

words = list(model.wv.vocab)

print( 'Vocabulary size: %d' %len(words))
from gensim.scripts.glove2word2vec import glove2word2vec

glove_input_file = '../input/nlpword2vecembeddingspretrained/glove.6B.100d.txt'

word2vec_output_file = 'glove.6B.100d.txt.word2vec'

glove2word2vec(glove_input_file, word2vec_output_file)

# num_features = 300  # Word vector dimensionality

# min_word_count = 40 # Minimum word count

# num_workers = 4     # Number of parallel threads

# context = 10        # Context window size

# downsampling = 1e-3 # (0.001) Downsample setting for frequent words



# # Initializing the train model

# from gensim.models import word2vec

# print("Training model....")

# model = word2vec.Word2Vec(sentences,\

#                           workers=num_workers,\

#                           size=num_features,\

#                           min_count=min_word_count,\

#                           window=context,

#                           sample=downsampling)



# # To make the model memory efficient

# model.init_sims(replace=True)



# # Saving the model for later use. Can be loaded using Word2Vec.load()

# model_name = "300features_40minwords_10context"

# model.save(model_name)