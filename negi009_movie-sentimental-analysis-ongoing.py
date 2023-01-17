# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

# add glove6b300dtxt dataset 



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/train.tsv',delimiter='\t')

test =pd.read_csv('../input/movie-review-sentiment-analysis-kernels-only/test.tsv',delimiter='\t')
train.head(10)
from sklearn.preprocessing import OneHotEncoder
# -1 for len(train)  One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.

train.Sentiment.values.reshape(-1,1)
train.Sentiment.values.shape
train.iloc[192]
ohe=OneHotEncoder(sparse=False)

# -1 for len(train)  One shape dimension can be -1. In this case, the value is inferred from the length of the array and remaining dimensions.

# OneHotEncoder Expected 2D array

ohe=ohe.fit(train.Sentiment.values.reshape(-1,1))

print(train.Sentiment.values[192])

print('sentimental distribution\n{}'.format(train.Sentiment.value_counts()/len(train)*100))
len(set(train['Phrase']).intersection(set(test['Phrase'])))/len(test)*100
from sklearn.feature_extraction.text import  CountVectorizer



cv1= CountVectorizer()

cv2= CountVectorizer()



cv1.fit(train.Phrase)

cv2.fit(test.Phrase)

print('train total vocabulary size = {}'.format(len(cv1.vocabulary_)))

print('test total vocabulary size = {}'.format(len(cv2.vocabulary_)))



print('comman word in both ={}'.format(len(set(cv1.vocabulary_.keys()).intersection(set(cv2.vocabulary_.keys())))))

groupby=train.groupby("SentenceId")

groupby.count()[:3]

def tranfer(data):

    data['Phrase_count']=data.groupby("SentenceId")['Phrase'].transform('count')

    data['word_count']=data['Phrase'].apply(lambda x :len(x.split(' ')) )

    data['upper_char']=data['Phrase'].apply(lambda x : x.lower()!=x)

    data['start_comma']=data['Phrase'].apply(lambda x : x.startswith(','))

    data['sentence_end']=data['Phrase'].apply(lambda x :x.endswith('.') )

    data['sentence_start']=data['Phrase'].apply(lambda x :x[0].upper()==x[0] )

    data["Phrase"] = data["Phrase"].apply(lambda x: x.lower())

    return data

train = tranfer(train)

test = tranfer(test)

        
NUM_FOLDS = 5



train["fold_id"] = train["SentenceId"].apply(lambda x: x%NUM_FOLDS)
train.groupby("Sentiment")[train.columns[4:]].mean()
NUM_FOLDS = 5



train["fold_id"] = train["SentenceId"].apply(lambda x: x%NUM_FOLDS)

# tranfer knowlege from pre train databset with 300 dimention , with diff features like gender , royal etc 

# glove Embeddig

EMBEDDING_FILE =  '../input/glove6b300dtxt/glove.6B.300d.txt'

f=open(EMBEDDING_FILE)

for line in f:

    # split each iteam in first line 

    value=line.split(' ')

    # first time is wrord 

    word = value[0]

    print('1st word=',word)

    print(value[:20])

    print(line[:20])

    break;

    

EMBEDDING_DIM=300
f=open(EMBEDDING_FILE)

# unin test and train unique

all_word = set(cv1.vocabulary_.keys()).union(set(cv2.vocabulary_.keys()))

# store 

embedding_index={}

for line in f:

    value=line.split(' ')

    word = value[0]

    if word in all_word:

        coef =value[1:]

        embedding_index[word]=coef;

    f.close

print('word not in Glove ={}'.format(len(set(all_word)-set(embedding_index))))



    
train.head()
max(max(train.Phrase.apply(lambda x : len(x.split(' ')))),max(test.Phrase.apply(lambda x : len(x.split(' ')))))
MAX_SEQUENCE_LENGTH=56
test.head()
from keras.preprocessing.text import Tokenizer

from keras.preprocessing.sequence import pad_sequences

import textblob

MAX_SEQUENCE_LENGTH = 60



tokenizer = Tokenizer()

# numbering each word in vocab

tokenizer.fit_on_texts(np.append(train["Phrase"].values, test["Phrase"].values))

word_index = tokenizer.word_index



nb_words = len(word_index) + 1

embedding_matrix = np.random.rand(nb_words, EMBEDDING_DIM + 2)



for word, i in word_index.items():

    embedding_vector = embedding_index.get(word)

    sent = textblob.TextBlob(word).sentiment

    if embedding_vector is not None:

        embedding_matrix[i] = np.append(embedding_vector, [sent.polarity, sent.subjectivity])

    else:

        embedding_matrix[i, -2:] = [sent.polarity, sent.subjectivity]

        

seq = pad_sequences(tokenizer.texts_to_sequences(train["Phrase"]), maxlen=MAX_SEQUENCE_LENGTH)

test_seq = pad_sequences(tokenizer.texts_to_sequences(test["Phrase"]), maxlen=MAX_SEQUENCE_LENGTH)
textblob.TextBlob('good').sentiment
from keras.layers import  *

from keras.models import Model

from keras.callbacks import EarlyStopping





dense_features = train.columns[4:10]



def build_model():

    embading_layer =Embedding(input_dim=nb_words,output_dim=EMBEDDING_DIM+2,

                              weights=[embedding_matrix],

                              input_length=MAX_SEQUENCE_LENGTH,

                              trainable=True)

    dropout = Dropout(0.2)

    mask =Masking()

    lstm= LSTM(50)

    

    seq_input =Input(shape=(MAX_SEQUENCE_LENGTH,),dtype='int32')

    dense_input =Input(shape=(len(dense_features),))

    dense_vect =BatchNormalization()(dense_input)

    phrase_vect=lstm(mask(dropout(embading_layer(seq_input))))

    

    f_vec= concatenate([dense_vect,phrase_vect])

    f_vec= Dense(50,activation='relu')(f_vec)

    f_vec=Dense(20,activation='relu')(f_vec)

    output = Dense(5,activation='softmax')(f_vec)

    

    model = Model(inputs=[seq_input,dense_input],outputs=[output])

    

    return model
test_preds = np.zeros((test.shape[0], 5))



for i in range(NUM_FOLDS):

    print("FOLD", i+1)

    

    print("Splitting the data into train and validation...")

    train_seq, val_seq = seq[train["fold_id"] != i], seq[train["fold_id"] == i]

    train_dense, val_dense = train[train["fold_id"] != i][dense_features], train[train["fold_id"] == i][dense_features]

    y_train = ohe.transform(train[train["fold_id"] != i]["Sentiment"].values.reshape(-1, 1))

    y_val = ohe.transform(train[train["fold_id"] == i]["Sentiment"].values.reshape(-1, 1))

    

    print("Building the model...")

    model = build_model()

    model.compile(loss="categorical_crossentropy", optimizer="nadam", metrics=["acc"])

    

    early_stopping = EarlyStopping(monitor="val_acc", patience=2, verbose=1)

    

    print("Training the model...")

    model.fit([train_seq, train_dense], y_train, validation_data=([val_seq, val_dense], y_val),

              epochs=15, batch_size=1024, shuffle=True, callbacks=[early_stopping], verbose=1)

    

    print("Predicting...")

    test_preds += model.predict([test_seq, test[dense_features]], batch_size=1024, verbose=1)

    print()

    

test_preds /= NUM_FOLDS
dense_features