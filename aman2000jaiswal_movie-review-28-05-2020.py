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
#LOADING DATASET

import zipfile

z=zipfile.ZipFile('/kaggle/input/movie-review-sentiment-analysis-kernels-only/train.tsv.zip')

train = pd.read_csv(z.open('train.tsv'),delimiter='\t')
train.head()
# REMOVING EXTRA SPACES



def remove_space(text):

    text=text.strip()

    text=text.split()

    text=[i.lower() for i in text]

    return ' '.join(text)

# TOKENIZATION

import nltk

nltk.download('punkt')



from nltk.tokenize import word_tokenize

def tokenize(text):

    tokens=word_tokenize(text)

    return tokens

# CONTRACTION MAPPING (CHANGE SHORTHAND WORD TO FULL)

import itertools



contraction = {

"'cause": 'because',

',cause': 'because',

';cause': 'because',

"ain't": 'am not',

'ain,t': 'am not',

'ain;t': 'am not',

'ain´t': 'am not',

'ain’t': 'am not',

"aren't": 'are not',

'aren,t': 'are not',

'aren;t': 'are not',

'aren´t': 'are not',

'aren’t': 'are not',

"I'd"   : 'I had',

"n't"   :  'not',

 "'d"  :  'had',

"hv'v":   'have it'

}



def mapping_replacer(x, dic=contraction):

    for word in dic.keys():

        if " " + word + " " in x:

            x = x.replace(" " + word + " ", " " + dic[word] + " ")

        elif word in x:

            x=x.replace(word,dic[word])    

    return word_tokenize(x)

def mapping(tokenize_sent,dic=contraction):

    return list(itertools.chain(*[mapping_replacer(word) for word in tokenize_sent]))  # flatting array to shape(-1,) 

                                                                                       #Itertools.chain for flatting
# stemming

from nltk.stem import SnowballStemmer

def stemming(word):

    s=SnowballStemmer('english')

    return s.stem(word)

def stem_process(list_of_word):

    return [stemming(word) for word in list_of_word]
#stopword 

from nltk.corpus import stopwords 

def remove_stopword(list_of_word):

    stop_words = set(stopwords.words('english'))

    filtered_list_of_word = [w for w in list_of_word if w not in stop_words] 

    return filtered_list_of_word

    
# full preprocessing steps (use of all above functions)



def preprocessing_steps(batch_of_text):

    process = [remove_space(text) for text in batch_of_text]

    tokenize_sents = [tokenize(text) for text in process]

    mapped_sents = [mapping(tokenize_sent) for tokenize_sent in tokenize_sents]

    stopped = [remove_stopword(list_of_word) for list_of_word in mapped_sents]

    stemmed = [stem_process(list_of_word) for list_of_word in stopped]

    return stemmed

    

    

    
# vocab creation and giving word to unique id

def create_vocab(batch):

    vocab =['PADPAD','UNKUNK'] + list(set(itertools.chain(*batch)))

    n_tokens=len(vocab)

    word_to_id={}

    id_to_word={}

    for i,j in enumerate(vocab):

        word_to_id[j]=i

        id_to_word[i]=j

    return vocab,word_to_id,id_to_word,n_tokens
# Converting sentences to matrix representation 



def to_mat(batch,word_to_id):

    mat=[]

    for i in batch:

        mat1=[]

        for j in i:

            try:

                mat1.append(word_to_id[j])

            except:

                mat1.append(1)

        mat.append(mat1)

    return mat    
# Picking up 10000 rows of sample from training data for testing  

X=train.Phrase.values[:]

y=train.Sentiment.values[:]

from sklearn.model_selection import train_test_split

X_train,X_valid,y_train,y_valid=train_test_split(X,y,test_size=0.20,random_state=0,stratify=y)
print("X_train -----------------------------------------------------------")

print(X_train[:5])

print("X_valid -------------------------------------------------------------------------")

print(X_valid[:5])

print("y_train --------------------------------------------------------------------------")

print(y_train[:5])

print("y_valid -------------------------------------")

print(y_valid[:5])
# preprocessing steps



X_train=preprocessing_steps(X_train)

vocab,word_to_id,id_to_word,n_tokens = create_vocab(X_train)

X_train = to_mat(X_train,word_to_id)

X_valid = preprocessing_steps(X_valid)

X_valid = to_mat(X_valid,word_to_id)
print(n_tokens)

print("X_train -----------------------------------------------------------")

print(X_train[:5])

print("X_valid -------------------------------------------------------------------------")

print(X_valid[:5])

print("y_train --------------------------------------------------------------------------")

print(y_train[:5])

print("y_valid -------------------------------------")

print(y_valid[:5])
import tensorflow as tf

import keras 

from keras.models import Sequential

from keras.layers import Dense,LSTM,Embedding,Dropout

from keras.layers import Input
# post padding with 0 



X_train=tf.keras.preprocessing.sequence.pad_sequences(

    X_train, maxlen=10, dtype='int32', padding='post',

    value=0.0)

X_valid=tf.keras.preprocessing.sequence.pad_sequences(

    X_valid, maxlen=10, dtype='int32', padding='post',

    value=0.0)
print("X_train -----------------------------------------------------------")

print(X_train[:5])

print("X_valid -------------------------------------------------------------------------")

print(X_valid[:5])
# convert labels(sentiments range 1 - 5) to one hot 

y_train=np.array(tf.keras.backend.one_hot(y_train,num_classes=5))

y_valid=np.array(tf.keras.backend.one_hot(y_valid,num_classes=5))
print("y_train --------------------------------------------------------------------------")

print(y_train[:5])

print("y_valid -------------------------------------")

print(y_valid[:5])
# # Simple lstm model for testing



# keras.backend.clear_session()

# model1=Sequential()

# model1.add(Embedding(n_tokens,10))

# model1.add(LSTM(10,return_sequences=False))

# model1.add(Dense(5,activation='softmax'))

# model1.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# model1.summary()

# print(model1.input_shape)

# print(model1.output_shape)
# model1.fit(X_train,y_train,batch_size=32,validation_data=[X_valid,y_valid],epochs=5)
tf.keras.backend.clear_session()

inputL=tf.keras.Input(shape=(None,))

embeddingL=tf.keras.layers.Embedding(n_tokens,100)(inputL)

lstmL1=tf.keras.layers.LSTM(300,return_sequences=True,activation='relu')(embeddingL)

bn1=tf.keras.layers.BatchNormalization(axis = -1, name = 'bn1')(lstmL1)

dropout1=tf.keras.layers.Dropout(0.5)(bn1)

lstmL2=tf.keras.layers.LSTM(300,return_sequences=False,activation='relu')(dropout1)

bn2=tf.keras.layers.BatchNormalization(axis = -1, name = 'bn2')(lstmL2)

dropout2=tf.keras.layers.Dropout(0.5)(bn2)

denseL2=tf.keras.layers.Dense(100,activation='relu')(dropout2)

denseL1=tf.keras.layers.Dense(5,activation='sigmoid')(denseL2)

model=tf.keras.Model(inputs=inputL,outputs=denseL1)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

print(model.summary())

print(model.input_shape)

print(model.output_shape)
model.fit(X_train,y_train,batch_size=64,validation_data=[X_valid,y_valid],epochs=20)
sm=['worst at all','good movie','best quality having lot of fun','seems fine','worthless give it to 0']

# sm=['good but not too much' 'hello this is good']

sm = preprocessing_steps(sm)

sm = to_mat(sm,word_to_id)

                 

sm=tf.keras.preprocessing.sequence.pad_sequences(

    sm, maxlen=10, dtype='int32', padding='post',

    value=0.0)

print(sm)
# print(model1.predict(sm))

# print(np.argmax(model1.predict(sm),axis=1))
print(model.predict(sm))

print(np.argmax(model.predict(sm),axis=1))
tz=zipfile.ZipFile('/kaggle/input/movie-review-sentiment-analysis-kernels-only/test.tsv.zip')

test = pd.read_csv(tz.open('test.tsv'),delimiter='\t')
tx=test.Phrase.values

ptx=preprocessing_steps(tx)

mtx=to_mat(ptx,word_to_id)

pmtx=tf.keras.preprocessing.sequence.pad_sequences(

    mtx, maxlen=10, dtype='int32', padding='post',

    value=0.0)

Sentiment0=pd.DataFrame((np.argmax(model.predict(pmtx),axis=1)).astype('int32').reshape(-1,1),columns=['Sentiment'])

df=pd.concat([test,Sentiment0],axis=1)

df=df[['PhraseId','Sentiment']]

df.to_csv('submission.csv',index=False)
df.Sentiment.value_counts()
# # spelling correction

# def edits1(word):

#     letters='abcdefghijklmnopqrstuvwxyz'

#     splits =[(word[:i],word[i:]) for i in range(len(word)+1)]

#     deletes=[ L+R[1:]  for L,R in splits if R]

#     transposes = [L + R[1] +R[0] + R[2:] for L,R in splits if len(R)>1]

#     replaces = [L + c + R[1:] for L,R in splits if R for c in letters]

#     inserts = [L + c + R  for L,R in splits for c in letters]

#     return (set(deletes+transposes+replaces+inserts))

# def edits2(word):

#     return (e2 for e1 in edits1(word) for e2 in edits1(e1))