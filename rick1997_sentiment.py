# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#-------Data visualisation imports
import seaborn as sns
#-------To split data into Training and Test Data
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
#-------For Natural Language Processing data cleaning
from sklearn.feature_extraction.text import TfidfTransformer
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import re
#TQDM is a progress bar library with good support for nested loops and Jupyter/IPython notebooks.
from tqdm import tqdm
#CountVectorizer converts collection of text docs to a matrix of token counts.
from sklearn.feature_extraction.text import CountVectorizer
import string
from nltk.corpus import stopwords
import nltk
from nltk.tokenize import word_tokenize
#-------Data model
from keras.utils import to_categorical
from tensorflow import set_random_seed
from sklearn.model_selection import train_test_split
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense,Dropout,Embedding,LSTM
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.models import Sequential
import keras.backend as K

train=pd.read_csv('/kaggle/input/sentiment-analysis1/train.tsv',sep='\t')
test= pd.read_csv('/kaggle/input/sentiment-analysis1/test.tsv', sep='\t')                       
sns.countplot(data=train,x='Sentiment')
train['Length'] = train['Phrase'].apply(lambda x: len(str(x).split(' ')))
train['Length'].unique()
train[train['Phrase'].str.len() == 0].head()

def clean_sentences(df):
    reviews=[]
    for sent in tqdm(df['Phrase']):
        review_text = BeautifulSoup(sent).get_text()
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        review_text=review_text.translate(str.maketrans('', '', string.punctuation))
        words = word_tokenize(review_text.lower())
        reviews.append(words)
    return reviews
train_sentences = clean_sentences(train)
test_sentences = clean_sentences(test)
print(len(train_sentences))
print(len(test_sentences))
target=train.Sentiment.values
y_target=to_categorical(target)
num_classes=y_target.shape[1]
X_train,X_val,y_train,y_val=train_test_split(train_sentences,y_target,test_size=0.2,stratify=y_target)
unique_words = set()
len_max = 0

for sent in tqdm(X_train):
    
    unique_words.update(sent)
    
    if(len_max<len(sent)):
        len_max = len(sent)
        
#length of the list of unique_words gives the no of unique words
print(len(list(unique_words)))
print(len_max)

tokenizer = Tokenizer(num_words=len(list(unique_words)))
tokenizer.fit_on_texts(list(X_train))

#texts_to_sequences(texts)

    # Arguments- texts: list of texts to turn to sequences.
    #Return: list of sequences (one per text input).
X_train = tokenizer.texts_to_sequences(X_train)
X_val = tokenizer.texts_to_sequences(X_val)
X_test = tokenizer.texts_to_sequences(test_sentences)

#padding done to equalize the lengths of all input reviews. LSTM networks needs all inputs to be same length.
#Therefore reviews lesser than max length will be made equal using extra zeros at end. This is padding.

X_train = sequence.pad_sequences(X_train, maxlen=len_max)
X_val = sequence.pad_sequences(X_val, maxlen=len_max)
X_test = sequence.pad_sequences(X_test, maxlen=len_max)

print(X_train.shape,X_val.shape,X_test.shape)
model=Sequential()
model.add(Embedding(len(list(unique_words)),300,input_length=len_max))
model.add(LSTM(128,dropout=0.5, recurrent_dropout=0.5,return_sequences=True))
model.add(LSTM(64,dropout=0.5, recurrent_dropout=0.5,return_sequences=False))
model.add(Dense(100,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.010),metrics=['accuracy'])
model.summary()
history=model.fit(X_train, y_train, validation_data=(X_val, y_val),epochs=6, batch_size=256, verbose=1)
test_prediction=model.predict(X_test,verbose=0)
yhat_classes = model.predict_classes(X_test, verbose=0)
test.head()
final_answer=pd.DataFrame({'PharaseId':test["PhraseId"].values,'Pharse':test["Phrase"],'predicted_category':yhat_classes})
final_answer["predicted_category"]=final_answer["predicted_category"].map({0:"negative",1:"somewhat negative",
                                                                           2:"neutral",3:"somewhat positive",
                                                                          4:"positive"})
final_answer.head()
filename = 'Movie Review Analysis(Sentiment).csv'
final_answer.to_csv(filename,index=False)
print('Saved file: ' + filename)
