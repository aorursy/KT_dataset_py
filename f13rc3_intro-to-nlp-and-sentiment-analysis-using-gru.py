# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
# importing neccesary libraries

%matplotlib inline
import sqlite3
import nltk
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.text import Tokenizer


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix, roc_curve, auc,accuracy_score,f1_score
from sklearn import metrics
import spacy
nlp = spacy.load('en_core_web_lg')
from nltk.stem.porter import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense , Input , LSTM , Embedding, Dropout , Activation, GRU, Flatten
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model, Sequential
from keras.layers import Convolution1D
from keras import initializers, regularizers, constraints, optimizers, layers

import os
file_names = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        file_names.append(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# using the sqlite table to read data
con = sqlite3.connect([file for file in file_names if file.endswith('sqlite')][0])

# filtering only positive and negative reviews i.e
# not taking into consideration those reviews with score =3
filtered_data = pd.read_sql_query("select * from Reviews where Score !=3",con)

# give reviews with score>3 a positive rating, and <3 a negative
def partition(x):
    if x<3:
        return 'negative'
    return 'positive'
# changing reviews with score<3 to be positive and vice versa
actualScore=filtered_data['Score']
positiveNegative = actualScore.map(partition)
filtered_data['Score'] = positiveNegative
filtered_data.shape 
filtered_data.head()
# checking the if 
# display = pd.read_sql_query('select * from Reviews group by UserId having count(distinct ProductId)>1', con)
# display
filtered_data[filtered_data.duplicated(subset={'UserId', 'ProfileName', 'Time', 'Text'}, keep='first')]
# sorting data according to ProductId in ascending order
sorted_data = filtered_data.sort_values('ProductId', axis=0, ascending=True)
# deduplication of entries
final = sorted_data.drop_duplicates(subset={'UserId', 'ProfileName', 'Time', 'Text'}, keep='first',inplace=False)
final.shape
# checking to see how much % of data still remains
(final['Id'].size)/(filtered_data['Id'].size)*100
display = pd.read_sql_query('select * from Reviews where Score!=3 and HelpfulnessNumerator>HelpfulnessDenominator', con)
display
final = final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]
final
print(final.shape)

# how many positive and negative reviews are present in our dataset?
final['Score'].value_counts()
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

stop = set(stopwords.words('english')) # set of stopwords
sno = nltk.stem.SnowballStemmer('english') # initializing the snowball stemmer

# function to clean teh word in html tags
def cleanhtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr,' ',sentence)
    return cleantext
# function to clean the word of any punctuation
def cleanpunc(sentence):
    cleaned = re.sub(r'[?|!|\'|"|#]', r'', sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ',cleaned)
    return cleaned
print(stop)
print('***********************************************************************************************************')
print(sno.stem('tasty'))
# code for implementing step by step the checks mentioned in the preprocessing 
# this code takes a while to run as it needs to run on 500k sentences.
i=0
str1= ' '
final_string = []
all_positive_words = []
all_negative_words =[]
for sent in final['Text'].values:
    filtered_sentence = []
    sent = cleanhtml(sent)
    # remove html tags
    for w in sent.split():
        for cleaned_words in cleanpunc(w).split():
            if ((cleaned_words.isalpha()) & (len(cleaned_words)>2)):
                if (cleaned_words.lower() not in stop):
                    s = (sno.stem(cleaned_words.lower()))#.encode('utf-8')
                    filtered_sentence.append(s)
                    if (final['Score'].values)[i] == 'positive':
                        all_positive_words.append(s)
                    if (final['Score'].values)[i] == 'negative':
                        all_negative_words.append(s)
                else:
                    continue
            else:
                continue
    str1 = " ".join(filtered_sentence) # final string of cleaned words
    
    final_string.append(str1)
    i+=1
    
filtered_sentence
final['Cleaned_text'] = final_string # adding the new column after cleaning the text
final.head(3)

# store the final table into a SQLite table for future
conn = sqlite3.connect('final.sqlite')
c = conn.cursor()
conn.text_factory = str
final.to_sql('Reviews', conn, schema=None, if_exists='replace')
final['Score'] = final['Score'].apply(lambda x:1 if x=='positive' else 0)
train_df, test_df = train_test_split(final, test_size = 0.2, random_state = 42)
print("Training data size : ", train_df.shape)
print("Test data size : ", test_df.shape)
type(train_df['Cleaned_text'][0])
top_words = 5000
tokenizer = Tokenizer(num_words=top_words)
tokenizer.fit_on_texts(train_df['Cleaned_text'])
list_tokenized_train = tokenizer.texts_to_sequences(train_df['Cleaned_text'])

max_review_length = 200
X_train = pad_sequences(list_tokenized_train, maxlen=max_review_length)
y_train = train_df['Score']
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words+1, embedding_vecor_length, input_length=max_review_length))
model.add(GRU(100))
# model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


model.fit(X_train,y_train, nb_epoch=10, batch_size=64, validation_split=0.2)


list_tokenized_test = tokenizer.texts_to_sequences(test_df['Text'])
X_test = pad_sequences(list_tokenized_test, maxlen=max_review_length)
y_test = test_df['Score']
prediction = model.predict(X_test)
y_pred = (prediction > 0.5)
print("Accuracy of the model : ", accuracy_score(y_pred, y_test))
print('F1-score: ', f1_score(y_pred, y_test))
print('Confusion matrix:')
confusion_matrix(y_test,y_pred)

prediction