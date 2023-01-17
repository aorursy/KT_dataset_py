import numpy as np

import pandas as pd

import sqlite3

import matplotlib.pyplot as plt
# using the SQLite Table to read data.

con = sqlite3.connect('../input/database.sqlite')

#con = sqlite3.connect('database.sqlite') 



#filtering only positive and negative reviews i.e. 

# not taking into consideration those reviews with Score=3

filtered_data = pd.read_sql_query("""SELECT * FROM Reviews WHERE Score != 3""", con) 



filtered_data.head(5)
# Give reviews with Score>3 a positive rating, and reviews with a score<3 a negative rating.

def partition(x):

    if x < 3:

        return 'negative'

    return 'positive'



#changing reviews with score less than 3 to be positive and vice-versa

actualScore = filtered_data['Score']

positiveNegative = actualScore.map(partition) 

filtered_data['Score'] = positiveNegative

sorted_data = filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')

filtered_data.head(5)
final = sorted_data.drop_duplicates(subset = {"UserId","ProfileName","Time","Text"}, keep ='first', inplace=False)

final.shape
final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]
final['Score'].value_counts()
import re

import string

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer

import nltk



stop = set(stopwords.words('english'))

sno = nltk.stem.SnowballStemmer('english')

#print(stop)

print(sno)
def cleanhtml(sentence): #function to clean the word of any html-tags

    cleanr = re.compile('<.*?>')

    cleantext = re.sub(cleanr, ' ', sentence)

    return cleantext



def cleanpunc(sentence): #function to clean the word of any punctuation or special characters

    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)

    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)

    return  cleaned
sno.stem('delicious')
i=0

str1=' '

final_string=[]

all_positive_words=[] # store words from +ve reviews here

all_negative_words=[] # store words from -ve reviews here.

s=''

for sent in final['Text'].values:

    filtered_sentence=[]

    #print(sent);

    sent=cleanhtml(sent) # remove HTMl tags

    for w in sent.split():

        for cleaned_words in cleanpunc(w).split():

            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):    

                if(cleaned_words.lower() not in stop):

                    s=(sno.stem(cleaned_words.lower())).encode('utf8')

                    filtered_sentence.append(s)

                    if (final['Score'].values)[i] == 'positive': 

                        all_positive_words.append(s) #list of all words used to describe positive reviews

                    if(final['Score'].values)[i] == 'negative':

                        all_negative_words.append(s) #list of all words used to describe negative reviews reviews

                else:

                    continue

            else:

                continue 

    #print(filtered_sentence)

    str1 = b" ".join(filtered_sentence) #final string of cleaned words

    #print("***********************************************************************")

    

    final_string.append(str1)

    i+=1

final['CleanedText']=final_string
conn = sqlite3.connect('final.sqlite')

c=conn.cursor()

conn.text_factory = str

final.to_sql('Reviews', conn, schema=None, if_exists='replace', index=True, index_label=None, chunksize=None, dtype=None)
import sqlite3

con = sqlite3.connect("final.sqlite")

cleaned_data = pd.read_sql_query("select * from Reviews", con)
Xtrain = cleaned_data.CleanedText[:10000]

Xcv = cleaned_data.CleanedText[10000:12000]

Xtest= cleaned_data.CleanedText[12000:14000]
def score(x):

    if x == 'negative':

        return 0

    return 1



cleaned_data['Score'] = cleaned_data.Score.map(score) 
Ytrain = cleaned_data.Score[:10000]

Ycv = cleaned_data.Score[10000:12000]

Ytest= cleaned_data.Score[12000:14000]
Xtrain.head()
# Bag of words

from sklearn.feature_extraction.text import CountVectorizer 



bow = CountVectorizer(max_features = 300)

X_train = bow.fit_transform(Xtrain)

X_cv= bow.transform(Xcv)

X_test = bow.transform(Xtest)
X_train.shape
from sklearn.preprocessing import StandardScaler

scaler1 = StandardScaler(with_mean=False)

scaler = scaler1.fit(X_train)

X_train= scaler.transform(X_train)

X_cv= scaler.transform(X_cv)

X_test = scaler.transform(X_test)
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()

clf = clf.fit(X_train,Ytrain)
Y_pred = clf.predict(X_test)
from sklearn.metrics import accuracy_score

acc = accuracy_score(Ytest, Y_pred)

print(acc)
Y_pred
from sklearn.metrics import f1_score

auc = f1_score(Ytest,Y_pred)

print(auc)
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression().fit(X_train, Ytrain)



y_Predict = clf.predict(X_test)

from sklearn.metrics import f1_score

auc = f1_score(Ytest,y_Predict)

print(auc)
