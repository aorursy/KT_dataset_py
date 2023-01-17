csv = pd.read_csv('/kaggle/input/amazon-fine-food-reviews/Reviews.csv')
import warnings

warnings.filterwarnings("ignore")



import sqlite3

import pandas as pd

import numpy as np

import nltk

import string

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

from nltk.stem.porter import PorterStemmer



import re

import string

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer



from gensim.models import Word2Vec

from gensim.models import KeyedVectors

import pickle



from tqdm import tqdm

import os
con = sqlite3.connect('/kaggle/input/amazon-fine-food-reviews/database.sqlite')

filtered_data = pd.read_sql_query("""SELECT * FROM Reviews

                                     WHERE Score != 3

                                     LIMIT 5000""",

                                 con)
def partition(x):

    if x < 3:

        return 0

    return 1

# The partation func determines the review with Score > 3 as positive and Score < 3 as Negative
#changing reviews with score less than 3 to be positive and vice-versa

actualScore = filtered_data['Score']

positiveNegative = actualScore.map(partition)

filtered_data['Score'] = positiveNegative



print("Number of data points in our data", filtered_data.shape)

filtered_data.head()
display= pd.read_sql_query("""

                                SELECT *

                                FROM Reviews

                                WHERE Score != 3 AND UserId="AR5J8UI46CURR"

                                ORDER BY ProductID

                           """, con)

display.head()
#Sorting the data on ProductID

sorted_data=filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')

sorted_data
#Deduplication of entries

#The following command drops the row which has same values for subset data except first row.

final=sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)

final.shape
#Checking to see how much % of data still remains

(final['Id'].size*1.0)/(filtered_data['Id'].size*1.0)*100
display= pd.read_sql_query("""

                            SELECT *

                            FROM Reviews

                            WHERE Score != 3 AND Id=44737 OR Id=64422

                            ORDER BY ProductID

                            """,con)



display.head()
final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]
#Before starting the next phase of preprocessing lets see the number of entries left

print(final.shape)



#How many positive and negative reviews are present in our dataset?

final['Score'].value_counts()
final
# printing some random reviews

sent_0 = final['Text'].values[0]

print(sent_0)

print("="*50)



sent_1000 = final['Text'].values[1000]

print(sent_1000)

print("="*50)



sent_1500 = final['Text'].values[1500]

print(sent_1500)

print("="*50)



sent_4900 = final['Text'].values[4900]

print(sent_4900)

print("="*50)
import re

#h

import string

from nltk.corpus import stopwords 

from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer



#set of all stop words

stop = set(stopwords.words('english')) 

#initialising the snowball stemmer

sno = nltk.stem.SnowballStemmer('english')



def cleanhtml(sentence):

    cleanr = re.compile('<.*?>')

    cleantext = re.sub(cleanr, ' ', sentence)

    return cleantext



def clearnpunc(sentence):

    cleaned = re.sub(r'[?|!|\"|#]',r'',sentence)

    cleaned = re.sub(r'[.|,|)|(\|/]',r' ',cleaned)

    return cleaned



print(stop)

print('*'*50)

print(sno.stem('tasty'))
i = 0

str1 = ' '

final_string = []

all_positive_words = []

all_negative_words = []

s = ' '

for sent in final['Text'].values:

    filtered_sentences = []

    sent = cleanhtml(sent)

    for w in sent.split():

        for cleaned_words in clearnpunc(w).split():

            if((cleaned_words.isalpha()) & (len(cleaned_words)>2)):

                if(cleaned_words.lower() not in stop):

                    s = (sno.stem(cleaned_words.lower())).encode('utf8')

                    filtered_sentences.append(s)

                    if(final['Score'].values)[i] == 1:

                        all_positive_words.append(s)

                    if(final['Score'].values)[i] == 0:

                        all_negative_words.append(s)

                else:

                    continue

            else:

                continue

    str1 = b" ".join(filtered_sentences)

    final_string.append(str1)

    i+=1
final['CleanedText'] = final_string
final.head(3)



#store final table into an SQlLite table for future

conn = sqlite3.connect('final.sqlite')

c = conn.cursor()

conn.text_factory = str

final.to_sql('Reviews', conn,  schema=None, if_exists='replace')
final['CleanedText'].head()