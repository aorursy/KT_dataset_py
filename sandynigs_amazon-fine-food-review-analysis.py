%matplotlib inline

import warnings

warnings.filterwarnings("ignore")



import sqlite3

import pandas as pd

import numpy as np

import nltk

import string

import matplotlib.pyplot as plt

import seaborn as sn

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.feature_extraction.text import TfidfVectorizer



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.metrics import confusion_matrix

from sklearn import metrics

from sklearn.metrics import roc_curve, auc

from nltk.stem.porter import PorterStemmer



import re

# Tutorial about Python regular expressions: https://pymotw.com/2/re/

import string

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer



from gensim.models import Word2Vec

from gensim.models import KeyedVectors

import pickle



from tqdm import tqdm

import os
#using the SQLite table to read data.

con = sqlite3.connect('../input/database.sqlite')



#Filetring reviews with only positive or negative reviews i.e not considering score=3

filtered_data = pd.read_sql_query("""

SELECT *

FROM Reviews

Where Score!=3

""", con)



#Replace the score with positive or negative (1,2 - negative and 4,5 - positive) 



def partition(x):

    if x<3:

        return 'negative'

    return 'positive'



actualScore = filtered_data['Score']

positiveNegative = actualScore.map(partition)

filtered_data['Score'] = positiveNegative
filtered_data.shape

filtered_data.head()
display = pd.read_sql_query("""

SELECT *

FROM Reviews

WHERE SCORE != 3 AND UserId = "AR5J8UI46CURR"

ORDER BY ProductId

""", con)

display
#Sorting data according to ProducId in dataframe

sorted_data = filtered_data.sort_values('ProductId', axis=0, ascending = True, inplace = False, kind='quicksort', na_position='last')
#Deduplication of entries

final = sorted_data.drop_duplicates(subset = {"UserId", "ProfileName", "Time", "Text"}, keep='first', inplace = False)

final.shape
#Checking to see how much % of data still remains

(final['Id'].size*1.0)/(filtered_data['Id'].size*1.0)*100
display= pd.read_sql_query("""

SELECT *

FROM Reviews

WHERE Score != 3 AND Id=44737 OR Id=64422

ORDER BY ProductID

""", con)



display.head()
final = final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]
#How many positive and negative reviews left?

final['Score'].value_counts()

#final.shape
#count_vec = CountVectorizer() #in scikit-learn

#final_counts = count_vec.fit_transform(final['Text'].values)
#type(final_counts)
#final_counts.shape
final['Text'].values[6]
#Find sentences having HTML tags

#i=0

#for sentence in final['Text'].values:

#    if(len(re.findall('<.*?>', sentence))):

#        print(i)

#        print(sentence)

#        break

#    i += 1
#Functions to clean HTML and Punctuation



import re #https://pymotw.com/2/re

import string

from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer



stop = set(stopwords.words('english')) #set of all stopwords 

sno = SnowballStemmer('english') #Initialize stemmer



def cleanhtml(sentence): #Function to clean html

    cleanr = re.compile('<.*?>')

    cleantext = re.sub(cleanr, ' ',sentence)

    return cleantext



def cleanpunctuation(sentence): #Function to clean all punctuation

    cleaned = re.sub(r'[?|!|\'|"|#]', r'',sentence)

    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ',cleaned)

    return cleaned



print(stop)

print(sno.stem('tasty'))
#Root word for tasty#Code for implementing step-by-step the checks mentioned in the pre-processing phase

#this code takes a while to run as it needs to run on 500k sentences.

if not os.path.isfile('finals.sqlite'):

    i=0

    str1=' '

    final_string=[]

    all_positive_words=[] # store words from +ve reviews here

    all_negative_words=[] # store words from -ve reviews here.

    s=''

    for sent in tqdm(final['Text'].values):

        filtered_sentence=[]

        #print(sent);

        sent=cleanhtml(sent) # remove HTMl tags

        for w in sent.split():

            for cleaned_words in cleanpunctuation(w).split():

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



    #############---- storing the data into .sqlite file ------########################

    final['CleanedText']=final_string #adding a column of CleanedText which displays the data after pre-processing of the review 

    final['CleanedText']=final['CleanedText'].str.decode("utf-8")

        # store final table into an SQlLite table for future.

    conn = sqlite3.connect('finals.sqlite')

    c=conn.cursor()

    conn.text_factory = str

    final.to_sql('Reviews', conn,  schema=None, if_exists='replace', \

                 index=True, index_label=None, chunksize=None, dtype=None)

    conn.close()

    

    

    with open('positive_words.pkl', 'wb') as f:

        pickle.dump(all_positive_words, f)

    with open('negitive_words.pkl', 'wb') as f:

        pickle.dump(all_negative_words, f)



print("cell exec")
#using the SQLite table to read data.

con = sqlite3.connect('finals.sqlite')



#Filetring reviews with only positive or negative reviews i.e not considering score=3

final = pd.read_sql_query("""

SELECT *

FROM Reviews

""", con)
final['Text'].iloc[6]
final['CleanedText'].iloc[6]
#freq_dist_positive = nltk.FreqDist(all_positive_words)

#freq_dist_negative = nltk.FreqDist(all_negative_words)
#print("Most Common Positive Words : ",freq_dist_positive.most_common(20))

#print("Most Common Negative Words : ",freq_dist_negative.most_common(20))
#bi-gram, tri-gram and n-gram



#removing stop words like "not" should be avoided before building n-grams

count_vect = CountVectorizer() #in scikit-learn

final_unigram_counts = count_vect.fit_transform(final['CleanedText'].values)

#print("the type of count vectorizer ",type(final_bigram_counts))

print("the shape text BOW vectorizer ",final_unigram_counts.get_shape())

#print("the number of unique words including both unigrams and bigrams ", final_bigram_counts.get_shape()[1])