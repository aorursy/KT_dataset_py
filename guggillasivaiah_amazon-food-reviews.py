%matplotlib inline



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



# using the sqllite table to read data.

con=sqlite3.connect('../input/amazon-fine-food-reviews/database.sqlite')



#filtering only positive 



filtered_data=pd.read_sql_query("""

SELECT *

FROM Reviews

WHERE Score!=3

""",con)



#give reviews with score>3 a positive rating, and reviews with a svore<3



def partition(x):

    if x<3:

        return 'negative'

    return 'positive'



#changing reviews with score lessthan 3 to be positive and vice versa

actualScore=filtered_data['Score']

positiveNegative=actualScore.map(partition)

filtered_data['Score']=positiveNegative



filtered_data.shape #looking at the number of attributes and size of the data

filtered_data.head()
#eda

#deduplication

display= pd.read_sql_query("""

SELECT *

FROM Reviews

WHERE Score != 3 AND UserId="AR5J8UI46CURR"

ORDER BY ProductID

""", con)

display
#sorting data according to productid in ascending order



sorted_data=filtered_data.sort_values('ProductId',axis=0, ascending=True,inplace=False)

sorted_data.head()
#deduplication of entries

final=sorted_data.drop_duplicates(subset={"UserId", "ProfileName","Time","Text" },keep='first',inplace=False)

final.shape
final['Score'].value_counts()
#checking to see how mucn % of data still remains



(final['Id'].size*1.0/filtered_data['Id'].size*1.0)*100
#data mining

# bag of words



count_vect=CountVectorizer() #in scikit-learn

final_counts=count_vect.fit_transform(final['Text'].values)
type(final_counts)
final_counts.shape
#stop word removel, stemming,tokenization, lematization



# find the sentences containing HTML tags



import re  # regular expression

i=0;

for sent in final['Text'].values:

    if(len(re.findall('<.*?>',sent))):

        print(i)

        print(sent)

        break;

    i+=1;
import string

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer



stop=set(stopwords.words('english')) #set of stopwords

sno=nltk.stem.SnowballStemmer('english')#initialising the snowball stemmer



def cleanhtml(sentence): #function to clean the word of any html-tags

    cleanr=re.compile('<.*?>')

    cleantext=re.sub(cleanr,' ',sentence)

    return cleantext

def cleanpunc(sentence):

    cleaned=re.sub(r'[?|!|\'|"|#]',r'',sentence)

    cleaned=re.sub(r'[.|,|)|(|\|/]',r'',cleaned)

    return cleaned

print(stop)

print('***************************************************')

print(sno.stem('tasty'))
#Code for implementing step-by-step the checks mentioned in the pre-processing phase

# this code takes a while to run as it needs to run on 500k sentences.

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

final['CleanedText']=final_string #adding a column of CleanedText which displays the data after pre-processing of the review
final.head(3)



#store final table into an SQLLite table for furure.

conn=sqlite3.connect('final.sqlite')

c=conn.cursor()

conn.text_factory =str

final.to_sql('Reviews',conn, if_exists='replace')
# Bi-Grams ans n-Grams



freq_dist_positive=nltk.FreqDist(all_positive_words)

freq_dist_negative=nltk.FreqDist(all_negative_words)

print("most common positive words :",freq_dist_positive.most_common(20))

print("most common negative words :",freq_dist_positive.most_common(20))
# bi-grams, tri-grams and n-gram

#removing stop words like 'not' should be avoided be before building n-grams

 

count_vect = CountVectorizer(ngram_range=(1,2))

final_bigram_counts = count_vect.fit_transform(final['Text'].values)

final_bigram_counts.get_shape()
# TF-IDF



tf_idf_vect = TfidfVectorizer(ngram_range=(1,2))

final_tf_idf = tf_idf_vect.fit_transform(final['Text'].values)

final_tf_idf.get_shape()
features = tf_idf_vect.get_feature_names()

len(features)
features[100000:100020]