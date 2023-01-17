# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python



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







# using the SQLite Table to read data.

con = sqlite3.connect('../input/database.sqlite') 







#filtering only positive and negative reviews i.e. 

# not taking into consideration those reviews with Score=3

filtered_data = pd.read_sql_query("""

SELECT *

FROM Reviews

WHERE Score != 3

""", con) 









# Give reviews with Score>3 a positive rating, and reviews with a score<3 a negative rating.

def partition(x):

    if x < 3:

        return 'negative'

    return 'positive'



#changing reviews with score less than 3 to be positive and vice-versa

actualScore = filtered_data['Score']

positiveNegative = actualScore.map(partition) 

filtered_data['Score'] = positiveNegative
filtered_data.shape #looking at the number of attributes and size of the data

filtered_data.head()
display= pd.read_sql_query("""

SELECT *

FROM Reviews

WHERE Score != 3 AND UserId="AR5J8UI46CURR"

ORDER BY ProductID

""", con)

display
#Sorting data according to ProductId in ascending order

sorted_data=filtered_data.sort_values('ProductId', axis=0, ascending=True, inplace=False, kind='quicksort', na_position='last')
#Deduplication of entries

final=sorted_data.drop_duplicates(subset={"UserId","ProfileName","Time","Text"}, keep='first', inplace=False)

final.shape
#Checking to see how much % of data still remains

(final['Id'].size*1.0)/(filtered_data['Id'].size*1.0)*100
#Observation:- It was also seen that in two rows given below the value of HelpfulnessNumerator is greater than HelpfulnessDenominator which is not practically possible hence these two rows too are removed from calcualtions



display= pd.read_sql_query("""

SELECT *

FROM Reviews

WHERE Score != 3 AND Id=44737 OR Id=64422

ORDER BY ProductID

""", con)

display
final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]

#Before starting the next phase of preprocessing lets see the number of entries left

print(final.shape)



#How many positive and negative reviews are present in our dataset?

final['Score'].value_counts()
import re

# Tutorial about Python regular expressions: https://pymotw.com/2/re/

import string

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

from nltk.stem.wordnet import WordNetLemmatizer



stop = set(stopwords.words('english')) #set of stopwords

sno = nltk.stem.SnowballStemmer('english') #initialising the snowball stemmer



def cleanhtml(sentence): #function to clean the word of any html-tags

    cleanr = re.compile('<.*?>')

    cleantext = re.sub(cleanr, ' ', sentence)

    return cleantext

def cleanpunc(sentence): #function to clean the word of any punctuation or special characters

    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)

    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)

    return  cleaned

print(stop)

print('************************************')

print(sno.stem('tasty'))



# find sentences containing HTML tags

i=0;

for sent in final['Text'].values:

    if (len(re.findall('<.*?>', sent))):

        print(i)

        print(sent)

        break;

    i += 1;    

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
inal.head(3) #below the processed review can be seen in the CleanedText Column 





# store final table into an SQlLite table for future.

conn = sqlite3.connect('final.sqlite')

c=conn.cursor()

conn.text_factory = str

final.to_sql('Reviews', conn, flavor=None, schema=None, if_exists='replace', index=True, index_label=None, chunksize=None, dtype=None)
#BoW

count_vect = CountVectorizer() #in scikit-learn

final_counts = count_vect.fit_transform(final['Text'].values)
type(final_counts)
final_counts.get_shape()
freq_dist_positive=nltk.FreqDist(all_positive_words)

freq_dist_negative=nltk.FreqDist(all_negative_words)

print("Most Common Positive Words : ",freq_dist_positive.most_common(20))

print("Most Common Negative Words : ",freq_dist_negative.most_common(20))
#bi-gram, tri-gram and n-gram



#removing stop words like "not" should be avoided before building n-grams

count_vect = CountVectorizer(ngram_range=(1,2) ) #in scikit-learn

final_bigram_counts = count_vect.fit_transform(final['Text'].values)
final_bigram_counts.get_shape()
tf_idf_vect = TfidfVectorizer(ngram_range=(1,2))

final_tf_idf = tf_idf_vect.fit_transform(final['Text'].values)
final_tf_idf.get_shape()
features = tf_idf_vect.get_feature_names()

len(features)
features[100000:100010]
print(final_tf_idf[3,:].toarray()[0]) 
# source: https://buhrmann.github.io/tfidf-analysis.html

def top_tfidf_feats(row, features, top_n=25):

    ''' Get top n tfidf values in row and return them with their corresponding feature names.'''

    topn_ids = np.argsort(row)[::-1][:top_n]

    top_feats = [(features[i], row[i]) for i in topn_ids]

    df = pd.DataFrame(top_feats)

    df.columns = ['feature', 'tfidf']

    return df



top_tfidf = top_tfidf_feats(final_tf_idf[1,:].toarray()[0],features,25)