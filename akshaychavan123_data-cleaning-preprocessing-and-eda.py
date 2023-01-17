import numpy as np

import pandas as pd

import sqlite3

import pickle
conn=sqlite3.connect("../input/amazon-fine-food-reviews/database.sqlite")

data=pd.read_sql_query("""

SELECT *

from Reviews

WHERE Score!=3

""",conn)

conn.close()

data.head()
data.shape
def partition(x):

    if x>3:

        return "positive"

    return "negative"

data['Score']=data['Score'].map(partition)
data.head()
data["Time"].nunique()
data["Time"].unique()
data[data["Time"]==  1303862400]
data[(data['Time'] == 1303862400) & (data['ProfileName'] == 'R. Ellis "Bobby"')]
sorted_data =data.sort_values("ProductId")
final=sorted_data.drop_duplicates(subset={"UserId","ProfileName","Text","Time"})

final.shape
final[final['HelpfulnessNumerator'] > final['HelpfulnessDenominator']]
final=final[final.HelpfulnessNumerator<=final.HelpfulnessDenominator]

final.shape
final['Score'].value_counts()
final.shape
conn = sqlite3.connect('final.sqlite')

c=conn.cursor()

final.to_sql('Reviews', conn, if_exists='replace')

conn.close()
conn = sqlite3.connect('final.sqlite')

final = pd.read_sql_query("""SELECT * FROM Reviews""", conn)

conn.close()

final.head()
final.shape
import re
import nltk

from nltk.corpus import stopwords
def cleanhtml(sentence):

    '''This function removes all the html tags in the given sentence'''

    cleanr = re.compile('<.*?>')    ## find the index of the html tags

    cleantext = re.sub(cleanr, ' ', sentence)  ## Substitute <space> in place of any html tag

    return cleantext
def cleanpunc(sentence):

    '''This function cleans all the punctuation or special characters from a given sentence'''

    cleaned = re.sub(r'[?|@|!|^|%|\'|"|#]',r'',sentence)

    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)

    return  cleaned
stop = set(stopwords.words('english')) #set of stopwords
sno = nltk.stem.SnowballStemmer('english') #initialising the snowball stemmer
def preprocessing(series):

    '''The function takes a Pandas Series object containing text in all the cells

       And performs following Preprocessing steps on each cell:

       1. Clean text from html tags

       2. Clean text from punctuations and special characters

       3. Retain only non-numeric Latin characters with lenght > 2

       4. Remove stopwords from the sentence

       5. Apply stemming to all the words in the sentence

       

       Return values:

       1. final_string - List of cleaned sentences

       2. list_of_sent - List of lists which can be used as input to the W2V model'''

    

    i = 0

    str1=" "

    final_string = []    ## This list will contain cleaned sentences

    list_of_sent = []    ## This is a list of lists used as input to the W2V model at a later stage

    

    ## Creating below lists for future use

    all_positive_words=[] # store words from +ve reviews here

    all_negative_words=[] # store words from -ve reviews here

    

    

    for sent in series.values:

        ## 

        filtered_sent = []

        sent = cleanhtml(sent)    ## Clean the HTML tags

        sent = cleanpunc(sent)    ## Clean the punctuations and special characters

        ## Sentences are cleaned and words are handled individually

        for cleaned_words in sent.split():

            ## Only consider non-numeric words with length at least 3

            if((cleaned_words.isalpha()) and (len(cleaned_words) > 2)):

                ## Only consider words which are not stopwords and convert them to lowet case

                if(cleaned_words.lower() not in stop):

                    ## Apply snowball stemmer and add them to the filtered_sent list

                    s = (sno.stem(cleaned_words.lower()))#.encode('utf-8')

                    filtered_sent.append(s)    ## This contains all the cleaned words for a sentence

                    if (final['Score'].values)[i] == 'positive':

                        all_positive_words.append(s) #list of all words used to describe positive reviews

                    if(final['Score'].values)[i] == 'negative':

                        all_negative_words.append(s) #list of all words used to describe negative reviews

        ## Below list is a list of lists used as input to W2V model later

        list_of_sent.append(filtered_sent)

        ## Join back all the words belonging to the same sentence

        str1 = " ".join(filtered_sent)

        ## Finally add the cleaned sentence in the below list

        final_string.append(str1)

        #print(i)

        i += 1

    return final_string, list_of_sent
## This takes around 1 hour

final_string, list_of_sent = preprocessing(final['Text'])
final['CleanedText']=final_string #adding a column of CleanedText which displays the data after pre-processing of the review
final.head()
conn = sqlite3.connect('final.sqlite')

c=conn.cursor()

final.to_sql('Reviews', conn, if_exists='replace', index = False)

conn.close()
with open('list_of_sent_for_input_to_w2v.pkl', 'wb') as pickle_file:

    pickle.dump(list_of_sent, pickle_file)
list_of_sent
from sklearn.feature_extraction.text import CountVectorizer
bow_vec=CountVectorizer()

bow=bow_vec.fit_transform(final['CleanedText'].values)
bow.shape
with open("bad_of_word_model.pkl",'wb') as bow_model:

    pickle.dump(bow,bow_model)
final.shape
from sklearn.feature_extraction.text import TfidfVectorizer
tf_idf_vec=TfidfVectorizer()

tf_idf=tf_idf_vec.fit_transform(final['CleanedText'].values)
tf_idf.shape
type(tf_idf)
with open("TF_IDF_model.pkl","wb") as tf_idf_model:

    pickle.dump(tf_idf,tf_idf_model)