import nltk 

nltk.download('stopwords')
import numpy as np                                  #for large and multi-dimensional arrays

import pandas as pd                                 #for data manipulation and analysis

import nltk                                         #Natural language processing tool-kit



from nltk.corpus import stopwords                   #Stopwords corpus

from nltk.stem import PorterStemmer                 # Stemmer



from sklearn.feature_extraction.text import CountVectorizer          #For Bag of words

from sklearn.feature_extraction.text import TfidfVectorizer          #For TF-IDF
import os

os.listdir('../input/nlp-topic-modelling')
data_path = "../input/nlp-topic-modelling/Reviews.csv"

data = pd.read_csv(data_path)

data_sel = data.head(10000)                                #Considering only top 10000 rows
data_sel.head()
# Shape of our data

data_sel.columns
data_sel.info()
# Write the code to remove all the rows from the dataset that have neutral review ie. Score value as 3





data_sel=data_sel[data_sel['Score']!=3]
data_sel.info()
# Write the code to replace the values of Score column with "positive" or "Negative" depending on the Score value





data_sel["Score"].replace({1: "Negative", 2: "Negative" , 4: "Positive" , 5:"Positive"}, inplace=True)

data_sel.head()
# Write the code to remove dulicates from the data and remove the rows where HelpfulnessNumerator is greater than 

# HelpfulnessDenominator. Store the resultant in a dataframe variable called "final"





data_sel.drop_duplicates(subset=['UserId', 'ProfileName','Text'],keep='first',inplace=True)



data_sel.drop(data_sel[data_sel['HelpfulnessNumerator'] > data_sel['HelpfulnessNumerator']].index, inplace=True)
final=data_sel

#final.head()

final.info()
final_X = final['Text']

final_y = final['Score']
print(final_X[1])

print(final_y[1])
stop = set(stopwords.words('english')) 

print(stop)
from nltk.tokenize import word_tokenize
# Solution



import re

temp =[]

snow = nltk.stem.SnowballStemmer('english')

for sentence in final_X:

    sentence = sentence.lower()                 # Converting to lowercase

    cleanr = re.compile('<.*?>')

    sentence = re.sub(cleanr, ' ', sentence)        #Removing HTML tags

    sentence = re.sub(r'[?|!|\'|"|#]',r'',sentence)

    sentence = re.sub(r'[.|,|)|(|\|/]',r' ',sentence)        #Removing Punctuations

    

    words = [snow.stem(word) for word in sentence.split() if word not in stopwords.words('english')]   # Stemming and removing stopwords

    temp.append(words)

    

final_X = temp    
print(final_X[1])
sent = []

for row in final_X:

    sequ = ''

    for word in row:

        sequ = sequ + ' ' + word

    sent.append(sequ)



final_X = sent

print(final_X[1])
# Here we use the CountVectorizer from sklearn to create bag of words

count_vect = CountVectorizer(max_features=5000)

bow_data = count_vect.fit_transform(final_X)
final_X[1]
print(bow_data[1])
final_B_X = final_X
count_vect = CountVectorizer(ngram_range=(1,2))

Bigram_data = count_vect.fit_transform(final_B_X)

print(Bigram_data[1])
final_tf = final_X

tf_idf = TfidfVectorizer(max_features=5000)

tf_data = tf_idf.fit_transform(final_tf)

print(tf_data[1])