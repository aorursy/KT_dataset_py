# Importing Required Variables
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import sys,math, copy, time
import re
import csv
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np
from nltk.stem import WordNetLemmatizer
from scipy.spatial.distance import cosine
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Reading the Data
clothing_review = pd.read_csv("../input/Womens Clothing E-Commerce Reviews.csv")
clothing_review = clothing_review.dropna(subset=['Review Text'])
clothing_review = clothing_review[clothing_review['Clothing ID'] == 862]

#Getting Keywords
keyWords = [ "dress","pretty"]

# Clearing the data from extra characters
data = []
for i in clothing_review["Review Text"]:
    j = i.lower()
    j = re.sub(r'[^A-Za-z ]', '', j)
    data.append(j)
    
# Tokenising the data
tokenizer = RegexpTokenizer(r'\w+')
for i in range(len(data)) :
    data[i] = tokenizer.tokenize(data[i])

# Getting the list of stop words
stopWords = list(stopwords.words('english'))
stopWords = [re.sub(r'[^A-Za-z ]', '', j) for j in stopWords]

# Lemmatizing and removing stop words
wordnet_lemmatizer = WordNetLemmatizer()
dataFiltered = []
for each_review in data :
    temp = []
    for word in each_review : 
        if not word in stopWords :
            temp.append(wordnet_lemmatizer.lemmatize(word))
    dataFiltered.append(temp)


dataFiltered.append(keyWords)

# Creating the word list
wordList = np.array(dataFiltered)
wordList = np.hstack(wordList)
wordList = list(set(wordList))
wordList.sort()
number_of_reviews = len(dataFiltered)
wordListIndex = { wordList[i]: i for i in range(len(wordList))}
nDocsPerWord = {i : 0 for i in wordList}




tf = np.zeros(shape=(number_of_reviews,len(wordList)))
te = np.zeros(shape=(number_of_reviews,len(wordList)))

for i in range(len(dataFiltered)):
    this_doc_accounted = []
    for j in dataFiltered[i] :
        tf[i][wordListIndex[j]] += 1
        te[i][wordListIndex[j]] = 1
        if not j in this_doc_accounted :
            this_doc_accounted.append(j)
            nDocsPerWord[j] += 1
            

tfIdf = copy.deepcopy(tf)

for i in range(number_of_reviews) :
    for k in dataFiltered[i]:
        j = wordListIndex[k]
        if tfIdf[i][j] != 0 :
            tfIdf[i][j] = tfIdf[i][j]*math.log(number_of_reviews/nDocsPerWord[wordList[j]])

print(tfIdf.shape)

k = 20
sum1 = te.sum(axis=0)
print(sum1.shape)
to_del = []
for i in range(len(sum1)) :
    if sum1[i] < k :
        to_del.append(i)
te = np.delete(te, to_del, axis = 1)
print(te.shape)


sum1 = tf.sum(axis=0)
print(sum1.shape)
to_del = []
for i in range(len(sum1)) :
    if sum1[i] < k :
        to_del.append(i)
tf = np.delete(tf, to_del, axis = 1)
print(tf.shape)

sum1 = tfIdf.sum(axis=0)
print(sum1.shape)
to_del = []
for i in range(len(sum1)) :
    if sum1[i] < k :
        to_del.append(i)
tfIdf = np.delete(tfIdf, to_del, axis = 1)
print(tfIdf.shape)


with open("te.dat",'w') as writefile :
    for i in te :
        for j in i :
#             print(j,end="\t")
            writefile.write(str(j) + "\t")
        writefile.write("\n")
#         print()
with open("tf.dat",'w') as writefile :
    for i in tf :
        for j in i :
#             print(j,end="\t")
            writefile.write(str(j) + "\t")
        writefile.write("\n")
#         print()
with open("tfIdf.dat",'w') as writefile :
    for i in tfIdf :
        for j in i :
#             print(j,end="\t")
            writefile.write(str(j) + "\t")
        writefile.write("\n")
#         print()