import sqlite3

import os 

import nltk

import numpy as np

import scipy



con = sqlite3.connect("../input/database.sqlite")

cur = con.cursor()

sqlString = """ 

            SELECT complaint_id, date_received, consumer_complaint_narrative, company, timely_response

            FROM consumer_complaints

            WHERE product = "Mortgage" AND 

                            consumer_complaint_narrative != ""

            """

cur.execute(sqlString)

complaints = cur.fetchall()

con.close()
import random

random.seed(7040)

rand_complaint = random.randint(0, len(complaints))

print(rand_complaint)

print(len(complaints))

print(complaints[rand_complaint])
complaint_list = []

for i in range(len(complaints)):

    complaint_list.append(complaints[i][2])
stopwords = nltk.corpus.stopwords.words('english')

stopwords.extend(['wells', 'fargo, bank', 'america','chase', 'x','xx','xxx','xxxx','xxxxx',

                'mortgage', 'x/xx/xxxx', 'mortgage', '00'])

print(stopwords[0:11])
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(stop_words = stopwords)

dtm = vectorizer.fit_transform(complaint_list)

vocab = vectorizer.get_feature_names()

dtm = dtm.toarray()

vocab = np.array(vocab)
from sklearn.metrics.pairwise import cosine_similarity

 

sim = 0

for i in range(len(dtm)):

    if i == 2:

        pass

    else:

        s = cosine_similarity(dtm[2].reshape(1,-1),dtm[i].reshape(1,-1))[0][0]

        if s > sim:

            sim = s

            closest = i

print(sim, closest)
print("COMPLAINT 2: ",complaints[2])

print("COMPLAINT ", closest,": ", complaints[closest])
for i in range(len(vocab)):

    if dtm[2][i] != 0:

        print(vocab[i], dtm[2][i])
# Given item's value and list of items with values, 

# return an ordered list of 5 items from list closest to given item

def addItem(itemValue, itemIndex, lst):

    newList = lst + [(itemValue, itemIndex)]

    newList = sorted(newList)

    while len(newList) > 5:

        newList.pop(0)

    return newList
nearestNeighbors=[(0,0)]

for i in range(len(dtm)):

    if i == 2:

        continue

    value = cosine_similarity(dtm[2].reshape(1,-1),dtm[i].reshape(1,-1))[0][0]

    if value > nearestNeighbors[0][0]:

        nearestNeighbors = addItem(value, i, nearestNeighbors)
nearestNeighbors
for tpl in nearestNeighbors:

    print(complaints[tpl[1]])
companies = np.array([complaints[i][3] for i in range(len(complaints))])

companies_unique = sorted(set(companies))

print(len(companies_unique))
# Start with an empty array for each company

dtm_companies = np.zeros((len(companies_unique), len(vocab)))

# Now, for each company we'll store the sum of the frequency of each vocab

# word in the dtm_companies array

for i, company in enumerate(companies_unique):

    dtm_companies[i, :] = np.sum(dtm[companies == company, :], axis=0) 
dist = 1 - cosine_similarity(dtm_companies)
from scipy.cluster.hierarchy import ward, dendrogram

linkage_matrix = ward(dist)
from scipy.cluster.hierarchy import ward, dendrogram

%matplotlib inline

import matplotlib.pyplot as plt

linkage_matrix = ward(dist)

dendrogram(linkage_matrix)

plt.show()