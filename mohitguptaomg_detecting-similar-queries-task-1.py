import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt



import math

import collections

from collections import Counter



from nltk import word_tokenize

from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

import re
data = pd.read_csv('../input/train.csv')
data.dropna()

print(data.head(15))
data.describe()
dup_check = data['is_duplicate'].value_counts()

plt.bar(dup_check.index, dup_check.values)

plt.ylabel('Number of Queries')

plt.xlabel('Is Duplicate')

plt.title('Data Distribution', fontsize = 18)

plt.show()
print("Above Graph Features :  [Is Not Duplicate | Is Duplicate]\n")

print("Above Graph Indices  : ", dup_check.index)

print("\nAbove Graph Values   : ", dup_check.values)
print("Above Graph %age Values :")

print( dup_check / dup_check.sum())
questions = data[['id', 'question1', 'question2', 'is_duplicate']]

word_count = []

for row in questions.itertuples():

    q1 = len(str(row[2]).split())

    q2 = len(str(row[3]).split())

    word_count.append(q1 + q2)  
len(word_count)
word_count = pd.DataFrame(data = word_count, columns = ['word_count'])
count = word_count['word_count'].value_counts()

plt.figure(figsize=(12,6))

plt.bar(count.index, count.values)

plt.ylabel('Number of occurrence of Queries with x words')

plt.xlabel('Number of Words')

plt.title('Word Distribution in Queries', fontsize = 18)

plt.xlim(0, 100)

plt.show()
ps = PorterStemmer()



def tokenize(text):

    text = re.sub('[^a-zA-Z]+', ' ', text)

    text = text.lower()

    text = text.split()

    text = [ps.stem(word) for word in text if not word in set(stopwords.words('english'))]

    text = ' '.join(text)

    words = word_tokenize(text)

    return Counter(words)
def get_cosine(vec1, vec2):

    intersection = set(vec1.keys()) & set(vec2.keys())

    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    

    sum1 = sum([vec1[x]**2 for x in vec1.keys()])

    sum2 = sum([vec2[x]**2 for x in vec2.keys()])

    denominator = math.sqrt(sum1) * math.sqrt(sum2)



    if not denominator:

        return 0.0

    else:

        return float(numerator) / float(denominator)
def DistJaccard(str1, str2):

    str1 = set(str1.split())

    str2 = set(str2.split())

    return float(len(str1 & str2)) / len(str1 | str2)    

queries = data[['id', 'question1', 'question2', 'is_duplicate']]

cosine_list = []

jaccard_list = []

i = 1

for row in questions.itertuples():

    text1 = str(row[2])

    text2 = str(row[3])

    vector1 = tokenize(text1)

    vector2 = tokenize(text2) 

    cosine = get_cosine(vector1, vector2)

    jaccard = DistJaccard(text1, text2)

    cosine_list.append(cosine)    

    jaccard_list.append(jaccard)

        

    i+=1

    if (i == 1000):

        break    
c = data[:999]
c.head()
c.insert(6, "Cosine Score", cosine_list) 

c.insert(7, "Jaccard Score", jaccard_list) 
c.head(25)