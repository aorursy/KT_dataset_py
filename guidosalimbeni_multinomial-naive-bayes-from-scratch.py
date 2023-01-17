import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn

import warnings

warnings.filterwarnings('ignore')
data = pd.read_json('/kaggle/input/news-category-dataset/News_Category_Dataset_v2.json', lines=True)

data.head()
data = data[["category", "headline"]]

data.category.value_counts().plot.bar(figsize = (20,10))
mapper = {}



for i,cat in enumerate(data["category"].unique()):

        mapper[cat] = i



data["category_target"] = data["category"].map(mapper)

data.head()
from sklearn.feature_extraction.text import CountVectorizer
#toy example

text=["My name is Paul my life is Jane! And we live our life together" , "My name is Guido my life is Victoria! And we live our life together"]

toy = CountVectorizer(stop_words = 'english')

# https://docs.python.org/2/library/re.html Token pattern explained token_pattern=r'\w+|\,',

toy.fit_transform(text)

print (toy.vocabulary_)

matrix = toy.transform(text)

print (matrix)

features = toy.get_feature_names()

df_res = pd.DataFrame(matrix.toarray(), columns=features)

df_res
vect = CountVectorizer(stop_words = 'english')

X_train_matrix = vect.fit_transform(data["headline"]) 

print (X_train_matrix.shape)
print ("shape of the matrix ", X_train_matrix.shape)

print ("one example" , data["headline"][1515])
column = vect.vocabulary_["hollywood"]

print (column)

vect.get_feature_names()[column]

y = data["category_target"]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train_matrix, y, test_size=0.3)

from sklearn.naive_bayes import MultinomialNB

clf=MultinomialNB()

clf.fit(X_train, y_train)

print (clf.score(X_train, y_train))

print (clf.score(X_test, y_test))

predicted_result=clf.predict(X_test)

from sklearn.metrics import classification_report

#print(classification_report(y_test,predicted_result))
pi = {}

All = data["category_target"].value_counts().sum()
for i, cat in enumerate (data["category_target"].value_counts(sort = False)):

    pi[i] = cat / All



print("Probability of each class:")

print("\n".join("{}: {}".format(k, v) for k, v in pi.items()))
vect = CountVectorizer(stop_words = 'english')

X_train_matrix = vect.fit_transform(data["headline"]) 
docIdx, wordIdx = X_train_matrix.nonzero()

count = X_train_matrix.data
classIdx = []



for idx in docIdx:

        

    classIdx.append(data["category_target"].iloc[idx])



    

    
df = pd.DataFrame()

df["docIdx"] = np.array(docIdx)

df["wordIdx"] = np.array(wordIdx)

df["count"] = np.array(count)

df["classIdx"] = np.array(classIdx)

df.info()
len(vect.vocabulary_)
#Alpha value for smoothing

a = 0.001

#Calculate probability of each word based on class

pb_ij = df.groupby(['classIdx','wordIdx'])

pb_j = df.groupby(['classIdx'])

Pr =  (pb_ij['count'].sum() + a) / (pb_j['count'].sum() + len(vect.vocabulary_))    

#Unstack series

Pr = Pr.unstack()



#Replace NaN or columns with 0 as word count with a/(count+|V|+1)

for c in range(0,41):

    Pr.loc[c,:] = Pr.loc[c,:].fillna(a/(pb_j['count'].sum()[c] + 16689))



#Convert to dictionary for greater speed

Pr_dict = Pr.to_dict()



Pr