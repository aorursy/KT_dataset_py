import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

data=pd.read_csv(r"../input/amazon-fine-food-reviews/Reviews.csv",encoding="latin1")


data=pd.concat([data.Score,data.Text],axis=1)

data = data.iloc[:30000]

data.dropna(axis=0,inplace=True)

data.head(10)
data.Score.value_counts()

import re

first_description=data.Text[0]

description=re.sub("[^a-zA-Z]"," ",first_description) 

description=description.lower()
print(data.Text[0])

print(description)
import nltk

#☻nltk.download("stopwords")      

#from nltk.corpus import stopwords

#description = [ word for word in description if not word in set(stopwords.words("english"))]

description= nltk.word_tokenize(description)
import nltk as nlp





lemma = nlp.WordNetLemmatizer()

description = [ lemma.lemmatize(word) for word in description] 



description =" ".join(description)
#%%

print(first_description)

print(description)
description_list = []

for description in data.Text:

    description = re.sub("[^a-zA-Z]"," ",description)

    description = description.lower()   

    description = nltk.word_tokenize(description)

    lemma = nlp.WordNetLemmatizer()

    description = [ lemma.lemmatize(word) for word in description]

    description = " ".join(description)

    description_list.append(description)
from sklearn.feature_extraction.text import CountVectorizer 

max_features = 15000



count_vectorizer = CountVectorizer(max_features=max_features,stop_words = "english")



sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()  # x



words=count_vectorizer.get_feature_names()

print("Most used words: ",words[50:100])
y = data.iloc[:,0].values   

x = sparce_matrix

# train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 42)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)
y_pred = nb.predict(x_test)

print("accuracy: ",nb.score(y_pred.reshape(-1,1),y_test))
from sklearn.linear_model import LogisticRegression

lr=LogisticRegression()

lr.fit(x_train,y_train)

print("lr accuracy: ",lr.score(x_test,y_test))