import numpy as np
import pandas as pd
import tensorflow as tf 
df=pd.read_csv("../input/twitter-user-gender-classification/gender-classifier-DFE-791531.csv",encoding='latin1')
df.head()

import nltk
from nltk.corpus import stopwords
import random
import string
df.shape
df.columns
df["all_descriptions"] = df['description']

data= pd.concat([df.all_descriptions,df.gender],axis=1)
data.dropna(axis=0,inplace=True)
data.shape
data.gender=[1 if each =="female" else 0 for each in data.gender]
data.head(15)
data.columns
data.rename(columns={"0":"description","gender":"gender"})
import re

first_description=data.all_descriptions[4]
description=re.sub("[^a-zA-Z]"," ",first_description)
description=description.lower()
description
description_list=[]
import nltk
nltk.download('punkt')
import nltk as nlp
nltk.download('wordnet')
from nltk.corpus import stopwords
for description in data.all_descriptions:
    description=re.sub("[^a-zA-Z]"," ",description)
    description=description.lower()
    description=nltk.word_tokenize(description)
    #description=[word for word in description if not word in set(stopwords.words("english"))]
    lema= nlp.WordNetLemmatizer()
    description=[lema.lemmatize(word)for word in description]
    description=" ".join(description)
    description_list.append(description)
description_list
from sklearn.feature_extraction.text import CountVectorizer
max_features=5000
count_vectorizer=CountVectorizer(max_features=max_features,stop_words="english")

sparse_matrix=count_vectorizer.fit_transform(description_list).toarray()
data.head()
x=sparse_matrix[:10000]
y=data.iloc[:10000,0]
x.shape
y.shape
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.1,random_state=42)

from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier( criterion = 'entropy', random_state = 0)
dtree.fit(x_train, y_train)
y_pred = dtree.predict(x_test)
print("accuracy: ",rfc.score(x_test,y_test))


































