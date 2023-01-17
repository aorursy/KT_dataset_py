# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# import twitter data
data = pd.read_csv("../input/gender-classifier-DFE-791531.csv",encoding="latin1")
data = pd.concat([data.gender,data.description],axis=1)

#let's drop NaN values
data.dropna(axis=0,inplace=True)
data.head()
data.shape
data.gender = [1 if each == "female" else 0 for each in data.gender]
data.head(10)
data.description[4]
# regular expression RE =>> "[^a-zA-Z]"
import re

first_description = data.description[4]
description = re.sub("[^a-zA-Z]"," ",first_description)
description = description.lower() #Year year are different words
description
import nltk
from nltk.corpus import stopwords
#remove irrelavent words for e.g. and,the ...

#description = description.split()
description = nltk.word_tokenize(description)
#if we use word_tokenize instead of split it will be better
#split() = shouldn't => shouldn't
#word_tokenize() = shouldn't => shouldn't and n't separate as two word
description = [word for word in description if not word in set(stopwords.words("english"))]
description
#Lemmatazation = loved => love
import nltk as nlp

lemma = nlp.WordNetLemmatizer()
description = [lemma.lemmatize(word) for word in description]
description
description = " ".join(description)
description
description_list = []
for description in data.description:
    description = re.sub("[^a-zA-Z]"," ",description)
    description = description.lower()
    description = nltk.word_tokenize(description)
    #description = [ word for word in description if not word in set(stopwords.words("english"))]
    lemma = nlp.WordNetLemmatizer()
    description = [lemma.lemmatize(word) for word in description]
    description = " ".join(description)
    description_list.append(description)
    
#description_list
from sklearn.feature_extraction.text import CountVectorizer
#we can define max_features 
max_features = 1000
count_vectorizer = CountVectorizer(max_features=max_features,stop_words = "english")
#count_vectorizer = CountVectorizer(stop_words = "english")

sparce_matrix = count_vectorizer.fit_transform(description_list).toarray() # x

print("{} most common words: {}".format(max_features,count_vectorizer.get_feature_names()))
y = data.iloc[:,0].values   # male or female classes
x = sparce_matrix
# train test split
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.1, random_state = 42)
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

print("accuracy: ",nb.score(x_test,y_test))

y_pred = nb.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix

cm_nb = confusion_matrix(y_true,y_pred)

sns.heatmap(cm_nb,annot=True,cmap="RdPu",fmt=".0f",cbar=False)
plt.show()
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators = 100)

rf.fit(x_train,y_train)

print("accuracy: ",rf.score(x_test,y_test))
y_pred = rf.predict(x_test)
y_true = y_test

from sklearn.metrics import confusion_matrix

cm_rf = confusion_matrix(y_true,y_pred)

sns.heatmap(cm_rf,annot=True,cmap="RdPu",fmt=".0f",cbar=False)
plt.show()