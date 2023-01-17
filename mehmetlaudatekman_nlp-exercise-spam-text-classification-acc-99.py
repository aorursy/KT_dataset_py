# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load





"""

Data Manipulating

"""

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



"""

NLTK (Natural Language Tool Kit) | RE (Regular Expressions) | Count Vectorizer (SKLearn)

"""

import nltk

import re

from sklearn.feature_extraction.text import CountVectorizer





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/sms-spam-collection-dataset/spam.csv',encoding="latin1")
data.head()
data.info()
data.isnull().sum()
x = data.drop("v1",axis=1)

y = data.v1
for ftr in x:  # Each iterable is a feature name

    if ftr != "v2":

        

        x.drop(ftr,axis=1,inplace=True)

        

x.info()
y = [1 if each == "ham" else 0 for each in y]

y[:5]
lemma = nltk.WordNetLemmatizer()
new_x = []

pattern = "[^a-zA-Z]"

for txt in x["v2"]:

    

    txt = re.sub(pattern," ",txt) #Cleaning

    txt = txt.lower() # Lowering

    txt = nltk.word_tokenize(txt) #Tokenizing

    txt = [lemma.lemmatize(each) for each in txt] # Lemmatizing

    txt = " ".join(txt) # Joining

    new_x.append(txt) # Appending 

    
new_x[:5]
CV = CountVectorizer(stop_words='english')

sparce_matrix = CV.fit_transform(new_x).toarray()
x = sparce_matrix
from sklearn.model_selection import train_test_split #Splitter

from sklearn.naive_bayes import GaussianNB # Naive Bayes

from sklearn.ensemble import RandomForestClassifier # Random Forest

from sklearn.tree import DecisionTreeClassifier # Decision Tree

from sklearn.linear_model import LogisticRegression # Logistic Regression

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1)

print("x_train len is ",len(x_train))

print("x_test len is",len(x_test))

print("y_train len is",len(y_train))

print("y_test len is",len(y_test))
gnb = GaussianNB()

gnb.fit(x_train,y_train)

gnb.score(x_test,y_test)
rfc = RandomForestClassifier(n_estimators=50,random_state=1)

rfc.fit(x_train,y_train)

rfc.score(x_test,y_test)
dtc = DecisionTreeClassifier(random_state=1)

dtc.fit(x_train,y_train)

dtc.score(x_test,y_test)
logreg = LogisticRegression(random_state=1)

logreg.fit(x_train,y_train)

logreg.score(x_test,y_test)