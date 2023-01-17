# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt



import warnings as wrn

wrn.filterwarnings('ignore')



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/twitter-user-gender-classification/gender-classifier-DFE-791531.csv',encoding='latin1')
data = data.loc[:,["gender","description"]]

data.head()
data.dropna(inplace=True)
gnd =  [0 if each == "male" else 1 for each in data.gender]

data.gender = gnd
import nltk # Natural Language Tool Kit

import re # Regular Expression



lemma = nltk.WordNetLemmatizer() # Lemmatizer (nltk library)

pattern = "[^a-zA-Z]"

desc_list = []

for each in data.description:

    each = re.sub(pattern," ",each) # Cleaning

    each = each.lower() # Converting to lower

    each = nltk.word_tokenize(each) # Converting string to list

    each = [lemma.lemmatize(each) for each in each] # Lemmatizing

    each = " ".join(each) # Converting list to string

    desc_list.append(each) 
desc_list[:5]
from sklearn.feature_extraction.text import CountVectorizer



most_used = 5000 # Most used 5000 words in bios

cv = CountVectorizer(max_features=most_used,stop_words='english') 
sparce_matrix = cv.fit_transform(desc_list).toarray()

sparce_matrix
from sklearn.model_selection import train_test_split



x = sparce_matrix

y = data.gender.values



x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=1,test_size=0.1)
from sklearn.naive_bayes import GaussianNB

NBC = GaussianNB()

NBC.fit(x_train,y_train)

print(NBC.score(x_test,y_test))
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()

LR.fit(x_train,y_train)

print(LR.score(x_test,y_test))
from sklearn.ensemble import RandomForestClassifier



RFC = RandomForestClassifier(n_estimators=20,random_state=1)



RFC.fit(x_train,y_train)

print(RFC.score(x_test,y_test))