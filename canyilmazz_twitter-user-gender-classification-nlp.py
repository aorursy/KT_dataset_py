# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/twitter-user-gender-classification/gender-classifier-DFE-791531.csv",encoding="latin1")
data.columns
#we use description and gender
data=pd.concat([data.gender,data.description],axis=1)
data.head()
data.dropna(inplace=True,axis=0)
data.gender=[1 if i=="female" else 0 for i in data.gender]
data
import re
import nltk
import nltk as nlp
nltk.download("stopwords")
from nltk.corpus import stopwords
description_list=[]
for description in data.description:
    description=re.sub("[^a-zA-Z]"," ",description)
    description=description.lower()
    description=nltk.word_tokenize(description)
    #description=[word for word in description if not word in set(stopwords.words("english"))]
    lemma=nlp.WordNetLemmatizer()
    description=[lemma.lemmatize(word) for word in description]
    description=" ".join(description)
    description_list.append(description)
from sklearn.feature_extraction.text import CountVectorizer 
max_features=5000
cv=CountVectorizer(max_features=max_features,stop_words="english")
space_matrix=cv.fit_transform(description_list).toarray()
print("{} most frequently used words:{}".format(max_features,cv.get_feature_names()))
y=data.iloc[:,0].values #male or female classes
x=space_matrix
#train test split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=42)
#Naive Bayes classification
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
model=nb.fit(x_train,y_train)
y_head=model.predict(x_test)
print("accuracy:",model.score(x_test,y_test))
y_true=y_test
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_true,y_head)

import seaborn as sns
sns.heatmap(cm,annot=True)