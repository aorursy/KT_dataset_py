# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
#reading the files
tn = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/True.csv")
fn = pd.read_csv("/kaggle/input/fake-and-real-news-dataset/Fake.csv")
tn.head()
fn.head()
#creating a column for true and false
tn['Verdict'] = 1
fn['Verdict'] = 0
#create a single dataset with 10000 rows of each which should be sufficient to make a good classifier.
df = pd.concat([tn.head(5000),fn.head(5000)], ignore_index = True)
#check the number of rows
len(df)
#creating a single column with text
df['Review'] = df['title'] + " " + df['text'] + " " + df['subject']
df.head()
#import the required packages
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
#create a list to store all the lines 
corpus = []
for i in range (0,10000):
    line = re.sub('[^a-zA-Z]', " ", df['Review'][i])
    line = line.lower()
    line = line.split()
    line = [ps.stem(word) for word in line if not word in set(stopwords.words('english'))]
    line = " ".join(line)
    corpus.append(line)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
#taking the verdict
y = df['Verdict'].values
len(corpus)
#split into train and test set 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)
#classify using Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
y_pred = classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#visualize the confusion matrix
import seaborn as sns
sns.heatmap(cm, annot = True)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)