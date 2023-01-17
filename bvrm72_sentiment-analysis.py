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

#Importing the necessary libraries 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier

from sklearn.feature_extraction.text import CountVectorizer
import xgboost as xgb

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
#Importing the dataset into dataframe. There is no header in the dataset, so used header=None option 
df= pd.read_csv(r"/kaggle/input/romanurdudataset/Roman Urdu DataSet.csv",encoding="ISO-8859-1", header=None)
df.head()
#finding out how many different categories are in the target column [ column ID -1]
df[1].value_counts()
#There is one row with value Neative, converting it into Negative
df[1].replace("Neative","Negative",inplace=True)
df.shape
df.isnull().sum()
#There are 20222 rows of column 2 are null, so this column is a wrong entry in the data, dropping that column
df.drop([2],axis="columns",inplace=True)
df.isnull().sum()
df.dropna(inplace=True)
##Converting all the strings to lower case, so that it is easy to compare
df[0]=df[0].str.lower()
df[1]=df[1].str.lower()
#Seperating independent and dependent variables.
X=df[0]
y=df[1]
#Splitting the data and train and test data
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=10)

clf=Pipeline([
    ('vectorizer',CountVectorizer()),
    ('nb',MultinomialNB())
])
clf.fit(X_train,y_train)
clf.score(X_train,y_train)
y_predicted=clf.predict(X_test)
clf.score(y_test,y_predicted)
test_sentence = "Movie achi thi magar hero bura tha"
test_1=pd.Series(test_sentence)
clf.predict(test_1)
pd.crosstab(y_test,y_predicted)
import seaborn as sns

cm=metrics.confusion_matrix(y_test, y_predicted)
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
from sklearn.linear_model import SGDClassifier

clf=Pipeline([
    ('vectorizer',TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 10), max_df=0.5)),
    ('sgd',SGDClassifier(alpha=.0001,penalty="l2"))
])

clf.fit(X_train,y_train)
clf.score(X_train,y_train)
y_predicted=clf.predict(X_test)
clf.score(y_test,y_predicted)
test_sentence = "Movie achi thi magar hero bura tha"
test_1=pd.Series(test_sentence)
clf.predict(test_1)
pd.crosstab(y_test,y_predicted)
cm=metrics.confusion_matrix(y_test, y_predicted)
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
clf=Pipeline([
    ('vectorizer',TfidfVectorizer(sublinear_tf=True, ngram_range=(1,2), max_df=0.5)),
    ('nb',MultinomialNB())
])
clf.fit(X_train,y_train)
clf.score(X_train,y_train)

y_predicted=clf.predict(X_test)
clf.score(y_test,y_predicted)
test_sentence = "Movie achi thi magar hero bura tha"
test_1=pd.Series(test_sentence)
clf.predict(test_1)
pd.crosstab(y_test,y_predicted)
reports =metrics.classification_report(y_test, y_predicted)
reports