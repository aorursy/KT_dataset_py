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
df = pd.read_csv('../input/moviereviews.tsv',sep='\t')
#Check length

len(df)

df['review'][1]
#Remove null values

df.isnull().sum()

df.dropna(inplace=True)
#remove whitespace values

blanks = []

for i,lb,rv in df.itertuples():

    if rv.isspace():

        blanks.append(i)

df.drop(blanks,inplace=True)
from sklearn.model_selection import train_test_split
#Split into test and train

X = df['review']

y = df['label']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size =0.33,random_state=42)
from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC
#Pipeline essentially combines multiple methods, if f(x), g(x) are given to pipe, f(g(x)) is returned

text_clf = Pipeline([('tfidf',TfidfVectorizer()),('clf',LinearSVC())])
#Fitting the model

text_clf.fit(X_train,y_train)
predictions  = text_clf.predict(X_test)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))
print(accuracy_score(y_test,predictions))