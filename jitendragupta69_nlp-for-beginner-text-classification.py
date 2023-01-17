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
# Read Movie review Dataset

data = pd.read_csv('/kaggle/input/moviereviews.tsv',sep='\t')
#Check the top 5 rows to see how datat looks like 

data.head()
#Check total number of rows in dataset

len(data)
# Now check for missing data

# Review column is string, there might be changes few reviews could be space 

data.isnull().sum()
# There are 35 reviews are missing from review column and we can remove those missing records

data.dropna(inplace=True) # for permanent drop inplace =True

data.isnull().sum()
print(data.itertuples)
# Check for blank reviews in the dataset

empty = []

for index,label,review in data.itertuples():

    if review.isspace():

        empty.append(index)

print(empty)
# Drop blank reviews from the dataset

data.drop(empty,inplace=True)
data.count()
# Assign independent and dependent variables

X = data['review']

y = data['label']
# Split dataset into training and testing dataset

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Create a pipeline for tf-idf vectorizer and svm model

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.svm import LinearSVC

classification_text = Pipeline([('tfidf',TfidfVectorizer()),

                                ('clasifier',LinearSVC())])
# Train our model

classification_text.fit(X_train,y_train)
# Predict the result

y_pred = classification_text.predict(X_test)
print(y_pred)
# Check the accuracy, confusion matrik and classifiction report

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))