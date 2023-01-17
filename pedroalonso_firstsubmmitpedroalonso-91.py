# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#1: Perform imports and load the dataset into a pandas DataFrame
import numpy as np
import pandas as pd
import re
import json
#df = pd.read_json('./Sarcasm_Headlines_Dataset.json', orient='records')
# df.head()
# cat $FILE | tr -d '\n' remove \n form the command line

with open('../input/Sarcasm_Headlines_Dataset.json', 'r') as f:
    jsonDict = f.readlines()
    
print(jsonDict[:5])
print(len(jsonDict))
# remove the trailing "\n" from each line, I did not do any real cleaning I wanted to see the pure accurasy, more in the few next cells
data = list(map(lambda x: x.rstrip(), jsonDict))
print(data[:5])

# each element of 'data' is an individual JSON object.
# i want to convert it into an *array* of JSON objects
# which, in and of itself, is one large JSON object
# basically... add square brackets to the beginning
# and end, and have all the individual business JSON objects
# separated by a comma
data_json_str = "[" + ",".join(data) + "]"

print(len(data_json_str))
print(data_json_str[:10])
# now, load it into pandas
data_df = pd.read_json(data_json_str)
print(data_df.head(10))
### Task #2: Check for missing values:
data_df.isnull().sum()
# Check for whitespace strings (it's OK if there aren't any!):
blanks = []  # start with an empty list

for i,link,headline, sarcastic in data_df.itertuples():  # iterate over the DataFrame
    if type(headline)==str:            # avoid NaN values
        if headline.isspace():         # test 'review' for whitespace
            blanks.append(i)     # add matching index numbers to the list
        
print(list(data_df.itertuples())[:5])
print(len(blanks))
print(data_df.shape)

### Task #3:  Remove NaN values:
data_df.dropna(inplace=True)
print(data_df.shape)
headlines = data_df['headline']
labels = data_df['is_sarcastic']

print(headlines[0], labels[0])

for line in headlines:
    print(line)
#Sanity check
data_df['is_sarcastic'].value_counts()

### Task #5: Split the data into train & test sets: use a validation at the end of phd to twiddle a little more
from sklearn.model_selection import train_test_split
#print(len(X))
X = data_df['headline']#.append(df['headline'])
#X.append(data_df['headline'])
y = data_df['is_sarcastic']#.append(df['is_sarcastic'])
print(X[:10], len(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.01, random_state=42)
print(X_test.shape)
### Task #6: Build a pipeline to vectorize the date, then train and fit a model
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

text_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LinearSVC())])
text_mlp_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', MLPClassifier(hidden_layer_sizes=(300,200), random_state=42, warm_start=True, solver='lbfgs'))]) 
# 84% default, 0.87% with (300,200) 0.85% (400,300, 100), 0.87% (400,200, 50), 0.85% (300,100, 50, 25), 0.84% hidden_layer_sizes=(400, 200, 100, 50), random_state=42, warm_start=True, 0.89% hidden_layer_sizes=(400,200, 50), random_state=42, warm_start=True, solver='lbfgs')
# 0.89 hidden_layer_sizes=(300,200), random_state=42, warm_start=True, solver='lbfgs'
text_rf_clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', RandomForestClassifier())]) 
text_clf.fit(X_train, y_train)
text_mlp_clf.fit(X_train, y_train)
text_rf_clf.fit(X_train, y_train)

### Task #7: Run predictions and analyze the results
# Form a prediction set
pred = text_clf.predict(X_test)
pred_mlp = text_mlp_clf.predict(X_test)
pred_rf = text_rf_clf.predict(X_test)
# Report the confusion matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, pred))
print(confusion_matrix(y_test, pred_mlp))
print(confusion_matrix(y_test, pred_rf))
# Print a classification report
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))
print(classification_report(y_test, pred_mlp))
print(classification_report(y_test, pred_rf))
##without the  full thing this get 0.91 and 0.84 and 0.80

