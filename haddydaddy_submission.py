

import numpy as np

import graphviz

import pandas as pd



from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from sklearn import tree

from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics 

                

import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
df = pd.read_csv('train.csv',sep=',')

df

test = pd.read_csv('test_2.csv',sep=',')

X_train = df['description']

y_train = df['class']

X_test = test['description']
count_vect = CountVectorizer(token_pattern= '((?:([@#]|[0-9]|[a-z]|[A-Z])+))', analyzer= 'word', min_df=2)
X_word_vect = count_vect.fit_transform(X_train)
print(X_word_vect.shape) 
clf = tree.DecisionTreeClassifier(criterion='entropy', random_state = 100, max_depth=30, \

                                  min_samples_leaf =2)
clf.fit(X_word_vect, y_train)
X_test_word_vect = count_vect.transform(X_test)
predicted_y = clf.predict(X_test_word_vect)

print(predicted_y)
test["class"] = predicted_y

test["class"] = test["class"].map(lambda x: "True" if x==1 else "False")

result = test[["ID","class"]]

result.to_csv("result1.csv", index=False)

result.head()