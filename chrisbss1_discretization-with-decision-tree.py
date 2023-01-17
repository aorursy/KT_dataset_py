from sklearn import datasets

import pandas as pd





# Convert the dataset to dataframe



iris = datasets.load_iris()

X = iris.data

y = iris.target

feature_names = iris.feature_names



df = pd.DataFrame(X, columns = feature_names)

df['target'] = y

df.head()
predictors = [i for i in feature_names if i not in ['target']]



X = df[predictors]

y = df[['target']]
from sklearn.tree import DecisionTreeClassifier

from sklearn import metrics

from sklearn.model_selection import cross_val_score

import numpy as np

from sklearn import preprocessing







a=2

b=10





values = []

X_discretized = pd.DataFrame()

label = preprocessing.LabelEncoder()





for feature in feature_names :

  for i in range(a,b+1):

    model = DecisionTreeClassifier(criterion='gini', max_features = None, max_leaf_nodes= i)

    model = cross_val_score(model, X[[feature]], y, cv=5, scoring='accuracy', n_jobs =-1)

    values.append(np.mean(model)) # Add of the score for each iteration

  model = DecisionTreeClassifier(criterion='gini', max_features = None, 

                                   max_leaf_nodes= values.index(max(values))) # Building of decision tree with the value which have the best accuracy

  model.fit(X[[feature]], y)

  X_discretized[feature] = label.fit_transform(model.predict_proba(X[[feature]])[:,1]) # Change the probability to label with LabelEncoder

X_discretized.sample(10)
for i in predictors :

  print(X_discretized[i].value_counts())