import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 
iris_df = pd.read_csv('../input/iris/Iris.csv')

print(iris_df.shape)

iris_df.head()
fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(26,4))

for i in range(1,len(iris_df.columns)):

    column = iris_df.columns[i]

    ax[i-1].hist(iris_df[column])

    ax[i-1].set_title(column)

    ax[i-1].set_xlabel(column)

    ax[i-1].set_ylabel("Frequency")



#fig.tight_layout()

fig.show()
fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(26,4))

for i in range(1,len(iris_df.columns)-1):

    column = iris_df.columns[i]

    sns.violinplot(iris_df['Species'], iris_df[column], ax=ax[i-1])

fig.show()
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV



feature_name = iris_df.columns[1:5]

X = iris_df[feature_name]

y = iris_df['Species']

parameter_search = [{

    'n_estimators': [1, 10, 25], 'max_depth': [1, 2, 3, 4, 5]

}]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = GridSearchCV(RandomForestClassifier(), parameter_search, cv=5, scoring='f1_micro')

clf.fit(X_train, y_train)
print(clf.best_score_)

print(clf.best_estimator_)
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint as sp_randint

param_dist = {"max_depth": [3, None],

              "max_features": sp_randint(1, 5),

              "min_samples_split": sp_randint(2, 11),

              "bootstrap": [True, False],

              "criterion": ["gini", "entropy"],

              "n_estimators": sp_randint(2,25)}



random_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_dist,

                                   n_iter=20, cv=5, iid=False)



random_search.fit(X_train, y_train)
print(random_search.best_score_)

print(random_search.best_estimator_)