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
import pandas as pd

import numpy as np

import matplotlib.pyplot
df = pd.read_csv('../input/Cancer1.csv')

df
df.head()
df.isnull().sum()
#Chcek the unique value.

df['Class'].unique()
df.drop(['Sample code number'],axis = 1, inplace = True)
df.head()
df.describe()
df.info()
df['Bare Nuclei']
df.replace('?',0, inplace=True)

df['Bare Nuclei']
from sklearn.preprocessing import Imputer

from sklearn.preprocessing import MinMaxScaler

# Convert the DataFrame object into NumPy array otherwise you will not be able to impute

values = df.values



# Now impute it

imputer = Imputer()

imputedData = imputer.fit_transform(values)
#Scaling the dataset.

scaler = MinMaxScaler(feature_range=(0, 1))

normalizedData = scaler.fit_transform(imputedData)
df.head()
# Bagged Decision Trees for Classification - necessary dependencies



from sklearn import model_selection

from sklearn.ensemble import BaggingClassifier

from sklearn.tree import DecisionTreeClassifier
# Segregate the features from the labels

X = normalizedData[:,0:9]

y = normalizedData[:,9]
print(X)
print(y)
kfold = model_selection.KFold(n_splits=10, random_state=7)

cart = DecisionTreeClassifier()

num_trees = 100

model = BaggingClassifier(base_estimator=cart, n_estimators=num_trees, random_state=7)

results = model_selection.cross_val_score(model, X, y, cv=kfold)

print(results.mean())
# AdaBoost Classification



from sklearn.ensemble import AdaBoostClassifier

seed = 7

num_trees = 70



kfold = model_selection.KFold(n_splits=10, random_state=seed)

model = AdaBoostClassifier(n_estimators=num_trees, random_state=seed)



results = model_selection.cross_val_score(model, X, y, cv=kfold)



print(results.mean())
# Voting Ensemble for Classification



from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.svm import SVC

from sklearn.ensemble import VotingClassifier



kfold = model_selection.KFold(n_splits=10, random_state=seed)



# create the sub models

estimators = []

model1 = LogisticRegression()

estimators.append(('logistic', model1))



model2 = DecisionTreeClassifier()

estimators.append(('cart', model2))



model3 = SVC()

estimators.append(('svm', model3))



# create the ensemble model

ensemble = VotingClassifier(estimators)

results = model_selection.cross_val_score(ensemble, X, y, cv=kfold)



print(results.mean())