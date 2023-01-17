# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

 

from matplotlib import pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/mushrooms.csv')
data.head()
data['class'].value_counts()
data.shape
data.describe()
from sklearn.preprocessing import LabelEncoder

labelencoder=LabelEncoder()

for col in data.columns:

    data[col] = labelencoder.fit_transform(data[col])

 

data.head()
data.info()
plt.figure(figsize=(20,10))

sns.heatmap(data.corr())
def drop_1(df):

    df = df.drop('veil-type',axis = 1)

    return df



X = data.iloc[:,1:23]

y = data.iloc[:,0]
from sklearn.ensemble import RandomForestClassifier as RF

from sklearn.metrics import accuracy_score as AUC

from sklearn.model_selection import train_test_split, GridSearchCV
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state = 17, test_size = 0.3)
hyperparameters = {

    "n_estimators": [100,150,200,250],

    "max_features": ["auto", "sqrt", 0.33],

    "min_samples_leaf": [1, 3, 5, 7, 9, 11, 13, 15],

    'max_depth': [5,10,15,20]

}



clf = GridSearchCV(

    estimator=RF(n_jobs = -1),

    param_grid=hyperparameters,n_jobs =-1

)

clf.fit(X_train,y_train)
cv_results = pd.DataFrame(clf.cv_results_)



cv_results.columns
clf.best_params_
from sklearn.metrics import r2_score as r2,mean_squared_error as mse
clf1 = RF(max_depth=10, max_features='auto', n_estimators=100,min_samples_leaf=1)



clf1.fit(X_train, y_train)



y_pred1 = clf1.predict(X_test)



r2(y_test, y_pred1), r2(y_train,clf1.predict(X_train))