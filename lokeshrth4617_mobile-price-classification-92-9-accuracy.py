# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler

from sklearn import ensemble

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from sklearn import model_selection

from sklearn import decomposition

from sklearn import pipeline

# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/mobile-price-classification/train.csv')

data.head()
data.info()
fig = plt.figure(figsize = (12,10))

for index, col in enumerate(data):

    plt.subplot(7,4,index+1)

    sns.distplot(data.loc[:,col],kde =False)

fig.tight_layout(pad=1.0)
fig = plt.figure(figsize =(12,10))

corr = data.corr()

sns.heatmap(corr, mask = corr<0.8,annot = True)

plt.show()
X = data.iloc[:,:-1].values

X
y = data['price_range'].values

y
data['price_range'].value_counts()
sc = StandardScaler()

X1 = sc.fit_transform(X)

X1
classifier = ensemble.RandomForestClassifier()

param_dist = {

    "n_estimators" : np.arange(100,1500,200),

    "max_depth" : np.arange(1,20),

    "criterion" : ["gini","entropy"]

}



model = model_selection.RandomizedSearchCV(

        estimator = classifier,

        param_distributions=param_dist,

        n_iter =10,

        scoring = 'accuracy',

        n_jobs = 1,

        cv = 4

)



model.fit(X1,y)

print(model.best_score_)

print(model.best_estimator_.get_params())
from xgboost import XGBClassifier

xgb = XGBClassifier()

param_dist = {

    "n_estimators" : np.arange(100,1500,200),

    "max_depth" : np.arange(1,20),

}



model1 = model_selection.RandomizedSearchCV(

        estimator = xgb,

        param_distributions=param_dist,

        n_iter =10,

        scoring = 'accuracy',

        n_jobs = 1,

        cv = 4

)



model1.fit(X1,y)

print(model1.best_score_)

print(model1.best_estimator_.get_params())
from xgboost import XGBClassifier

xgb = XGBClassifier()

param_dist = {

    "n_estimators" : np.arange(100,1500,200),

    "max_depth" : np.arange(1,20),

}



model1 = model_selection.RandomizedSearchCV(

        estimator = xgb,

        param_distributions=param_dist,

        n_iter =10,

        scoring = 'precision_macro',

        n_jobs = 1,

        cv = 4

)



model1.fit(X1,y)

print(model1.best_score_)

print(model1.best_estimator_.get_params())