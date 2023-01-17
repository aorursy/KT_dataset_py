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

Churn_Modelling = pd.read_csv("../input/churn-modelling/Churn_Modelling.csv")

Churn_Modelling.head()
Churn_Modelling.shape
data = Churn_Modelling.copy()

corr = data.corr()



import seaborn as sns

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
# variable selection 

X = data.iloc[:,3:13] #drop - rownumber, customerId, Surname

Y = data.iloc[:,13]
#select categorical features

for i in X:

    if X[i].dtypes == object:

        print(i)
Geography_df = pd.get_dummies(X['Geography'], drop_first=True)

Geography_df.head()
Gender_df = pd.get_dummies(X['Gender'], drop_first=True)

Gender_df.head()
X = X.drop(['Geography','Gender'], axis=1)

X.columns
X = pd.concat([X,Geography_df,Gender_df], axis = 1)

X.head()
#parameters

params = {

    "learning_rate"    :[0.05,0.10,0.15,0.20,0.25,0.30],

    "max_depth"        :[ 3,4,5,6,8,10,12,15 ],

    "min_child_weight" :[ 1,3,5,7 ],

    "gamma"            :[ 0.0,0.1,0.2,0.3,0.4 ],

    "colsample_bytree" :[ 0.3, 0.4, 0.5, 0.7 ]

}
# Hyperparameter optimization using RandomizedSearchCV

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

import xgboost
#timer

def timer(start_time=None):

    if not start_time:

        start_time = datetime.now()

        return start_time

    elif start_time:

        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)

        tmin, tsec = divmod(temp_sec, 60)

        print('\n Time taken: %i hours %i minutes and %s seconds. ' %(thour, tmin, round(tsec,2)))
classifier = xgboost.XGBClassifier()
random_search=RandomizedSearchCV(classifier, param_distributions=params, n_iter=5, scoring='roc_auc',n_jobs=-1, cv=5,verbose=3)
from datetime import datetime



start_time = timer(None)

random_search.fit(X,Y)

timer(start_time)
random_search.best_estimator_
random_search.best_params_
classifier = xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,

              colsample_bynode=1, colsample_bytree=0.7, gamma=0.3,

              learning_rate=0.1, max_delta_step=0, max_depth=6,

              min_child_weight=7, missing=None, n_estimators=100, n_jobs=1,

              nthread=None, objective='binary:logistic', random_state=0,

              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,

              silent=None, subsample=1, verbosity=1)
from sklearn.model_selection import cross_val_score

score = cross_val_score(classifier,X,Y,cv=10)

score
score.mean()