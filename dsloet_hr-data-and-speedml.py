# data analysis and wrangling

import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
df = pd.read_csv('../input/HR_comma_sep.csv')
df.head()
len(df)
from sklearn.model_selection import train_test_split

train, test = train_test_split(df, test_size = 0.3)

len(train)

len(test)
train.head()
test.head()
print(train.columns.values)
train.info()
train.describe()
train.describe(include=['O'])
from speedml import Speedml
train.to_csv('train.csv')
test.to_csv('test.csv')
sml = Speedml('train.csv', 'test.csv', target = 'left')
sml.train.head()
sml.plot.correlate()
sml.plot.distribute()
sml.configure('overfit_threshold',sml.np.sqrt(sml.train.shape[0])/sml.train.shape[0])
sml.feature.density('satisfaction_level')
sml.train[['satisfaction_level', 'satisfaction_level_density']].head()
sml.feature.density('last_evaluation')
sml.train[['last_evaluation', 'last_evaluation_density']].head()
sml.plot.crosstab('left', 'satisfaction_level')
sml.plot.crosstab('left', 'last_evaluation')
sml.plot.crosstab('left', 'salary')
sml.plot.crosstab('left', 'sales')
sml.feature.labels(['sales'])

sml.train.head()
sml.feature.labels(['salary'])

sml.train.head()
sml.eda()
sml.plot.continuous('promotion_last_5years')
sml.plot.crosstab('left', 'promotion_last_5years')
sml.plot.importance()
sml.feature.drop('Unnamed: 0')
sml.plot.importance()
sml.feature.outliers('promotion_last_5years', upper=97)
sml.plot.continuous('promotion_last_5years')
sml.plot.importance()
sml.feature.drop('promotion_last_5years')
sml.plot.correlate()
sml.model.data()
select_params = {'max_depth': [11,12,13], 'min_child_weight': [0,1,2]}

fixed_params = {'learning_rate': 0.1, 'subsample': 0.8, 

                'colsample_bytree': 0.8, 'seed':0, 

                'objective': 'binary:logistic'}



sml.xgb.hyper(select_params, fixed_params)
select_params = {'learning_rate': [0.3, 0.1, 0.01], 'subsample': [0.7,0.8,0.9]}

fixed_params = {'max_depth': 12, 'min_child_weight': 0, 

                'colsample_bytree': 0.8, 'seed':0, 

                'objective': 'binary:logistic'}



sml.xgb.hyper(select_params, fixed_params)


tuned_params = {'learning_rate': 0.1, 'subsample': 0.9, 

                'max_depth': 12, 'min_child_weight': 0,

                'seed':0, 'colsample_bytree': 0.8, 

                'objective': 'binary:logistic'}

sml.xgb.cv(tuned_params)
sml.xgb.cv_results.tail(10)
tuned_params['n_estimators'] = sml.xgb.cv_results.shape[0] - 1

sml.xgb.params(tuned_params)


sml.xgb.classifier()
sml.model.evaluate()
sml.plot.model_ranks()
sml.model.ranks()
sml.xgb.fit()

sml.xgb.predict()



sml.xgb.feature_selection()
sml.xgb.sample_accuracy()