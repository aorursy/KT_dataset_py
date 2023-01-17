import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df_sa = pd.read_csv('../input/articles-sharing-reading-from-cit-deskdrop/shared_articles.csv')

df_ui = pd.read_csv('../input/articles-sharing-reading-from-cit-deskdrop/users_interactions.csv')
#function to calculate virality

def virality_result(row): 

    return (1* row['VIEW']) + (4*row['LIKE']) + (10*row['COMMENT CREATED']) + (25*row['FOLLOW']) + (100*row['BOOKMARK'])



#rearrange using a pivot table to find the count of each event to use to calculate virality

df_ui['COUNTER'] = 1

group_data = df_ui.groupby(['contentId','eventType'])['COUNTER'].sum().reset_index() 



events_df = group_data.pivot_table('COUNTER', ['contentId'], 'eventType')

events_df = events_df.fillna(0)



events_df['virality'] = events_df.apply(lambda row: virality_result(row), axis=1)

events_df = events_df.fillna(0)

def invested_interactions(row):

    return row['FOLLOW'] + row['BOOKMARK']



def feedback_interactions(row):

    return row['LIKE'] + row['COMMENT CREATED']



events_df['INVESTED INTERACTIONS'] = events_df.apply(lambda row: invested_interactions(row), axis=1)

events_df['FEEDBACK INTERACTIONS'] = events_df.apply(lambda row: feedback_interactions(row), axis=1)

events_df = events_df.fillna(0)
import seaborn as sns

import matplotlib.pyplot as plot 

from sklearn.model_selection import train_test_split



dummy = pd.get_dummies(df_sa['eventType']) #turn categorical data into numerical data



df_sa = pd.concat([df_sa, dummy], axis=1)

events_df = pd.merge(df_sa, events_df, on='contentId', how='inner') #inner join the data



train, test = train_test_split(events_df, test_size=0.2)

plot.figure(figsize=(16,12))

sns.heatmap(train.corr(),annot=True,fmt=".2f") #
from sklearn import model_selection

independent_variables = ['VIEW', 'LIKE', 'COMMENT CREATED', 'FOLLOW','BOOKMARK' ]



X = events_df[independent_variables]

y = events_df.virality

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.80,test_size=0.20, random_state=10)
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from math import sqrt



def baseline(baseline, baseline_type, X_train, X_test, y_train, y_test):

    print('\n' + baseline_type)

    baseline.fit(X_train, y_train)



    print('Train')

    prediction_train = baseline.predict(X_train)

    r_squared = r2_score(y_train, prediction_train)

    adj_r2 = (1 - (1 - r_squared) * ((X_train.shape[0] - 1) / 

          (X_train.shape[0] - X_train.shape[1] - 1)))

    print('MAE ', mean_absolute_error(y_train, prediction_train))

    print('RMSE ', np.sqrt(mean_squared_error(y_train, prediction_train)))

    print('Adjusted R2 ', adj_r2)



    print('\nTest')

    prediction_test = baseline.predict(X_test)

    r_squared = r2_score(y_test, prediction_test)

    adj_r2 = (1 - (1 - r_squared) * ((X_test.shape[0] - 1) / 

          (X_test.shape[0] - X_test.shape[1] - 1)))

    print('MAE ', mean_absolute_error(y_test, prediction_test))

    print('RMSE ', np.sqrt(mean_squared_error(y_test,prediction_test)))

    print('Adjusted R2', adj_r2)

from sklearn.linear_model import Lasso

from sklearn.linear_model import ElasticNetCV

from sklearn.linear_model import RidgeCV



lasso = Lasso(alpha=0.01, max_iter=1000)

ridge = RidgeCV(cv = 5)

elastic = ElasticNetCV(cv=5, random_state=0, max_iter=1000)



baseline(lasso, "---LASSO---", X_train, X_test, y_train, y_test)

baseline(ridge, "---RIDGE---", X_train, X_test, y_train, y_test)

baseline(elastic, "---ELASTIC---", X_train, X_test, y_train, y_test)