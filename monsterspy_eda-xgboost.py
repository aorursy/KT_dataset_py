import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV

from sklearn.metrics import mean_squared_error, r2_score

import seaborn as sns

import xgboost as xgb

import math

from sklearn.metrics.scorer import make_scorer

pd.set_option('chained_assignment',None) 

import matplotlib.pyplot as plt

%matplotlib inline



import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning) 



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv("../input/train.csv", parse_dates = [0])

test = pd.read_csv("../input/test.csv", parse_dates = [0])

sample_sub = pd.read_csv("../input/sampleSubmission.csv")
def feature_extraction(df):

    df['year'] = df.datetime.dt.year

    df['month'] = df.datetime.dt.month

    df['dayofweek'] = df.datetime.dt.dayofweek

    df['hour'] = df.datetime.dt.hour

    df['day'] = df.datetime.dt.day
feature_extraction(train)

feature_extraction(test)
group_season = train.groupby(['season'])['count'].sum().reset_index()

ax = sns.barplot(x = group_season['season'], y = group_season['count'])

ax.set(xlabel='season', ylabel='count')

plt.show()
ax =  sns.distplot(np.log1p(train['count']))

ax.set(xlabel = 'log1p count')

plt.show()
ax = sns.boxplot(y = train['count'])

plt.show()
group_dow = train.groupby(['dayofweek'])['count'].sum().reset_index()

ax = sns.barplot(x = group_dow['dayofweek'], y = group_dow['count'])

ax.set(xlabel='dayofweek', ylabel='count')

plt.show()
group_mn = train.groupby(['month'])['count'].sum().reset_index()

ax = sns.barplot(x = group_mn['month'], y = group_mn['count'])

ax.set(xlabel='month', ylabel='count')

plt.show()
group_hr = train.groupby(['hour'])['count'].sum().reset_index()

ax = sns.barplot(x = group_hr['hour'], y = group_hr['count'])

ax.set(xlabel='hour', ylabel='count')

plt.show()
train.groupby(['year','month'])['count'].sum().plot(kind='bar')
matt = train[['hour','humidity','temp','dayofweek','count']].corr()

mask = np.array(matt)

mask[np.tril_indices_from(mask)] = False

fig,ax= plt.subplots()

fig.set_size_inches(8,6)

sns.heatmap(matt, mask = mask, vmax = .8 , annot = True)
def evalerror(preds, dtrain):

    labels = dtrain.get_label()

    assert len(preds) == len(labels)

    labels = labels.tolist()

    preds = preds.tolist()

    # I have added the max since applying regression we obtain negative values of preds

    # and therefore an error because of the logarithm

    terms_to_sum = [(math.log(labels[i] + 1) - math.log(max(0,preds[i]) + 1)) ** 2.0 

                for i,pred in enumerate(labels)]

    return 'error', (sum(terms_to_sum) * (1.0/len(preds))) ** 0.5
X = train.drop(['datetime','casual','registered','count'], axis = 1)

y = np.log1p(train['count'])

x_test = test.drop(['datetime'], axis = 1)
x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=4242)



d_train = xgb.DMatrix(x_train, label=y_train)

d_valid = xgb.DMatrix(x_valid, label=y_valid)

d_test = xgb.DMatrix(x_test)



params = {}

params['objective'] = 'reg:linear'

params['eta'] = 0.1

params['max_depth'] = 5



watchlist = [(d_train, 'train'), (d_valid, 'valid')]



clf = xgb.train(params, d_train, 2000, watchlist, early_stopping_rounds=50, feval = evalerror, maximize=False, verbose_eval=10)
xgb.plot_importance(clf)
p_test = np.expm1(clf.predict(d_test))

date = test['datetime']

res = pd.concat([date , pd.Series(p_test)], axis = 1)

res.columns = ['datetime','count']
res.head()
sns.boxplot(y = train['count'])
sns.boxplot(y = pd.Series(p_test))