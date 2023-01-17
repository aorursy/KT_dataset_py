import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import clone
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from xgboost import XGBClassifier
import copy
from sklearn import model_selection
from sklearn import ensemble
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

import plotly.offline as py
py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.head()
trace = go.Histogram(x=train['Target'].values)

layout = go.Layout(
    title="Histogram with Frequency Count"
)

fig = go.Figure(data=go.Data([trace]), layout=layout)
py.iplot(fig)
# train[train.columns[train.isna().sum()!=0]]
train[train.columns[train.isna().sum()!=0]] = train[train.columns[train.isna().sum()!=0]].fillna(0)
for df in (train, test):
    df['RentByRoom'] = df['v2a1']/df['rooms']
    df['TabletsByPeople'] = df['v18q1']/df['r4t3']
    df['SizeByPeople'] = df['tamhog']/df['r4t3']
    df['PhoneByPeople'] = df['qmobilephone']/df['r4t3']
    df['PeopleByRoom'] = df['r4t3']/df['rooms']    
y = train['Target']
X = train.drop(['Target', 'Id'], axis=1)
test_id = test['Id']
test.drop('Id', axis=1, inplace=True)
train_test_df = pd.concat([X, test], axis=0)
cols = [col for col in train_test_df.columns if train_test_df[col].dtype == 'object']

le = LabelEncoder()
for col in cols:
    le.fit(train_test_df[col])
    X[col] = le.transform(X[col])
    test[col] = le.transform(test[col])
def get_lgb_model():
    lgb_model = lgb.LGBMClassifier(objective='multiclass',num_leaves=144,
                      learning_rate=0.05, n_estimators=300, max_depth=13,
                      metric='merror',is_training_metric=True,
                      max_bin = 55, bagging_fraction = 0.8,verbose=-1,
                      bagging_freq = 5, feature_fraction = 0.9) 
    return lgb_model
lgb_model = get_lgb_model()
lgb_model.fit(X, y)
target_hat = lgb_model.predict(test)
pred = pd.DataFrame({'Id': test_id, 'Target': target_hat})
pred.to_csv('submission.csv', index=False)