import pandas as pd #Analysis 
import matplotlib.pyplot as plt #Visulization
import seaborn as sns #Visulization
import numpy as np #Analysis 
from scipy.stats import norm #Analysis 
from sklearn.preprocessing import StandardScaler #Analysis 
from scipy import stats #Analysis 
import warnings 
warnings.filterwarnings('ignore')
%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
df_train = pd.read_csv('../input/train_V2.csv')
df_test  = pd.read_csv('../input/test_V2.csv')
df_train.shape, df_test.shape
df_train.head()
df_test.head()
df_train[df_train['groupId']==2]
len(df_train[df_train['matchId']==0]) 
temp = df_train[df_train['matchId']==0]['groupId'].value_counts().sort_values(ascending=False)
trace = go.Bar(
    x = temp.index,
    y = (temp)
)
data = [trace]
layout = go.Layout(
    title = "GroupId of Match Id:0",
    xaxis=dict(
        title='groupId',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of groupId of type of MatchId 0',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
temp = df_train['assists'].value_counts().sort_values(ascending=False)

#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp)
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='assists',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of assists',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
temp = df_train['kills'].value_counts().sort_values(ascending=False)

trace = go.Bar(
    x = temp.index,
    y = (temp)
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='kills',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of kills',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
temp = df_train['killStreaks'].value_counts().sort_values(ascending=False)

trace = go.Bar(
    x = temp.index,
    y = (temp)
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='killStreaks',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of killStreaks',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
temp = df_train['roadKills'].value_counts().sort_values(ascending=False)

trace = go.Bar(
    x = temp.index,
    y = (temp)
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='roadKills',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of roadKills',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
df_train['roadKills'].value_counts()
temp = df_train['teamKills'].value_counts().sort_values(ascending=False)

trace = go.Bar(
    x = temp.index,
    y = (temp),
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='teamKills',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of teamKills',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
0
df_train['teamKills'].value_counts()
f, ax = plt.subplots(figsize=(18, 8))
sns.distplot(df_train['longestKill'])
temp = df_train['weaponsAcquired'].value_counts().sort_values(ascending=False)

trace = go.Bar(
    x = temp.index,
    y = (temp),
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='weaponsAcquired',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of weaponsAcquired',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
temp = df_train['headshotKills'].value_counts().sort_values(ascending=False)

trace = go.Bar(
    x = temp.index,
    y = (temp),
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='headshotKills',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of headshotKills',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
df_train['headshotKills'].value_counts()
temp = df_train['DBNOs'].value_counts().sort_values(ascending=False)

trace = go.Bar(
    x = temp.index,
    y = (temp),
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='DBNOs',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of DBNOs',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
temp = df_train['boosts'].value_counts().sort_values(ascending=False)

#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp),
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='boosts',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of boosts',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
temp = df_train['heals'].value_counts().sort_values(ascending=False)

#print("Total number of states : ",len(temp))
trace = go.Bar(
    x = temp.index,
    y = (temp),
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='heals',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of heals',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
f, ax = plt.subplots(figsize=(8, 6))
sns.distplot(df_train['damageDealt'])
f, ax = plt.subplots(figsize=(8, 6))
df_train['revives'].value_counts().sort_values(ascending=False).plot.bar()
f, ax = plt.subplots(figsize=(18, 8))
sns.distplot(df_train['walkDistance'])
f, ax = plt.subplots(figsize=(18, 8))
sns.distplot(df_train['rideDistance'])
temp = df_train['vehicleDestroys'].value_counts().sort_values(ascending=False)

trace = go.Bar(
    x = temp.index,
    y = (temp),
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='vehicleDestroys',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of vehicleDestroys',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
df_train['vehicleDestroys'].value_counts()
temp = df_train['weaponsAcquired'].value_counts().sort_values(ascending=False)

trace = go.Bar(
    x = temp.index,
    y = (temp),
)
data = [trace]
layout = go.Layout(
    title = "",
    xaxis=dict(
        title='weaponsAcquired',
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
    ),
    yaxis=dict(
        title='Count of weaponsAcquired',
        titlefont=dict(
            size=16,
            color='rgb(107, 107, 107)'
        ),
        tickfont=dict(
            size=14,
            color='rgb(107, 107, 107)'
        )
)
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='schoolStateNames')
import os
import time
import gc
import warnings
warnings.filterwarnings("ignore")
# data manipulation
import json
from pandas.io.json import json_normalize
import numpy as np
import pandas as pd
# plot
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
# model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb
train_idx = df_train.Id
test_idx = df_test.Id
df_train["winPlacePerc"] = df_train["winPlacePerc"].astype('float')
train_y = df_train["winPlacePerc"]
train_target = df_train.groupby("Id")["winPlacePerc"].mean()

y_train = df_train["winPlacePerc"]
x_train = df_train.drop(["winPlacePerc"], axis=1)
x_test = df_test.copy()
folds = KFold(n_splits=5,random_state=6)
oof_preds = np.zeros(x_train.shape[0])
sub_preds = np.zeros(x_test.shape[0])

start = time.time()
valid_score = 0
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(x_train, y_train)):
    trn_x, trn_y = x_train.iloc[trn_idx], y_train[trn_idx]
    val_x, val_y = x_train.iloc[val_idx], y_train[val_idx]    
    
    train_data = lgb.Dataset(data=trn_x, label=trn_y)
    valid_data = lgb.Dataset(data=val_x, label=val_y)
    
    params = {"objective" : "regression", "metric" : "mae", 'n_estimators':10000, 'early_stopping_rounds':100,
              "num_leaves" : 30, "learning_rate" : 0.1, "bagging_fraction" : 0.9,
               "bagging_seed" : 0}
    
    lgb_model = lgb.train(params, train_data, valid_sets=[train_data, valid_data], verbose_eval=1000) 
    
    oof_preds[val_idx] = lgb_model.predict(val_x, num_iteration=lgb_model.best_iteration)
    oof_preds[oof_preds>1] = 1
    oof_preds[oof_preds<0] = 0
    sub_pred = lgb_model.predict(x_test, num_iteration=lgb_model.best_iteration) / folds.n_splits
    sub_pred[sub_pred>1] = 1 # should be greater or equal to 1
    sub_pred[sub_pred<0] = 0 
    sub_preds += sub_pred
    print('Fold %2d RMSE : %.6f' % (n_fold + 1, mean_absolute_error(val_y, oof_preds[val_idx])))
    valid_score += mean_absolute_error(val_y, oof_preds[val_idx])
test_pred = pd.DataFrame({"Id":test_idx})
test_pred["winPlacePerc"] = sub_preds
test_pred.columns = ["Id", "winPlacePerc"]
test_pred.to_csv("lgb_base_model.csv", index=False) # submission




