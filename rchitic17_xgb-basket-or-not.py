import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sn
from datetime import datetime
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv('../input/data.csv')
data.info()
object_vars=[var for var in data if data[var].dtype=='object']
numerical_vars=[var for var in data if data[var].dtype=='float' or data[var].dtype=='int']
for var in object_vars:
    print(data[var].value_counts())
#drop team id and team name columns(only one value - LAL)
data=data.drop(['team_id','team_name'],axis=1)
#set game date as datetime
#get info from matchup: home or away?
data['home']=data['matchup'].apply(lambda x: 1 if 'vs' in x else 0)
data=data.drop('matchup',axis=1)
#plt.subplot()
plt.figure(figsize=(5,5))
plt.scatter(x=data['loc_x'],y=data['loc_y'],alpha=0.02)

plt.figure(figsize=(5,5))
plt.scatter(x=data['lon'],y=data['lat'],alpha=.02)
#unnecessary columns, since they are correlated to loc_x and loc_y
data=data.drop(['lon','lat'],axis=1)
data['time_remaining_seconds']=data['minutes_remaining']*60+data['seconds_remaining']
data=data.drop(['minutes_remaining','seconds_remaining'],axis=1)
data['time_remaining_seconds']
data['last_3_seconds']=data.time_remaining_seconds.apply(lambda x: 1 if x<4 else 0)
#drop shot_id column (useless)
data=data.drop('shot_id',axis=1)
#visualize difference between shot types
fig,ax=plt.subplots()
sn.barplot(x='combined_shot_type',y='shot_made_flag',data=data)
#Replace the 20 least common action types with value 'Other'
rare_action_types=data['action_type'].value_counts().sort_values(ascending=True).index.values[:20]
data.loc[data['action_type'].isin(rare_action_types),'action_type']='Other'

#keep the three columns, they are not redundant
pd.DataFrame({'area':data.shot_zone_area,'basic':data.shot_zone_basic,'range':data.shot_zone_range}).head(10)
#drop playoffs - not relevant
sn.countplot('playoffs',hue='shot_made_flag',data=data)
data=data.drop('playoffs',1)
#get month from date
data['game_date']=pd.to_datetime(data['game_date'])
data['game_month']=data['game_date'].dt.month
data=data.drop('game_date',axis=1)
data=data.drop(['game_id','game_event_id'],axis=1)
data.info()

# transform categorical data to type 'category'
categorical_vars=['action_type','combined_shot_type','season','opponent','shot_type','period','shot_zone_basic','shot_zone_area','shot_zone_range','game_month']
for var in categorical_vars:
        data=pd.concat([data,pd.get_dummies(data[var],prefix=var)], 1)
        data=data.drop(var,1)
#separate train and test sets
train=data[pd.notnull(data['shot_made_flag'])]
test=data[pd.isnull(data['shot_made_flag'])]
y_train=train['shot_made_flag']
train=train.drop('shot_made_flag',1)
y_train=y_train.astype('int')
test=test.drop('shot_made_flag',1)
train.info()
#Correlation between numerical variables and shots made
sn.heatmap(data.corr())
#Evaluation with log loss
from sklearn.metrics import log_loss
from sklearn.metrics import make_scorer
#log_scorer=make_scorer(log_loss,greater_is_better=False)
def log_scorer(estimator, X, y):
    pred_probs = estimator.predict_proba(X)[:, 1]
    return log_loss(y, pred_probs)
#from xgboost.sklearn import XGBClassifier
#model = XGBClassifier(colsample_bytree= 0.8, learning_rate= 0.01, max_depth= 7, n_estimators= 400, seed= 1234, subsample= 0.5)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=400,max_depth=5)
model.fit(train, y_train)
pred=model.predict_proba(train)[:,1]
log_loss(y_train,pred)
target_y = model.predict_proba(test)[:,1]
log_loss(y_train,pred)

from sklearn.model_selection import cross_val_score
cv=cross_val_score(model,train,y_train,scoring=log_scorer,cv=5)
cv
sub = pd.read_csv("../input/sample_submission.csv")
sub['shot_made_flag'] = target_y
sub.to_csv("submission.csv", index=False)
