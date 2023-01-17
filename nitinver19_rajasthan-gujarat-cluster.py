#importing relevant libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import os

from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score,GridSearchCV

import lightgbm as lgb

from xgboost import XGBRegressor

%matplotlib inline
train_data=pd.DataFrame(pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv'))

test_data=pd.DataFrame(pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv'))

train_data.info()
#checking the null values

train_data.isnull().sum()
#removing the null value

train_data = train_data[train_data["winPlacePerc"].isna() != True]
fig,ax=plt.subplots(figsize=(15,12))

ax=sns.heatmap(train_data.corr(),annot=True)
f, axes = plt.subplots(3,2, figsize=(10, 10))

sns.distplot(train_data['kills'], color="b", ax=axes[0, 0])

sns.distplot(train_data['walkDistance'], color="r", ax=axes[0, 1])

sns.distplot(train_data['killStreaks'], color="g", ax=axes[1, 0])

sns.distplot(train_data['longestKill'], color="m", ax=axes[1, 1])

sns.distplot(train_data['rideDistance'], color="g", ax=axes[2, 0])

sns.distplot(train_data['matchDuration'], color="m", ax=axes[2, 1])

plt.setp(axes, yticks=[])

plt.tight_layout()
#Specifying input and target variable

y=train_data['winPlacePerc']

X = train_data.drop(['winPlacePerc','Id','groupId','matchId','matchType'],axis=1,inplace=False)
X.columns
#Splitting into Train-Test Data

X_train,X_valid,y_train,y_valid=train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=0)
matches_num = train_data.loc[:,"matchId"].nunique()

print("There are {} matches registered in our data.".format(matches_num))
match_types = train_data.loc[:,"matchType"].value_counts().to_frame().reset_index()

match_types.columns = ["Type","Count"]

match_types
plt.figure(figsize=(15,8))

ticks = match_types.Type.values

ax = sns.barplot(x="Type", y="Count", data=match_types)

ax.set_xticklabels(ticks,rotation=60, fontsize=14)

ax.set_title("Match types")

plt.show()
plt.figure(figsize=(15,8))

ax1 = sns.boxplot(x="kills",y="damageDealt", data = train_data)

ax1.set_title("Damage Dealt vs. Number of Kills")

plt.show()
plt.figure(figsize=(15,8))

ax2 = sns.boxplot(x="DBNOs",y="kills", data = train_data)

ax2.set_title("Number of DBNOs vs. Number of Kills")

plt.show()
plt.figure(figsize = (15, 15))

sns.pointplot(train_data["heals"], train_data["walkDistance"],color = "blue",label='heals',linestyles="-")

sns.pointplot(train_data["boosts"], train_data["walkDistance"],color = "red",label='boosts',linestyles="--")

plt.xlabel("heals/boost")

plt.legend(['heals','boosts']) 

plt.grid()

plt.show()
plt.figure(figsize=(10,7))

sns.pointplot(x="kills", y="winPlacePerc", data=train_data)

plt.show()
x_test = test_data.drop(['Id','groupId','matchId','matchType'],axis=1)
import xgboost as xgb

model = xgb.XGBRegressor(max_depth=17, gamma=0.3, learning_rate= 0.1)

model.fit(X_train,y_train)
xgb.plot_importance(model)
preds = model.predict(x_test)
test_id = test_data["Id"]

submit_xgb = pd.DataFrame({'Id': test_id, "winPlacePerc": preds} , columns=['Id', 'winPlacePerc'])

print(submit_xgb.head())

submit_xgb.to_csv("submission.csv", index = False)