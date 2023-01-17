import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train_V2.csv')
train.head()
plt.figure(figsize=(15,10))
sns.countplot(x='kills',data=train,palette='RdBu_r')
sns.jointplot(x="winPlacePerc", y="kills", data=train, height=10, ratio=3)
plt.show()
kills = train.copy()

kills['killsCategories'] = pd.cut(kills['kills'], [-1, 0, 2, 5, 10, 60], labels=['0_kills','1-2_kills', '3-5_kills', '6-10_kills', '10+_kills'])

plt.figure(figsize=(15,8))
sns.boxplot(x="killsCategories", y="winPlacePerc", data=kills)
plt.show()
sns.jointplot(x="winPlacePerc", y="heals", data=train, height=10, ratio=3)
plt.show()
sns.jointplot(x="winPlacePerc", y="boosts", data=train, height=10, ratio=3)
plt.show()
x,y = plt.subplots(figsize=(15, 15))
sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=y)
plt.show()
train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')

data = train.copy()
data = data[data['playersJoined']>49]
plt.figure(figsize=(15,10))
sns.countplot(data['playersJoined'])
plt.title("Players Joined",fontsize=15)
plt.show()
train.head()
train = train.drop(['Id','groupId','matchId','playersJoined'],axis=1)
train['winPlacePerc']=pd.to_numeric(train['winPlacePerc'],errors = 'coerce')

train = pd.get_dummies(train,columns=['matchType'])
train.head()
train = train.dropna(how = 'any')

from sklearn.model_selection import train_test_split
X= train.drop('winPlacePerc',axis= 1)
y= train['winPlacePerc']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =.3, random_state = 101)

from sklearn.linear_model import LinearRegression
linear_model = LinearRegression()
linear_model.fit(X_train,y_train)
linear_model.score(X_train,y_train)

predictions = linear_model.predict(X_test)

from sklearn.metrics import mean_squared_error
linear_model_mse = mean_squared_error(predictions,y_test)
linear_model_mse
