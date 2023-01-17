# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data=pd.read_csv('../input/train.csv')
train_data.head()
from datetime import datetime
train_data['date']  = train_data.datetime.apply(lambda x: x.split()[0])
train_data['hour'] = train_data.datetime.apply(lambda x: x.split()[1].split(':')[0])
train_data['weekday'] = train_data.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').weekday())
train_data['month'] = train_data.date.apply(lambda dateString : datetime.strptime(dateString, '%Y-%m-%d').month)
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline
fig,axes = plt.subplots(ncols=2,nrows=2)
fig.set_size_inches(15,10)
sns.regplot(data=train_data,x='temp',y='count',ax=axes[0][0])
sns.regplot(data=train_data,x='atemp',y='count',ax=axes[0][1])
sns.regplot(data=train_data,x='humidity',y='count',ax=axes[1][0])
sns.regplot(data=train_data,x='windspeed',y='count',ax=axes[1][1])
plt.show()
X_Train=train_data.drop(columns=['datetime','count','date','casual','registered'], axis=1)
Y_Train=train_data['count']
from sklearn.ensemble import GradientBoostingRegressor

params={'n_estimators':500, 'max_depth': 6,'min_samples_split': 2,
        'learning_rate': 0.01, 'loss':'ls'}

gbr_model=GradientBoostingRegressor(**params)
gbr_model.fit(X_Train, Y_Train)
gbr_model.score(X_Train, Y_Train)
prediction=gbr_model.predict(X_Train)
i = 0
for v in prediction:
    if prediction[i] < 0:
        prediction[i] = 0
    i = i + 1

import numpy as np
def rmsle(prediction, actual):
    log1 = np.nan_to_num(np.array([np.log(v + 1) for v in prediction]))
    log2 = np.nan_to_num(np.array([np.log(v + 1) for v in actual]))
    calc = (log1 - log2) ** 2
    return np.sqrt(np.mean(calc))

plt.figure(figsize=(5, 5))
plt.scatter(prediction, Y_Train)
plt.plot( [0,1000],[0,1000], color='red')
plt.xlim(-100, 1000)
plt.ylim(-100, 1000)
plt.xlabel('prediction')
plt.ylabel('Y_Train')
plt.title('Gradeint Boosting Regression Model')

print("RMSLE: ", rmsle(prediction, Y_Train))

