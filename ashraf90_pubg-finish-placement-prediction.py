# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Import libiraies

import pandas as pd     

import numpy as np      

import matplotlib.pyplot as plt  

%matplotlib inline

import seaborn as sns  
# read the data

train = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')

train.head()
#See the no.of rows and features

train.shape
pd.options.display.max_columns = 60
#More information about the features and the data types

train.info()
# It is a time to detect the Missing values so let's see how many missing values in our data  

train.isna().sum()
# Here I dropped this missing value 

train.drop(2744604,inplace=True)
#Finally we haven't have any missing data

train.isna().sum().sum()
#Here we collect all these distance that user walked into one feature 

train['all_distance']=train['rideDistance']+train['walkDistance']+train['swimDistance']

#Delete the old feature of distance 

train.drop(columns=['walkDistance','swimDistance','rideDistance'],axis=1,inplace=True)
#Here  we collect all the medicine that user taked into one feature 

train['medicine']=train['boosts']+train['heals']



train.drop(columns=['boosts','heals'],axis=1,inplace=True)


train.drop(columns=['Id','groupId','matchId','rankPoints','matchType'],inplace=True)
sns.distplot(train['medicine'])

#Drop users that used more than 40 medicine help which may be hacker

train.drop(train[train['medicine'] > 40].index, inplace=True)
#Distribution of  medicine feature after detect outliers 

sns.distplot(train['medicine'])
#Let's see the distribution of Kills of user

sns.distplot(train['kills'])
# Drop the user that kill more than 20 kills 

train.drop(train[train['kills'] > 20].index, inplace=True)
#distribution of kills feature after 

sns.distplot(train['kills'],bins=[0,5,10,15,20])
sns.distplot(train['longestKill'])
# Drop the user that kill an enemy from more than 600 kills 

train.drop(train[train['longestKill'] > 600].index, inplace=True)
sns.distplot(train['longestKill'])
sns.distplot(train.killStreaks)
train.drop(train[train['killStreaks']>12].index,inplace=True)
sns.distplot(train.killStreaks)
sns.distplot(train.weaponsAcquired)
train.drop(train[train['weaponsAcquired'] >= 80].index, inplace=True)
sns.distplot(train['weaponsAcquired'])
train.head()
train.shape
print('We delete {} rows and {} columns'.format(4446966-4445429,29-21))
# define dependant and independant features

X=train.drop(columns=['winPlacePerc'])

y=train['winPlacePerc']
from sklearn.feature_selection import f_regression

from sklearn.feature_selection import SelectKBest

best_feature = SelectKBest(score_func=f_regression,k='all')

fit = best_feature.fit(X,y)
score = pd.DataFrame(fit.scores_)

columns = pd.DataFrame(X.columns)

featureScores = pd.concat([columns,score],axis=1)

featureScores.columns = ['Feature','Score']

featureScores = featureScores.sort_values(by='Score',ascending=False).reset_index(drop=True)



featureScores
# Select the most 10 features 

X= X[featureScores.Feature[:10].values]


from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.pipeline import Pipeline

from sklearn.model_selection import cross_val_score

from sklearn.preprocessing import StandardScaler
#Split data into train and test data

X_train ,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
# prepare steps to pipeline with linear regression model

steps=[('Scalar',StandardScaler()), ('Linear Regression',LinearRegression())]
pipeline = Pipeline(steps)
pipeline.fit(X_train,y_train)
y_pred=pipeline.predict(X_test)


print(' Linear Regression score {}'.format(pipeline.score(X_test,y_test)))
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import *
print('Root mean square error {}'.format(np.sqrt(mean_squared_error(y_test,y_pred))))
# Random forest regressor  

regressor = RandomForestRegressor(n_estimators = 20, random_state = 0)
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
accuracy=r2_score(y_test,y_pred)

print('The accuracy is {}'.format(accuracy))
from sklearn.metrics import mean_squared_error

print('Root mean square error {}'.format(np.sqrt(mean_squared_error(y_test,y_pred))))
test = pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')
#Here we collect all these distance that user walked into one feature 

test['all_distance']=test['rideDistance']+test['walkDistance']+test['swimDistance']

#Delete the old feature of distance 

test.drop(columns=['walkDistance','swimDistance','rideDistance'],axis=1,inplace=True)

#Here  we collect all the medicine that user taked into one feature 

test['medicine']=test['boosts']+test['heals']



test.drop(columns=['boosts','heals'],axis=1,inplace=True)

test.drop(columns=['Id','groupId','matchId','rankPoints','matchType'],inplace=True)

test.drop(test[test['medicine'] > 40].index, inplace=True)

test.drop(test[test['kills'] > 20].index, inplace=True)

test.drop(test[test['longestKill'] > 600].index, inplace=True)

test.drop(test[test['killStreaks']>12].index,inplace=True)

test.drop(test[test['weaponsAcquired'] >= 80].index, inplace=True)

test.head()
# import submission data

submission=pd.read_csv('/kaggle/input/pubg-finish-placement-prediction/sample_submission_V2.csv')
test1=test
test1 = test1[X.columns]
test1=StandardScaler().fit_transform(test1)

test1=pd.DataFrame(test1,columns=X.columns)
prediction = regressor.predict(test1)
test['winPlacePerc'] = prediction
submission['winPlacePerc'] = test['winPlacePerc']
submission
submission.to_csv('submission.csv',index=False)