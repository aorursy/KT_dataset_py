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
# import all packages and set plots to be embedded inline

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 

plt.style.use('fivethirtyeight')
df = pd.read_csv(r'/kaggle/input/pubg-finish-placement-prediction/train_V2.csv')
df.head(10)
df.shape
df.info()
#statistical summary about the data

df.describe()
#check for missing data

df.isnull().sum()
#delete missing record

df.dropna(inplace=True)
df.isnull().sum()
#check for duplicates

df.duplicated().sum()
#showing the columns name and position

for i,col in enumerate(df.columns):

    print(i,col)
plt.figure(figsize=(22,10))

sns.distplot(df.assists,bins=80,kde=False)

plt.show()
plt.figure(figsize=(22,10))

sns.distplot(df.boosts,bins=80,kde=False,color='#0000A0')

plt.show()
plt.figure(figsize=(22,10))

sns.distplot(df.damageDealt,bins=80,kde=False,color='#800080')

plt.show()
g = pd.cut(df['DBNOs'],[-1,0,1,2,3,4,np.inf],labels=['0','1','2','3','4','+5']).value_counts()



#initializing plot

ax = g.plot.barh(color = '#007482', fontsize = 15)



#giving a title

ax.set(title = 'The Most Common Number of Knocks')



#x-label

ax.set_ylabel('Number of knocks', color = 'g', fontsize = '18')



#giving the figure size(width, height)

ax.figure.set_size_inches(22, 12)



#shwoing the plot

plt.show()
g = pd.cut(df['headshotKills'],[-1,0,1,2,3,4,np.inf],labels=['0','1','2','3','4','+5']).value_counts()



#initializing plot

ax = g.plot.bar(color = '#800080', fontsize = 15)



#giving a title

ax.set(title = 'The Most Common Number of Headshots')



#x-label

ax.set_ylabel('Number of Headshots', color = 'g', fontsize = '18')



#giving the figure size(width, height)

ax.figure.set_size_inches(22, 12)



#shwoing the plot

plt.show()
g = pd.cut(df['heals'],[-1,0,1,2,3,4,np.inf],labels=['0','1','2','3','4','+5']).value_counts()



#initializing plot

ax = g.plot.bar(color = '#FF00FF', fontsize = 15)



#giving a title

ax.set(title = 'The Most Common Number of Heals')



#x-label

ax.set_ylabel('Number of Heals', color = 'g', fontsize = '18')



#giving the figure size(width, height)

ax.figure.set_size_inches(22, 12)



#shwoing the plot

plt.show()
g = pd.cut(df['killPlace'],[-1,1,3,6,10,np.inf],labels=['1','2-3','4-6','7-10','+10']).value_counts()



#initializing plot

ax = g.plot.barh(color = '#808080', fontsize = 15)



#giving a title

ax.set(title = 'The Most Common Number of killplace')



#x-label

ax.set_ylabel('Number of killplace', color = 'g', fontsize = '18')



#giving the figure size(width, height)

ax.figure.set_size_inches(22, 12)



#shwoing the plot

plt.show()
plt.figure(figsize=(22,10))

sns.distplot(df.killPoints,bins=80,kde=False,color='#FF00FF')

plt.show()
g = pd.cut(df['kills'],[0,1,3,6,10,np.inf],labels=['0-1','2-3','4-6','7-10','+10']).value_counts()



#initializing plot

ax = g.plot.barh(color = '#FFA500', fontsize = 15)



#giving a title

ax.set(title = 'The Most Common Number of Kills')



#x-label

ax.set_ylabel('Number of Kills', color = 'g', fontsize = '18')



#giving the figure size(width, height)

ax.figure.set_size_inches(22, 12)



#shwoing the plot

plt.show()
plt.figure(figsize=(22,10))

sns.distplot(df.matchDuration,bins=80,kde=False,color='#808000')

plt.show()
plt.figure(figsize=(22,10))

label=df.matchType.value_counts().index

plt.pie(df.matchType.value_counts(),explode=[0.1]*len(label),labels=label,autopct='%.1f%%',shadow=True)

plt.axis('equal')

plt.title('User Type')

plt.show()
plt.figure(figsize=(22,10))

sns.distplot(df.rankPoints,bins=80,kde=False,color='#000080')

plt.show()
g = pd.cut(df['revives'],[-1,0,1,3,6,10,np.inf],labels=['0','1','2-30','4-6','7-10','+10']).value_counts()



#initializing plot

ax = g.plot.barh(color = '#00806A', fontsize = 15)



#giving a title

ax.set(title = 'The Most Common Number of Revives')



#x-label

ax.set_ylabel('Number of Revives', color = 'g', fontsize = '18')



#giving the figure size(width, height)

ax.figure.set_size_inches(22, 12)



#shwoing the plot

plt.show()
plt.figure(figsize=(22,10))

sns.distplot(df.rideDistance,bins=80,kde=False,color='#00806A')

plt.show()
plt.figure(figsize=(22,10))

sns.distplot(df.swimDistance,bins=80,kde=False,color='#158000')

plt.show()
plt.figure(figsize=(22,10))

sns.distplot(df.walkDistance,bins=80,kde=False,color='#006A80')

plt.show()
plt.figure(figsize=(22,10))

sns.distplot(df.winPlacePerc,bins=80,kde=False,color='#3CA6BC')

plt.show()
plt.figure(figsize=(22,10))

sns.scatterplot(x=df['winPlacePerc'],y=df['kills'])

plt.show()
plt.figure(figsize=(22,10))

sns.scatterplot(x=df['winPlacePerc'],y=df['walkDistance'])

plt.show()
plt.figure(figsize=(22,10))

sns.scatterplot(x="winPlacePerc", y="boosts", data=df)

plt.show()
plt.figure(figsize =(20,10))

sns.pointplot(x='vehicleDestroys',y='winPlacePerc',data=df,color='#606060',alpha=0.8)

plt.xlabel('Number of Vehicle Destroys',fontsize = 15,color='blue')

plt.ylabel('Win Percentage',fontsize = 15,color='blue')

plt.title('Vehicle Destroys/ Win Ratio',fontsize = 20,color='blue')

plt.show()
df.matchType=df.matchType.astype('category').cat.codes

plt.figure(figsize=(22, 15))

sns.heatmap(df.corr(), annot=True, linewidths=.5, fmt= '.1f')

plt.show()
#delete string columns ('Id','groupId','matchId')

#delete columns that had low correlation with winplaceperc ('Id','groupId','matchId')

df.drop(['Id','groupId','matchId','rankPoints','roadKills','vehicleDestroys'],axis=1,inplace=True)
#drop outliers from the data

for col in df.columns:

    df1=df[col]

    Q1 = df1.quantile(0.01)

    Q3 = df1.quantile(0.99)

    IQR = Q3-Q1

    minimum = Q1 - 1.5*IQR

    maximum = Q3 + 1.5*IQR

    condition = (df1 <= maximum) & (df1 >= minimum)

    df=df[condition]
#shape of data after deleting outliers

df.shape
#split the data

X=df.drop(['winPlacePerc'],axis=1)

y=df['winPlacePerc']
from sklearn.feature_selection import f_regression

from sklearn.feature_selection import SelectKBest
best_feature = SelectKBest(score_func=f_regression,k='all')

fit = best_feature.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)

dfcolumns = pd.DataFrame(X.columns)

featureScores = pd.concat([dfcolumns,dfscores],axis=1)

featureScores.columns = ['Feature','Score']

featureScores = featureScores.sort_values(by='Score',ascending=False).reset_index(drop=True)



featureScores
#select the best 15 feature

X= X[featureScores.Feature[:15].values]
from sklearn.preprocessing import StandardScaler

cols = X.columns

scaler = StandardScaler()

X=scaler.fit_transform(X)

X=pd.DataFrame(X,columns=cols)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10,random_state=42)
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
model_reg = cross_val_score(reg,X_train,y_train,cv=3,scoring='neg_mean_squared_error')
-model_reg
from sklearn.model_selection import GridSearchCV
param_grid={'fit_intercept':[True,False],'normalize':[True,False]}
grid= GridSearchCV(reg,param_grid,cv=3,scoring='neg_mean_squared_error')
grid.fit(X_train,y_train)
grid.best_estimator_
-grid.best_score_
from sklearn.linear_model import Lasso
lasso=Lasso()
model_lasso = cross_val_score(lasso,X_train,y_train,cv=3,scoring='neg_mean_squared_error')
-model_lasso
from sklearn.linear_model import ElasticNet
elastic=ElasticNet()
model_elastic = cross_val_score(elastic,X_train,y_train,cv=3,scoring='neg_mean_squared_error')
-model_elastic
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor()
model_tree = cross_val_score(tree,X_train,y_train,cv=3,scoring='neg_mean_squared_error')
- model_tree
from sklearn.ensemble import VotingRegressor
reg=LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=True)

tree=DecisionTreeRegressor()

regressor=[('Linear Regression', reg), ('decision Tree', tree)]

# i didn't use random forest in voting cause it took much time and i haven't now😢😢
vc = VotingRegressor(estimators=regressor)
vc.fit(X_train,y_train)
y_pred = vc.predict(X_test)
from sklearn.metrics import r2_score,mean_squared_error
mean_squared_error(y_test,y_pred)
r2_score(y_test,y_pred)
test = pd.read_csv(r'/kaggle/input/pubg-finish-placement-prediction/test_V2.csv')
test.head()
test_pred=test.copy()
test_pred = test_pred[X.columns]
test_pred=scaler.fit_transform(test_pred)

test_pred=pd.DataFrame(test_pred,columns=cols)
prediction = vc.predict(test_pred)
test['winPlacePerc'] = prediction
sub = pd.read_csv(r'/kaggle/input/pubg-finish-placement-prediction/sample_submission_V2.csv')
sub['winPlacePerc'] = test['winPlacePerc']
sub
sub.to_csv('submission.csv',index=False)