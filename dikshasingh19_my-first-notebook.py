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
#We will start will importing all the libraries

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt



import sklearn

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import RFE

from sklearn.linear_model import LinearRegression



import statsmodels.api as sm

from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.metrics import r2_score



#we will import warnings

import warnings

warnings.filterwarnings('ignore')
# We will read our dataset

#and check for first 5 records.



bike=pd.read_csv("/kaggle/input/boombikes/day.csv")

bike.head()
#check for shape(how many rows and columns we have)

bike.shape
#check for info)(what are the datatypes)

bike.info()
#check for columns

bike.columns
#check for some statistics

bike.describe()
#check for null values if any

bike.isnull().sum()
g=sns.PairGrid(bike, vars=["casual", "registered","cnt"])

g = g.map(plt.scatter)
# Drop 'casual' and 'registered'.

bike.drop('casual',axis=1,inplace=True)

bike.drop('registered',axis=1,inplace=True)
#It was unnessary there in the data so I drop the instant column and date column to remove the redundancy

bike.drop('instant',axis=1,inplace=True)

bike.drop('dteday',axis=1,inplace=True)
bike.head()
# Plotting heatmap

plt.figure(figsize=(16,10))

ax=sns.heatmap(bike.corr(),annot=True,cmap='YlGnBu')

bottom,top=ax.get_ylim()

ax.set_ylim(bottom+0.5,top-0.5)

plt.show()
# Drop temp

bike.drop('temp',axis=1,inplace=True)
#check bike.head()

bike.head()
# we will make pairplot to visualize the numerical variable and checking some linear relation.

sns.set(style="ticks", color_codes=True)

sns.pairplot(bike,vars=["atemp","hum","windspeed","cnt"])

plt.show()
map_season={1:'spring', 2:'summer', 3:'fall', 4:'winter'}

bike['season']=bike['season'].map(map_season)

bike.season.value_counts()
map_mnth={1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}

bike['mnth']=bike['mnth'].map(map_mnth)

bike.mnth.value_counts()
map_weekday= {0: 'Sunday', 1: 'Monday', 2: 'Tuesday', 3: 'Wednesday', 4: 'Thursday', 5: 'Friday', 6: 'Saturday'}

bike['weekday']=bike['weekday'].map(map_weekday)

bike.weekday.value_counts()
map_weathersit={1: 'Clear', 2: 'Mist + Cloudy', 3: 'Light Snow', 4: 'Heavy Rain + Ice Pallets'}

bike['weathersit']=bike['weathersit'].map(map_weathersit)

bike.weathersit.value_counts()
# to visualize categorical and continous variable we will see boxplots.

plt.figure(figsize=(20,12))

plt.subplot(3,3,1)

sns.boxplot(x='season',y='cnt',data=bike)

plt.subplot(3,3,2)

sns.boxplot(x='yr',y='cnt',data=bike)

plt.subplot(3,3,3)

sns.boxplot(x='mnth',y='cnt',data=bike)

plt.subplot(3,3,4)

sns.boxplot(x='holiday',y='cnt',data=bike)

plt.subplot(3,3,5)

sns.boxplot(x='weekday',y='cnt',data=bike)

plt.subplot(3,3,6)

sns.boxplot(x='workingday',y='cnt',data=bike)

plt.subplot(3,3,7)

sns.boxplot(x='weathersit',y='cnt',data=bike)

plt.show()
# we will do one-hot encoding for season

season_dum=pd.get_dummies(bike['season'],drop_first=True)

season_dum.head()
#  we will do one-hot encoding for mnth

mnth_dum=pd.get_dummies(bike['mnth'],drop_first=True)

mnth_dum.head()
# we will do one-hot encoding for weekday

weekday_dum=pd.get_dummies(bike['weekday'],drop_first=True)

weekday_dum.head()
weathersit_dum=pd.get_dummies(bike['weathersit'],drop_first=True)

weathersit_dum.head()
# after that we will concatenate all the three with our original dataset.

bike=pd.concat([bike,season_dum,mnth_dum,weekday_dum,weathersit_dum],axis=1)

bike.head()
# As we have dummified columns for mnth,season, weekday,weathersit so now will drop them from our dataset.

bike.drop('mnth',axis=1,inplace=True)

bike.drop('season',axis=1,inplace=True)

bike.drop('weekday',axis=1,inplace=True)

bike.drop('weathersit',axis=1,inplace=True)
bike.head()
bike.info()
# Perform Train and Test split

bike_train,bike_test=train_test_split(bike,train_size=0.7,random_state=80)

print(bike_train.shape)

print(bike_test.shape)
#we will do scaling so initialise scaler

scaler=MinMaxScaler()
# It has to be performed on numerical value so will fit_transform on our train data.

num_vars=['atemp','hum','windspeed','cnt']

bike_train[num_vars]=scaler.fit_transform(bike_train[num_vars])

bike_train.head()
# Describe the numerical variables.

bike_train[num_vars].describe()
# Divide train data into X and y.

X_train=bike_train

y_train=bike_train.pop('cnt')
#Initialise model Instance

lm=LinearRegression()
# we will fit() on train data

lm.fit(X_train,y_train)
# Pass model instance and no. of variables to RFE to let the model select features on its own.

rfe=RFE(lm,15)

rfe.fit(X_train,y_train)
# theses are the features that model has selected.

list(zip(X_train.columns,rfe.support_,rfe.ranking_))
#columns that RFE is supportiing

col=X_train.columns[rfe.support_]

col
#columns that RFE does not support.

X_train.columns[~rfe.support_]
# Now will train the model using features that RFE has selected.

X_train_rfe=X_train[col]

X_train_rfe=sm.add_constant(X_train_rfe) # adding constant

lm=sm.OLS(y_train,X_train_rfe).fit()    # Running linear mode;

print(lm.summary())
#checking VIF of all the variables.

vif=pd.DataFrame()

X=X_train_rfe

vif['Features']=X.columns

vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]

vif['VIF']=round(vif['VIF'],2)

vif=vif.sort_values(by="VIF",ascending=False)

vif
X_train_new1= X_train_rfe.drop('const',axis=1)
#checking VIF of all the variables.

vif=pd.DataFrame()

X=X_train_new1

vif['Features']=X.columns

vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]

vif['VIF']=round(vif['VIF'],2)

vif=vif.sort_values(by="VIF",ascending=False)

vif
#Dropping Februry and training the model

X_train_new2=X_train_new1.drop(["February"],axis=1)

X_train_lm2=sm.add_constant(X_train_new2)

lm=sm.OLS(y_train,X_train_lm2).fit()

print(lm.summary())
#check VIF again

vif=pd.DataFrame()

X=X_train_new2

vif['Features']=X.columns

vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]

vif['VIF']=round(vif['VIF'],2)

vif=vif.sort_values(by="VIF",ascending=False)

vif
# dropping holiday and training the model.

X_train_new3=X_train_new2.drop(["hum"],axis=1)

X_train_lm3=sm.add_constant(X_train_new3)

lm=sm.OLS(y_train,X_train_lm3).fit()

print(lm.summary())
#checking VIF

vif=pd.DataFrame()

X=X_train_new3

vif['Features']=X.columns

vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]

vif['VIF']=round(vif['VIF'],2)

vif=vif.sort_values(by="VIF",ascending=False)

vif
#dropping hum and training the model

X_train_new4=X_train_new3.drop(["holiday"],axis=1)

X_train_lm4=sm.add_constant(X_train_new4)

lm=sm.OLS(y_train,X_train_lm4).fit()

print(lm.summary())
#checking VIF again to check multicollinearity.

vif=pd.DataFrame()

X=X_train_new4

vif['Features']=X.columns

vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]

vif['VIF']=round(vif['VIF'],2)

vif=vif.sort_values(by="VIF",ascending=False)

vif
# I decided to drop January also on the basis of p value.

X_train_new5=X_train_new4.drop(["January"],axis=1)

X_train_lm5=sm.add_constant(X_train_new5)

lm=sm.OLS(y_train,X_train_lm5).fit()

print(lm.summary())
#checking VIF

vif=pd.DataFrame()

X=X_train_new5

vif['Features']=X.columns

vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]

vif['VIF']=round(vif['VIF'],2)

vif=vif.sort_values(by="VIF",ascending=False)

vif
# ro reduce the number of extra variables I am dropping July on the basis of p value.

X_train_new6=X_train_new5.drop(["July"],axis=1)

X_train_lm6=sm.add_constant(X_train_new6)

lm=sm.OLS(y_train,X_train_lm6).fit()

print(lm.summary())
#checking VIF 

vif=pd.DataFrame()

X=X_train_new6

vif['Features']=X.columns

vif['VIF']=[variance_inflation_factor(X.values,i) for i in range(X.shape[1])]

vif['VIF']=round(vif['VIF'],2)

vif=vif.sort_values(by="VIF",ascending=False)

vif
#calculting Predicted value of y for train set.

y_train_pred=lm.predict(X_train_lm6)

y_train_pred
#visualizing error terms.

fig=plt.figure()

res=y_train - y_train_pred

sns.distplot(res,bins=20)

fig.suptitle('Error Terms', fontsize = 20)                  # Plot heading 

plt.xlabel('Errors', fontsize = 18)     
#So performing scaling on Test by Transform().

num_vars=['atemp','hum','windspeed','cnt']

bike_test[num_vars]=scaler.transform(bike_test[num_vars])

bike_test.head()
#Describing numerical variables for the test set.

bike_test[num_vars].describe()
# Decide X and y for the test set.

X_test=bike_test

y_test=bike_test.pop('cnt')
# adding constant as our final trained model also having constant.

X_test_new=X_test[X_train_new6.columns]

X_test_new=sm.add_constant(X_test_new)

X_test_new.head()
#predicting Y values on our unseen data(Test data)

y_test_pred=lm.predict(X_test_new)
#checking R2 for our predicted model.

r2=r2_score(y_true=y_test,y_pred=y_test_pred)

print(r2)
#checking adjusted R squared for our predicted model.

adj_r2 = 1 - (1-r2)*(len(bike) - 1) / (len(bike) - (bike.shape[1] - 1) - 1)

print(adj_r2)
#Evaluating the final model.

fig = plt.figure()

plt.scatter(y_test,y_test_pred)

fig.suptitle('y_test vs y_tet_pred', fontsize=20)           # Plot heading 

plt.xlabel('y_test', fontsize=18)                          # X-label

plt.ylabel('y_pred', fontsize=16)                          # Y-label