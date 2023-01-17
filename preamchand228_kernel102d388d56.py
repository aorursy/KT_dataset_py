# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv("/kaggle/input/2019-coronavirus-dataset-01212020-01262020/2019_nC0v_20200121_20200126_cleaned.csv")
df.head()
df["Date last updated"]=pd.to_datetime(df["Date last updated"])
df.dtypes
df["Province/State"]=df["Province/State"].astype(str)

df["Country"]=df["Country"].astype(str)
df.dtypes
df=df.drop(["Unnamed: 0"],axis=1)
df.dtypes
df.head()
import seaborn as sns

import matplotlib.pyplot as plt 

plt.figure(figsize=(45,20))

chart=sns.countplot("Province/State",data=df,palette='Set1')

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

chart.axes.set_title("province wise count",fontsize=25)

chart.set_xlabel("Province name",fontsize=45)

chart.set_ylabel("count",fontsize=30)

chart.tick_params(labelsize=30)
df["Province/State"].value_counts()


for i,j in list(df[df["Province/State"]=="0"].loc[:,"Province/State"].items()):

    df.iloc[i,0]="hubei-wuhan"

df[df["Province/State"]=="0"]
df1.dtypes
list(df1.items())

df1.values

df1.index
df1.dtypes
df[df["Province/State"]=="0"]
df.dtypes
# country wise count of cases reported

import seaborn as sns

import matplotlib.pyplot as plt 

plt.figure(figsize=(45,20))

chart=sns.countplot("Country",data=df,palette='Set1')

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

chart.axes.set_title("Country wise count",fontsize=40)

chart.set_xlabel("Country name",fontsize=45)

chart.set_ylabel("count",fontsize=30)

chart.tick_params(labelsize=30)
df.head()
df['Day'] = df['Date last updated'].dt.day

df['Month'] = df['Date last updated'].dt.month

df['Year'] = df['Date last updated'].dt.year
df.head()
df["Day"]=df["Day"].astype(str)

df["Month"]=df["Month"].astype(str)

df["Year"]=df["Year"].astype(str)
# Day wise  cases reported

import seaborn as sns

import matplotlib.pyplot as plt 

plt.figure(figsize=(45,20))

chart=sns.countplot("Day",data=df,palette='Set1')

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

chart.axes.set_title("Day wise count",fontsize=40)

chart.set_xlabel("Day of the month",fontsize=45)

chart.set_ylabel("count",fontsize=30)

chart.tick_params(labelsize=30)
# Day wise Confirmed cases

import seaborn as sns

import matplotlib.pyplot as plt 

plt.figure(figsize=(60,40))

chart=sns.barplot(y="Confirmed",x="Day",data=df,palette='Set1')

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

chart.axes.set_title("Day wise count of suspected",fontsize=40)

chart.set_xlabel("Day of the month",fontsize=45)

chart.set_ylabel("count",fontsize=30)

chart.set_ylim(0,100)

chart.tick_params(labelsize=30)

df.dtypes
# Day wise cases found as suspected

import seaborn as sns

import matplotlib.pyplot as plt 

plt.figure(figsize=(60,40))

chart=sns.barplot(y="Suspected",x="Day",data=df,palette='Set1')

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

chart.axes.set_title("Day wise count of suspected",fontsize=40)

chart.set_xlabel("Day of the month",fontsize=45)

chart.set_ylabel("count",fontsize=30)

chart.set_ylim(0,100)

chart.tick_params(labelsize=30)
# Cases recovered

import matplotlib.pyplot as plt 

plt.figure(figsize=(60,40))

chart=sns.barplot(y="Recovered",x="Day",data=df,palette='Set1')

chart.set_xticklabels(chart.get_xticklabels(), rotation=45)

chart.axes.set_title("Day wise recovered cases",fontsize=40)

chart.set_xlabel("Day of the month",fontsize=45)

chart.set_ylabel("count",fontsize=30)

chart.set_ylim(0,20)

chart.tick_params(labelsize=30)
df.head()
df[df["Deaths"]>0]
df.corr()
cp=pd.get_dummies(df['Province/State'])

df=pd.concat([df,cp],axis=1)

df=df.drop("Province/State",axis=1)
df.head()
cp=pd.get_dummies(df['Country'])

df=pd.concat([df,cp],axis=1)

df=df.drop("Country",axis=1)
df=df.drop("Date last updated",axis=1)
df.head()
df["Day"]=df["Day"].astype(int)

df["Month"]=df["Month"].astype(int)

df["Year"]=df["Year"].astype(int)
df.head()
df.dtypes
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import accuracy_score

from  sklearn.decomposition import PCA

from sklearn.metrics import r2_score

from sklearn.metrics import mean_squared_error
x=df.drop(["Deaths"])

y=df["Deaths"]
x=df.drop(["Deaths"],axis=1)

y=df["Deaths"]
df.dtypes
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=0)
print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
y_train=y_train.to_numpy()

y_test=y_test.to_numpy()

y_train=y_train.reshape(257,1)

y_test=y_test.reshape(111,1)
scalerX = StandardScaler().fit(X_train)

scalery = StandardScaler().fit(y_train)



x_train = scalerX.transform(X_train)

y_train = scalery.transform(y_train)

x_test = scalerX.transform(X_test)

y_test = scalery.transform(y_test)
y_train=y_train.reshape(257)

y_test=y_test.reshape(111)
scaler = StandardScaler()

data_rescaled_X_train = scaler.fit_transform(x)
pca = PCA().fit(data_rescaled_X_train)

#Plotting the Cumulative Summation of the Explained Variance

plt.figure()

plt.plot(np.cumsum(pca.explained_variance_ratio_))

plt.xlabel('Number of Components')

plt.ylabel('Variance (%)') #for each component

plt.title('Corona Dataset Explained Variance')

plt.show()
df_pca=PCA(n_components=50)

pctrain = df_pca.fit_transform(X_train)

pctest = df_pca.transform(X_test)
linreg = LinearRegression()

linreg.fit(pctrain, y_train)

y_pred = linreg.predict(pctest)

print("R2 score",r2_score(y_test,y_pred))

print("RMSE",np.sqrt(mean_squared_error(y_test,y_pred)))

print("MSE",mean_squared_error(y_test,y_pred))
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 1000, random_state = 1)

rf.fit(pctrain, y_train)

y_pred = rf.predict(pctest)

print("R2 score",r2_score(y_test,y_pred))

print("RMSE",np.sqrt(mean_squared_error(y_test,y_pred)))

print("MSE",mean_squared_error(y_test,y_pred))
# some data is not acting in favour to gradboost

from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(loss='ls', learning_rate=0.02, n_estimators=300, subsample=1.0)

gbr.fit(pctrain, y_train)

y_pred = gbr.predict(pctest)

print("R2 score",r2_score(y_test,y_pred))

print("RMSE",np.sqrt(mean_squared_error(y_test,y_pred)))

print("MSE",mean_squared_error(y_test,y_pred))
# same for Adaboost

from  sklearn.ensemble import AdaBoostRegressor

abr=AdaBoostRegressor(base_estimator=None, n_estimators=300, learning_rate=0.01, loss='linear', random_state=1)

abr.fit(pctrain, y_train)

y_pred = abr.predict(pctest)

print("R2 score",r2_score(y_test,y_pred))

print("RMSE",np.sqrt(mean_squared_error(y_test,y_pred)))

print("MSE",mean_squared_error(y_test,y_pred))