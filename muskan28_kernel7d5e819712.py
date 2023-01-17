# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from pandas import Series, DataFrame

data=pd.read_csv("/kaggle/input/OnlineNewsPopularity.csv")

df=pd.DataFrame(data)

df.head()
df.shape
all_columns=list(df.columns)

categorical_cols = all_columns[13:19] + all_columns[31:39]

categorical_cols
df.describe()
df[' shares'].max()
df.hist(figsize=(20,20))

plt.show()
figsize=plt.rcParams["figure.figsize"]

figsize[0]=10

figsize[1]=10

plt.hist(x=df[' shares'])
u=df[' shares'].median()

s=df[' shares'].std()
df=df[df[' shares']<(u+2*s) ]

df.shape
df.isna().sum()
for i in df.columns:

    c=0

    for j in df.index:

        if(df[i][j]==0):

            c=c+1

    if(c!=0):

        print(i,c)
for i in df.index:

    if(df[' n_tokens_content'][i]==0):

        df=df.drop(i,axis=0)   
df.shape
df.var().sort_values()
corrmat=df.corr()

df.corr()[' shares'].sort_values(ascending=False)
reduced_column= df.corr()[' shares'].nsmallest(35)

reduced_column
df.shape
model=RandomForestRegressor(random_state=42)

dx=df.drop(['url',' timedelta',' shares'],axis=1);

dy=df[' shares']

model.fit(dx,dy)
figsize=plt.rcParams['figure.figsize']

figsize[0]=10

figsize[1]=10
feature=pd.Series(model.feature_importances_,index=dx.columns)

columns=df.columns

feature.nlargest(30).plot(kind='barh')
plt.barh(range(len(feature)),feature)

plt.yticks(range(len(feature)),columns)
X=df.drop(['url',' timedelta',' shares'],axis=1)

for i in X.columns:

    if( corrmat[' shares'][i]<= 0.00689):

        X=X.drop(i,axis=1)

print(X.shape)

Y=df[' shares']

x_train,x_cv,y_train,y_cv= train_test_split(X,Y,test_size=0.2,random_state=42)

some_x_cv=x_cv.iloc[:500]

some_y_cv=y_cv.iloc[:500]

some_x_data=x_train.iloc[:500]

some_y_data=y_train.iloc[:500]
lreg=LinearRegression()

lreg.fit(x_train,y_train)

lreg.coef_
lreg.intercept_
pred=lreg.predict(some_x_data)

model1_test= pd.DataFrame({'Actual': some_y_data.values.flatten(),'Predicted': pred.flatten()})

model1_test
lin_mse = mean_squared_error(some_y_data,pred)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
lin_mae=mean_absolute_error(some_y_data,pred)

lin_mae
a,b=plt.subplots()

sns.regplot(x=some_y_data,y=pred)
pred=lreg.predict(some_x_cv)

model1_test= pd.DataFrame({'Actual': some_y_cv.values.flatten(),'Predicted': pred.flatten()})

model1_test
lin_mse = mean_squared_error(some_y_cv,pred)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
lin_mae=mean_absolute_error(some_y_cv,pred)

lin_mae
a,b=plt.subplots()

sns.regplot(x=some_y_cv,y=pred)
model2= DecisionTreeRegressor(random_state=42)

model2.fit(x_train,y_train)
model2_pred=model2.predict(some_x_data)
model2_check= pd.DataFrame({'Actual': some_y_data.values.flatten(),'Predicted': model2_pred.flatten()})

model2_check
model2_mse = mean_squared_error(some_y_data,model2_pred)

model2_rmse = np.sqrt(model2_mse)

model2_rmse
a,b=plt.subplots(figsize=(17,4))

sns.regplot(x=some_y_data,y=model2_pred)
model2_test_pred=model2.predict(some_x_cv)
model2_test_check= pd.DataFrame({'Actual': some_y_cv.values.flatten(),'Predicted': model2_test_pred.flatten()})

model2_test_check
model2_test_mse = mean_squared_error(some_y_cv,model2_test_pred)

model2_test_rmse = np.sqrt(model2_test_mse)

model2_test_rmse
a,b=plt.subplots(figsize=(17,4))

sns.regplot(x=some_y_cv,y=model2_pred)
model3= RandomForestRegressor(random_state=42)

model3.fit(x_train,y_train)
model3_pred=model3.predict(some_x_data)

model3_check= pd.DataFrame({'Actual': some_y_data.values.flatten(),'Predicted': model3_pred.flatten()})

model3_check
rf_mse = mean_squared_error(some_y_data,model3_pred)

rf_rmse = np.sqrt(rf_mse)

rf_rmse
a,b=plt.subplots(figsize=(17,4))

sns.regplot(x=some_y_data,y=model3_pred)
model3_pred=model3.predict(some_x_cv)

model3_check= pd.DataFrame({'Actual': some_y_cv.values.flatten(),'Predicted': model3_pred.flatten()})

model3_check
rf_mse = mean_squared_error(some_y_cv,model3_pred)

rf_rmse = np.sqrt(rf_mse)

rf_rmse
a,b=plt.subplots(figsize=(17,4))

sns.regplot(x=some_y_cv,y=model3_pred)
from sklearn.metrics import r2_score



r2_score( some_y_cv, model3_pred)
score_tree=cross_val_score(model2,x_train,y_train,scoring="neg_mean_squared_error",cv=10)

tree_rmse=np.sqrt(-score_tree)

tree_rmse
tree_rmse.mean()
score_linear=cross_val_score(lreg,x_train,y_train,scoring="neg_mean_squared_error",cv=20)

linear_rmse=np.sqrt(-score_tree)

linear_rmse
linear_rmse.mean()
score_linear=cross_val_score(model3,x_train,y_train,scoring="neg_mean_squared_error",cv=20)

rfd_rmse=np.sqrt(-score_tree)

rfd_rmse
rfd_rmse.mean()