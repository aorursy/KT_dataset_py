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
df=pd.read_csv('../input/vehicle-dataset-from-cardekho/car data.csv')
df.head()
df.info()
df.isnull().sum()
for i in df.columns:
    if type(df[i][1])==str:
        print(i)
        print(df[i].nunique())
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.pairplot(df)
sns.scatterplot(df['Kms_Driven'],df['Selling_Price'],)
#SP is greater for cars with minimum kms_driven
sns.scatterplot(df['Present_Price'],df['Selling_Price'])
fig,ax=plt.subplots(1,2)

ax[0].bar(df['Fuel_Type'],df['Selling_Price'])
ax[0].set_title('SP vs fueltype')
ax[1].bar(df['Fuel_Type'],df['Present_Price'])
ax[1].set_title('PP vs fueltype')
fig,ax=plt.subplots(1,2)
p=df[df['Fuel_Type']=='Petrol'][['Selling_Price','Present_Price']].mean()
ax[0].bar(p.index,p.values)
ax[0].set_title('pertol sp Vs pp')

d=df[df['Fuel_Type']=='Diesel'][['Selling_Price','Present_Price']].mean()
ax[1].bar(d.index,d.values)
ax[1].set_title('diesel sp Vs pp')

#PP for Petrol car is less compared to Diesel
#When we look at SP vs pp mean values for petrol the valuation btw them is less ,whereas for diesel vehicles you wont be able to sell them at good price compared to petrol

df['Car_Name'].value_counts()[df['Car_Name'].value_counts()>5]
df[df['Car_Name']=='city'].groupby('Year')[['Selling_Price','Present_Price']].mean().plot(kind='bar')
df.groupby('Year')[['Selling_Price','Present_Price']].mean().plot(kind='bar')
df1=df.copy()
df1['No_of_years']=2020-df1['Year']
fig,ax=plt.subplots(1,2,figsize=(20,4))

d1=df1[df1['Fuel_Type']=='Diesel'].groupby(['No_of_years'])[['Selling_Price','Present_Price']].mean().sort_values(by='No_of_years')
p1=df1[df1['Fuel_Type']=='Petrol'].groupby(['No_of_years'])[['Selling_Price','Present_Price']].mean().sort_values(by='No_of_years')
ax[0].plot(d1.index,d1.values)
ax[0].set_title('Diesel sp,pp vs no_of years')

ax[1].plot(p1.index,p1.values)
ax[1].set_title('Petrol sp,pp vs no_of years')
fig.legend(['Selling','Present'])

#definitely no_of years plays a part, as no _of _years increases the SP value decreases
sns.barplot(df['Owner'],df['Selling_Price'])
dummy=pd.get_dummies(df1[['Fuel_Type', 'Seller_Type', 'Transmission']],drop_first=True)
df1=df1.join(dummy)
dummy=pd.get_dummies(df['Owner'],drop_first=True)
df1=df1.join(dummy)
df1.drop(['Fuel_Type', 'Seller_Type', 'Transmission', 'Owner'],axis=1,inplace=True)
plt.figure(figsize=(15,15))
sns.heatmap(df1.corr(),annot=True,cmap='RdYlGn')

df1.columns
X=df1.iloc[:,3:-2]
y=df1.iloc[:,2]
X
from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(X,y)
print(model.feature_importances_)
feat_importance=pd.Series(model.feature_importances_,index=X.columns)
feat_importance.sort_values().plot(kind='bar')
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
x_train.shape
rf=RandomForestRegressor()
rf.fit(x_train,y_train)
pred=rf.predict(x_test)
print(r2_score(pred,y_test))
sns.distplot(y_test-pred)
#with hyperparametre

from sklearn.model_selection import RandomizedSearchCV
n_estimators=[int(x) for x in np.linspace(100,1000,10)]
max_features=['auto','sqrt']
max_depth=[int(x) for x in np.linspace(2,6,4)]
min_samples_split=[3,6,9,15,20,100]
min_samples_leaf=[2,3,4,6,8]
rand_grid={'n_estimators':n_estimators,
           'max_features':max_features,
          'max_depth':max_depth,
          'min_samples_split':min_samples_split,
          'min_samples_leaf':min_samples_leaf}


rf=RandomForestRegressor()
rf_random=RandomizedSearchCV(estimator=rf,param_distributions=rand_grid,scoring='neg_mean_squared_error',n_iter=10,cv=3)
rf_random.fit(x_train,y_train)
pred=rf_random.predict(x_test)
rf_random.best_estimator_
r2_score(y_test,pred)
sns.scatterplot(y_test,pred)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
RF=RandomForestRegressor(max_depth=4, max_features='sqrt', min_samples_leaf=3,
                      min_samples_split=3, n_estimators=400)
rf_pip=Pipeline([('std_scalr',StandardScaler()),('rf',RF)])
rf_pip.fit(x_train,y_train)
print(r2_score(y_test,pred))
sns.scatterplot(y_test,pred)
from sklearn.linear_model import LinearRegression
pip_lr=Pipeline([('LReg',LinearRegression())])
pip_dtr=Pipeline([('DTC',DecisionTreeRegressor())])

def pippp(pip):
    pip.fit(x_train,y_train)
    pred=pip.predict(x_test)
    print(f'{pip}  {r2_score(y_test,pred)}')
    y_test.reset_index()['Selling_Price'].plot(label='original',figsize=(8,5))
    plt.plot(pred,label='pred')
    plt.legend()
    
pippp(pip_lr)    

pippp(pip_dtr)
x_test.head()
#just checking 

a=np.array([1.78,6000,4,0,1,1,1])
a=a.reshape(1,-1)
print(rf_pip.predict(a))
print(y_test.values[0])


#close values.