
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

        
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv('/kaggle/input/car-price-prediction/CarPrice_Assignment.csv')
df.head()
df.columns
df.shape
df.info()
df.describe()
df.nunique()
df['enginetype'].unique()
df['cylindernumber'].unique()
df['enginetype'].value_counts()
df['cylindernumber']=df['cylindernumber'].map({'four':4,'six':6,'five':5,'eight':8,'two':2,'three':3,'twelve':12})
df['cylindernumber'].dtype
df['enginetype']=df['enginetype'].map({'ohc':1,'ohcf':2,'ohcv':3,'dohc':4,'l':5,'rotar':6,'dohcv':7})
sns.boxplot(data=df,x='cylindernumber',y='price')
sns.boxplot(data=df,x='enginelocation',y='price')
sns.boxplot(data=df,x='carbody',y='price')
plt.scatter(x='compressionratio',y='price',data=df)
df.head(1)
sns.boxplot(data=df,x='symboling',y='price')
sns.scatterplot(x='price',y='curbweight',hue='doornumber',data=df)
plt.rcParams['figure.figsize']=(12,12)
corr=df.corr()
sns.heatmap(corr,fmt='.2f',annot=True,cmap=plt.cm.Blues)
df_corr=df.corr().abs()
df_corr


upper=df_corr.where(np.triu(np.ones(df_corr.shape),k=1).astype(np.bool))
upper

to_drop=[column for column in upper.columns if any (upper[column]>0.95)]
print('----------------------------')
print(to_drop)


df1=df.drop(to_drop,axis=1)
df1.columns
df1.drop('car_ID',axis=1,inplace=True)
df1.info()
df1.nunique()
dummies=pd.get_dummies(df1[['fueltype','aspiration','doornumber','carbody','drivewheel',
                            'enginelocation','fuelsystem']])
dummies.head(2)
df1=pd.concat([df1,dummies],axis=1)
df1.drop(['fueltype','aspiration','doornumber','carbody','drivewheel','enginelocation','fuelsystem'],
         axis=1,inplace=True)
#checking for null values in enginetype feature
df1[df1['enginetype'].isnull()]
#filling NaN values with most common enginetype which is 1
df1['enginetype']=df1['enginetype'].fillna('1')
df1[df1['enginetype'].isnull()]
df1['enginetype']=df1['enginetype'].astype('int')
df1.info()
X=df1.loc[:,df1.columns!='price']
y=df1.loc[:,'price']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1234)
X_train.drop('CarName',axis=1,inplace=True)
X_test.drop('CarName',axis=1,inplace=True)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

lr=LinearRegression()
lr.fit(X_train,y_train)
lr.score(X_train,y_train)
lr_pred=lr.predict(X_test)
print('MSE:',mean_squared_error(lr_pred,y_test))
print('MAE:',mean_absolute_error(lr_pred,y_test))
print('r2_score:',r2_score(lr_pred,y_test))
prediction=pd.DataFrame({'Actual':y_test,'Predicted':lr_pred})
prediction.head(10)
lr.coef_
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
svc=SVR()
svc.fit(X_train,y_train)
svc.score(X_train,y_train)
svc_pred=svc.predict(X_test)
print('MSE:',mean_squared_error(svc_pred,y_test))
print('MAE:',mean_absolute_error(svc_pred,y_test))
print('r2_score:',r2_score(svc_pred,y_test))
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
svr=SVR()
svr.fit(X_train_scaled,y_train)
svr.score(X_train_scaled,y_train)
svr_pred=svr.predict(X_test_scaled)

print('MSE:',mean_squared_error(svr_pred,y_test))
print('MAE:',mean_absolute_error(svr_pred,y_test))
print('r2_score:',r2_score(svr_pred,y_test))
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
rf.fit(X_train,y_train)
rf.score(X_train,y_train)
rf_pred=rf.predict(X_test)
print('MSE:',mean_squared_error(rf_pred,y_test))
print('MAE:',mean_absolute_error(rf_pred,y_test))
print('r2_score:',r2_score(rf_pred,y_test))
prediction_rf=pd.DataFrame({'Actual':y_test,'Predicted':rf_pred})
prediction_rf.head(10)
from sklearn.ensemble import BaggingRegressor
bag=BaggingRegressor()
bag.fit(X_train,y_train)
bag.score(X_train,y_train)
bag_pred=bag.predict(X_test)
print('MSE:',mean_squared_error(bag_pred,y_test))
print('MAE:',mean_absolute_error(bag_pred,y_test))
print('r2_score:',r2_score(bag_pred,y_test))
#Support vector Regressor performs bad and it is less generally used in regression problems
#Linear regression and random forest gives good prediction accuracy on the data
