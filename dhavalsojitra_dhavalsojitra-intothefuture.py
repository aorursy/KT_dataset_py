import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
train = pd.read_csv('../input/into-the-future/train.csv',parse_dates=True,index_col='time')
test = pd.read_csv('../input/into-the-future/test.csv',parse_dates=True,index_col='time')
train=train.drop('id',axis=1)
print(train.isnull().sum())
print(train.head())
sns.heatmap(train.corr(),annot=True)
X=train['feature_1']
y=train['feature_2']
X=pd.DataFrame(X)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
lr=LinearRegression()
lr.fit(X_train,y_train)

y_pred_lr=lr.predict(X_test)
print(f'MSE ={mean_squared_error(y_pred_lr,y_test)}')
print(f'MAE ={mean_absolute_error(y_pred_lr,y_test)}')
print(f'RMSE ={np.sqrt(mean_squared_error(y_pred_lr,y_test))}')
dt=DecisionTreeRegressor(min_samples_split=10)
dt.fit(X_train,y_train)

y_pred_dt=dt.predict(X_test)
print(f'MSE ={mean_squared_error(y_pred_dt,y_test)}')
print(f'MAE ={mean_absolute_error(y_pred_dt,y_test)}')
print(f'RMSE ={np.sqrt(mean_squared_error(y_pred_dt,y_test))}')
rf=RandomForestRegressor(n_estimators=25)
rf.fit(X_train,y_train)

y_pred_rf=rf.predict(X_test)
print(f'MSE ={mean_squared_error(y_pred_rf,y_test)}')
print(f'MAE ={mean_absolute_error(y_pred_rf,y_test)}')
print(f'RMSE ={np.sqrt(mean_squared_error(y_pred_rf,y_test))}')
test=test.drop('id',axis=1)
print(test.isnull().sum())
print(test.head())
test_X=pd.DataFrame(test['feature_1'])
temp=[]
for i in range(0,len(test_X)):
    y_pred1=rf.predict(test_X.iloc[i:i+1])
    temp2=float(y_pred1)
    temp.append(temp2)
data={'id':range(564,939),'feature_2':temp}
submission=pd.DataFrame(data)
submission.to_csv(r'E:\Python,ML,DL,NLP\Currently Working\Into future time series\output.csv',index=False)


