import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
data=pd.read_csv("../input/avocado-prices/avocado.csv")
data.shape
data.head()
a=np.sum(data['type']=='conventional')

b=np.sum(data['type']=='organic')

list1=[a,b]

list2=['conventional','organic']

plt.pie(list1,labels=list2)

plt.title('Distribution of avacado type in datset')

plt.show()
data.info()
data.describe()
data.columns
new_data=data.drop('Unnamed: 0',axis=1)
new_data.isnull().sum()
new_data.type.unique()
new_data.region.nunique()
new_data['year'].unique()
new_data.columns
import seaborn as sns

a=new_data[['AveragePrice','Total Volume','Total Bags','year']]

sns.pairplot(a,height=2)
#The avg price most of the times is in between 0.9 to 1.3

new_data['AveragePrice'].head(900).value_counts().sort_index().plot.area()
new_data['AveragePrice'].min()
# very less correlation between year and price

import seaborn as sns

a=(new_data.loc[:,['year','AveragePrice']]).corr()

sns.heatmap(a,annot=True)
#No strong correlation between variables

import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize=(12,6))

sns.heatmap(new_data.corr(),cmap='coolwarm',annot=True)
#creating dummy variable for the type 

dummy=pd.get_dummies(new_data['type'],prefix='type')
dummy.head()

new_data1=pd.concat([new_data,dummy],axis=1)
new_data1.head()
new_data1.drop(['type','region','Date'],axis=1,inplace=True)
new_data1.head()
X=pd.DataFrame()

Y=pd.DataFrame()
type(Y)
X=new_data1.loc[:,new_data1.columns!='AveragePrice']

Y=new_data1['AveragePrice']
X.head()
Y.head()
type(Y)
from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=0,test_size=0.20)
# linear regression for predictiong the average prices

from sklearn.linear_model import LinearRegression

linreg=LinearRegression()

linreg.fit(X_train,Y_train)

Ypredict_train=linreg.predict(X_train)

Ypredict_test=linreg.predict(X_test)
#As we can see that we dont have a straigt line so I am not sure that this model will give better results or not,lets see

plt.scatter(x=Y_test,y=Ypredict_test)

plt.title("Actual and the predcited value chart")
import numpy as np

from sklearn.metrics import mean_absolute_error

MAE_train=mean_absolute_error(Ypredict_train,Y_train)

MAE_test=mean_absolute_error(Ypredict_test,Y_test)
print('MAE_train of Linear Regression',MAE_train)

print('MAE_test of Linear Regression' ,MAE_test)
from sklearn.metrics import mean_squared_error

MSE_train=mean_squared_error(Ypredict_train,Y_train)

MSE_test=mean_squared_error(Ypredict_test,Y_test)
print('MSE_train of Linear Regression',MSE_train)

print('MSE_test of Linear Regression',MSE_test)
RMSE_train=np.sqrt(mean_squared_error(Ypredict_train,Y_train))

RMSE_test=np.sqrt(mean_squared_error(Ypredict_test,Y_test))
print('RMSE_train of Linear Regression',RMSE_train)

print('RMSE_test of Linear Regression',RMSE_test)
#R square value for train data

ss_residual_train=sum((Y_train-Ypredict_train)**2)

ss_Total_train=sum((Y_train-np.mean(Y_train))**2)

R_squared_train=(1-(float(ss_residual_train))/ss_Total_train)

print('R_squared_train of Linear Regression',R_squared_train)

Adjusted_RSquared_train=1-(1-R_squared_train)*(len(Y_train)-1)/(len(Y_train)-(X_train.shape[1]-1))

print('Adjusted_RSquared_train of Linear Regression',Adjusted_RSquared_train)
#R square value for test data

ss_residual_test=sum((Y_test-Ypredict_test)**2)

ss_Total_test=sum((Y_test-np.mean(Y_test))**2)

R_squared_test=(1-(float(ss_residual_test))/ss_Total_test)

print('R_squared_test of Linear Regression',R_squared_test)

Adjusted_RSquared_test=1-(1-R_squared_test)*(len(Y_test)-1)/(len(Y_test)-(X_test.shape[1]-1))

print('Adjusted_RSquared_test of Linear Regression',Adjusted_RSquared_test)
#creating a dummy variable for region

dummies1=pd.get_dummies(data['region'],prefix='region')

dummies1.tail()
new_data2=pd.concat([data,dummies1,dummy],axis=1)
new_data2.head()
new_data2.drop(['Unnamed: 0','Date','type','region'],axis=1,inplace=True)
new_data2.head()
X1=new_data2.loc[:,new_data2.columns!='AveragePrice']

Y1=new_data2['AveragePrice']
X1_train,X1_test,Y1_train,Y1_test=train_test_split(X1,Y1,random_state=0,test_size=0.20)
linreg.fit(X1_train,Y1_train)

Ypredict_train1=linreg.predict(X1_train)

Ypredict_test1=linreg.predict(X1_test)
MAE_train1=mean_absolute_error(Ypredict_train1,Y1_train)

MAE_test1=mean_absolute_error(Ypredict_test1,Y1_test)
print('MAE_train1 of Linear Regression', MAE_train1) 

print('MAE_test1 of Linear Regression' ,MAE_test1)
MSE_train1=mean_squared_error(Ypredict_train1,Y1_train)

MSE_test1=mean_squared_error(Ypredict_test1,Y1_test)
print('MSE_train1 of Linear Regression', MSE_train1)

print('MSE_test1 of Linear Regression', MSE_test1)
RMSE_train1=np.sqrt(mean_squared_error(Ypredict_train1,Y1_train))

RMSE_test1=np.sqrt(mean_squared_error(Ypredict_test1,Y1_test))
print('RMSE_train1 of Linear Regression', RMSE_train1)

print('RMSE_test1 of Linear Regression', RMSE_test1)
ss_residual_train1=sum((Y1_train-Ypredict_train1)**2)

ss_Total_train1=sum((Y1_train-np.mean(Y1_train))**2)

R_squared_train1=(1-(float(ss_residual_train1))/ss_Total_train1)

print('R_squared_train1 of Linear Regression', R_squared_train1)

Adjusted_RSquared_train1=1-(1-R_squared_train1)*(len(Y1_train)-1)/(len(Y1_train)-(X1_train.shape[1]-1))

print('Adjusted_RSquared_train1 of Linear Regression', Adjusted_RSquared_train1)
ss_residual_test1=sum((Y1_test-Ypredict_test1)**2)

ss_Total_test1=sum((Y1_test-np.mean(Y1_test))**2)

R_squared_test1=(1-(float(ss_residual_test1))/ss_Total_test1)

print('R_squared_test1 of Linear Regression', R_squared_test1)

Adjusted_RSquared_test1=1-(1-R_squared_test1)*(len(Y1_test)-1)/(len(Y1_test)-(X1_test.shape[1]-1))

print('Adjusted_RSquared_test1 of Linear Regression', Adjusted_RSquared_test1)
from sklearn.tree import DecisionTreeRegressor

decision_regression=DecisionTreeRegressor()

decision_regression.fit(X_train,Y_train)

pred=decision_regression.predict(X_test)
#nearly straight line so this model will give better results

plt.scatter(x=Y_test,y=pred)

plt.xlabel('Actual values')

plt.ylabel('predicted values')

plt.title("Actual and the predicted values ")
from sklearn.metrics import mean_absolute_error,mean_squared_error

import numpy as np

print('MAE of DecisionTree of case1:',mean_absolute_error(Y_test, pred))

print('MSE of DecisionTree of case1 ', mean_squared_error(Y_test, pred))

print('RMSE of DecisionTree of case1:', np.sqrt(mean_squared_error(Y_test, pred)))
from sklearn.tree import DecisionTreeRegressor

decision_regression=DecisionTreeRegressor()

decision_regression.fit(X1_train,Y1_train)

pred=decision_regression.predict(X1_test)
from sklearn.metrics import mean_absolute_error,mean_squared_error

import numpy as np

print('MAE of DecisionTree of case2:',mean_absolute_error(Y1_test, pred))

print('MSE of DecisionTree of case2:', mean_squared_error(Y1_test, pred))

print('RMSE of DecisionTree of case2:', np.sqrt(mean_squared_error(Y1_test, pred)))
from sklearn.ensemble import RandomForestRegressor

random_forest = RandomForestRegressor()

random_forest.fit(X_train,Y_train)

pred=random_forest.predict(X_test)
from sklearn.metrics import mean_absolute_error,mean_squared_error

import numpy as np

print('MAE of RandomForest of case1:',mean_absolute_error(Y_test, pred))

print('MSE of RandomForest of case1:', mean_squared_error(Y_test, pred))

print('RMSE of RandomForest of case1:', np.sqrt(mean_squared_error(Y_test, pred)))
from sklearn.ensemble import RandomForestRegressor

random_forest = RandomForestRegressor()

random_forest.fit(X1_train,Y1_train)

pred=random_forest.predict(X1_test)
from sklearn.metrics import mean_absolute_error,mean_squared_error

import numpy as np

print('MAE of RandomForest of case2:',mean_absolute_error(Y_test, pred))

print('MSE of RandomForest of case2:', mean_squared_error(Y_test, pred))

print('RMSE of RandomForest of case2:', np.sqrt(mean_squared_error(Y_test, pred)))