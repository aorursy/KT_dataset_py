import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

x=np.array([6.5,6.8,7.0,8.0,8.5,9.8,10.5,5.5,5.0])

x_=np.mean(x)

y=np.array([180,195,210,235,280,350,400,170,160])

y_=np.mean(y)
plt.scatter(x,y)
slope=np.sum((x-x_)*(y-y_))/np.sum((x-x_)**2)
slope
np.cov(x,y,ddof=1)/np.var(x,ddof=1)
intercept=y_-slope*(x_)

intercept
ypred=intercept+slope*(x)
ypred
y
plt.plot(y,ypred,"*")
np.sum((y-ypred))
from sklearn.linear_model import LinearRegression
X=pd.DataFrame(x)
#inbuiltr mmodel approach

model=LinearRegression()

model.fit(X,y)

model.predict(X)
ypred #  formula approach
model.coef_
model.intercept_
plt.plot(x,y,'*')

plt.plot(x,ypred,'o-')
X1=pd.DataFrame([7.5])
model.predict(X1)
import statsmodels.formula.api as smf
A=pd.DataFrame({'bmi':[6.5,6.8,7.0,8.0,8.5,9.8,10.5,5.5,5.0],'glu':[180,195,210,235,280,350,400,170,160]})
statmoddel=smf.ols('glu~bmi',A).fit()
yypred=statmoddel.predict(A['bmi'])
statmoddel.params
statmoddel.summary()
plt.plot(y,ypred,'*')
cor=np.corrcoef(y,ypred)

cor
cor*cor
sse=sum((y-ypred)**2)

sse
var=sum((y-np.mean(y))**2)

var
1-(sse/var)
Data=pd.read_csv('../input/car-mpg (1).csv')
Data.describe().T
Data.head()
Data.info()
Data['hp'].unique()
temp=pd.DataFrame(Data.hp.str.isdigit())
temp[temp['hp']==False]
Data.hp.replace('?',np.NaN,inplace=True)
Data.hp.unique()
Data[Data.isnull().any(axis=1)]
Data.hp=Data.hp.fillna(Data['hp'].median())
Data.hp.dtype
Data.hp=Data.hp.astype('float64')
Data.hp.dtype
Data.info()
Data=Data.drop('car_name',axis=1)
Data.corr()
sns.pairplot(Data,diag_kind='kde')
Data.columns
import statsmodels.formula.api as smf

m1=smf.ols('mpg~cyl+disp+hp+wt+acc+yr+origin+car_type',Data).fit()
m1.summary()
Data=Data.drop('acc',axis=1)
m1=smf.ols('mpg~cyl+disp+hp+wt+yr+origin+car_type',Data).fit()
m1.summary()
x=Data.drop('mpg',axis=1)

y=Data[['mpg']]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.3,random_state=2)
from sklearn.linear_model import LinearRegression
LR=LinearRegression()

LR.fit(X_train,Y_train)

mpg_pred=LR.predict(X_test)
LR.score(X_test,Y_test)# adjusted R2
from sklearn import metrics
rmse=np.sqrt(metrics.mean_squared_error(Y_test,mpg_pred))

rmse
LR.score(X_test,Y_test)
plt.plot(Y_test,mpg_pred,'*')