

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

data=pd.read_csv('../input/insurance/insurance.csv')
data.isnull().sum()
data.dtypes
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from sklearn.model_selection import train_test_split,cross_val_score
plt.boxplot(data=data,x='age')
plt.boxplot(data=data,x='bmi')
plt.boxplot(data=data,x='children')
plt.boxplot(data=data,x='charges')
sns.pairplot(data=data,hue='smoker')
sns.pairplot(data=data,hue='sex')
sns.pairplot(data=data,hue='region')
from scipy import stats
from scipy.stats import skew,norm
sns.distplot(data['charges'],fit=norm)
plt.xlabel('charges')
plt.ylabel('frequency')
print('skewness:',data['charges'].skew())

fig=plt.figure()
res=stats.probplot(data['charges'],plot=plt)
plt.show()
data['charges']=np.log1p(data['charges'])
sns.distplot(data['charges'],fit=norm)
plt.xlabel('charges')
plt.ylabel('frequency')
print('skewness:',data['charges'].skew())

fig=plt.figure()
res=stats.probplot(data['charges'],plot=plt)
plt.show()
data.head()
data['smoker'].replace({'yes':1,'no':0},inplace=True)

data=pd.get_dummies(data)
data.head()
data.drop('sex_female',axis=1,inplace=True)
corr_data=data.corr()
sns.heatmap(corr_data)

y=data['charges']
data.drop('charges',axis=1,inplace=True)
data.shape
x_train,x_test,y_train,true_p=train_test_split(data,y,test_size=0.3,random_state=101)
print(x_train.shape,y_train.shape,x_test.shape,true_p.shape)
true_p.head()
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.metrics import r2_score,mean_squared_error,make_scorer
scorer=make_scorer(mean_squared_error,greater_is_better=False)
def rmse_cv_train(model):
    rmse=np.sqrt(-cross_val_score(model,x_train,y_train,scoring=scorer,cv=5))
    return rmse
def rmse_cv_test(model):
    rmse=np.sqrt(-cross_val_score(model,x_test,true_p,scoring=scorer,cv=5))
    return rmse
lreg=LinearRegression()
lreg.fit(x_train,y_train)
print('rmse of train data:',rmse_cv_train(lreg).mean())
print('rmse of test data:',rmse_cv_test(lreg).mean())
pred=lreg.predict(x_test)
score=r2_score(pred,true_p)
score
final_data=pd.DataFrame()
final_data['true value'],final_data['predicted']=[true_p.values,pred]
final_data.head()
ridge=Ridge()
ridge.fit(x_train,y_train)
print('rmse of train:',rmse_cv_train(ridge).mean())
print('rmse of test:',rmse_cv_test(ridge).mean())
pred=lreg.predict(x_test)
score=r2_score(pred,true_p)
score
lasso=Lasso()
lasso.fit(x_train,y_train)
print('rmse of train:',rmse_cv_train(lasso).mean())
print('rmse of test:',rmse_cv_test(lasso).mean())
pred=lreg.predict(x_test)
score=r2_score(pred,true_p)
score