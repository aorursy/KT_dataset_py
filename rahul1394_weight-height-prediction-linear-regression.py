import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,StandardScaler

from sklearn.metrics import r2_score,mean_squared_error
weight_height_dataset = pd.read_csv('../input/weight-height/weight-height.csv')

weight_height_dataset.head()
weight_height_dataset.info()
weight_height_dataset.describe()
weight_height_dataset.duplicated().sum()
weight_height_dataset.isnull().sum()
sns.boxplot(weight_height_dataset.Weight)

plt.show()
sns.boxplot(weight_height_dataset.Height)

plt.show()
q1 = weight_height_dataset['Weight'].quantile(0.25)

q3 = weight_height_dataset['Weight'].quantile(0.75)

iqr = q3 - q1

ul = q3 + 1.5*iqr

ll = q1 - 1.5*iqr

weight_height_dataset = weight_height_dataset[(weight_height_dataset.Weight >= ll) & (weight_height_dataset.Weight <= ul)]
q1 = weight_height_dataset['Height'].quantile(0.25)

q3 = weight_height_dataset['Height'].quantile(0.75)

iqr = q3 - q1

ul = q3 + 1.5*iqr

ll = q1 - 1.5*iqr

weight_height_dataset = weight_height_dataset[(weight_height_dataset.Height >= ll) & (weight_height_dataset.Height <= ul)]
sns.scatterplot(weight_height_dataset.Weight,weight_height_dataset.Height,color='g')

plt.show()
from sklearn.model_selection import train_test_split
x = pd.DataFrame(weight_height_dataset['Weight'])

y = pd.DataFrame(weight_height_dataset['Height'])
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.30,random_state=123)

print(xtrain.shape,ytrain.shape,xtest.shape,ytest.shape)
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(xtrain,ytrain)

yPredict = lr.predict(xtest)
print(lr.coef_)

print(lr.intercept_)
r2_score(ytest,yPredict)
np.sqrt(mean_squared_error(ytest,yPredict))
sns.scatterplot(xtrain.Weight,ytrain.Height)

plt.plot(xtrain.Weight,lr.predict(xtrain),c='r')

plt.show()
sns.scatterplot(xtest.Weight,ytest.Height,color='r')

plt.plot(xtest.Weight,yPredict,c='b')

plt.show()
residual = ytest - yPredict
sns.residplot(yPredict,residual)

plt.show()
import pylab

import scipy.stats as stats
stats.probplot(residual.Height,plot=pylab)

plt.show()
test,pvalue = stats.shapiro(residual)

print(pvalue)
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = [variance_inflation_factor(weight_height_dataset.drop('Gender',axis=1).values,i) for i in range(weight_height_dataset.drop('Gender',axis=1).shape[1])]
pd.DataFrame({'vif':vif},index=['Weight','Height']).T
from statsmodels.stats.api import het_goldfeldquandt
df = pd.DataFrame(weight_height_dataset['Height'])
residual2 = df - lr.predict(df)
ftest,pvalue,result = het_goldfeldquandt(residual2,weight_height_dataset.drop('Gender',axis=1))

print(pvalue)
from statsmodels.stats.stattools import durbin_watson
print(durbin_watson(residual))
import statsmodels.api as sms
model = sms.OLS(y,x).fit()

model.summary()
test,pvalue = sms.stats.diagnostic.linear_rainbow(model)

pvalue
weight_height_dataset[['Female','Male']] = pd.get_dummies(weight_height_dataset['Gender'])

weight_height_dataset.head()
weight_height_dataset.drop('Gender',axis=1,inplace=True)
weight_height_dataset.head()
temp = pd.DataFrame(StandardScaler().fit_transform(weight_height_dataset),columns=weight_height_dataset.columns)

temp.head()
x = temp.drop('Height',axis=1)

y = temp['Height']
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.30,random_state=123)

print(xtrain.shape,ytrain.shape,xtest.shape,ytest.shape)
lr = LinearRegression()

lr.fit(xtrain,ytrain)

yPredict = lr.predict(xtest)
r2_score(ytest,yPredict)
np.sqrt(mean_squared_error(ytest,yPredict))