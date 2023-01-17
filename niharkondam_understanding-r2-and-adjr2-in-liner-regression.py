import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
x=np.array([6.5,6.8,7.0,8.0,8.5,9.8,10.5,5.5,5.0])

y=np.array([180,195,210,235,280,350,400,170,160])
sns.scatterplot(x,y)
b1=((x-np.mean(x))*(y-np.mean(y))).sum()/((x-np.mean(x))**2).sum()
b1==np.cov(x,y,ddof=1)[1,0]/np.var(x,ddof=1)
b0=np.mean(y)-(b1*np.mean(x))

b0
y_pred=b0+(b1*x)

y_pred
sum_of_residues= (y-y_pred).sum()

sum_of_residues
sse=((y-y_pred)**2).sum()

sse
mse=sse/len(x)

mse
sns.scatterplot(x,y)

sns.lineplot(x,y_pred,)
from sklearn.linear_model import LinearRegression
df=pd.DataFrame(x,columns=['x'])
lr=LinearRegression()

lr.fit(df,y)

ypred=lr.predict(df)
lr.coef_
lr.intercept_
ypred,y_pred
lr.score(df,y)
1-(((y-ypred)**2).sum()/((y-y.mean())**2).sum())
(np.corrcoef(y,ypred)[1,0])**2
from statsmodels.formula.api import ols
df['y']=y
df
model=ols('y~x',df).fit()
model.params
model.predict(df.x)
model.summary()
n,p=df.drop('y',1).shape

r2=1-(((y-ypred)**2).sum()/((y-y.mean())**2).sum())

adjusted_r2= 1 - ((1-r2) * (n-1) / (n-p-1))

adjusted_r2
df.shape
df=pd.read_csv('../input/learn-ml-datasets/car-mpg (1).csv')
df.head()
df.info()
df.describe()
[(i,v) for i,v in enumerate(df.hp) if not v.isdigit()]
df.shape
df.drop('car_name',1,inplace=True)

df.shape
df.replace('?',np.nan,inplace=True)
df.isnull().sum()
df.info()
df.fillna(df.hp.median(),inplace=True)
df.info()
df['hp']=df.hp.astype(int)
df.info()
plt.figure(figsize=(16,8))

sns.heatmap(df.corr(),annot=True,cmap='coolwarm',center=0)
sns.pairplot(df,diag_kind='kde')
from sklearn.model_selection import train_test_split
x=df.drop(['mpg'],1)

y=df.mpg
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=2)
lr=LinearRegression()

lr.fit(xtrain,ytrain)

ypred=lr.predict(xtest)

lr.score(xtest,ytest),lr.score(xtrain,ytrain)
from sklearn.metrics import r2_score

r2_score(ytest,ypred)
r2=1-(((ytest-ypred)**2).sum()/((ytest-np.mean(y))**2).sum())

n,p=x.shape

adj_r2=1-((1-r2)*(n-1)/(n-p-1))

r2,adj_r2
#correlation of ytest and ypred as cr

cr=((ytest-(np.mean(ytest)))*(ypred-(np.mean(ypred)))).sum()/(((((ytest-(np.mean(ytest)))**2).sum())**0.5)*(((ypred-(np.mean(ypred)))**2).sum())**0.5)

cr**2
mse=np.mean((ytest-ypred)**2)

mse
from sklearn.metrics import mean_squared_error

mean_squared_error(ytest,ypred)
rmse=mse**0.5

rmse
plt.figure(figsize=(16,8))

sns.scatterplot(ytest,ypred)

sns.scatterplot(ytest,ytest)
for i in df.columns:

  print(i,df[i].nunique())
for i in ['cyl','origin','car_type']:

  df[i]=df[i].apply(str)
df.info()
plt.figure(figsize=(16,8))

sns.heatmap(df.corr(),annot=True,cmap='coolwarm',center=0)
x=df.drop(['mpg'],1)

x=pd.get_dummies(x,drop_first=True)

y=df.mpg
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=2)
lr=LinearRegression()

lr.fit(xtrain,ytrain)

ypred=lr.predict(xtest)

lr.score(xtest,ytest),lr.score(xtrain,ytrain)
# We can see how the adjusted_r2 is being affected by increasing the number of features using dummification 

r2=1-(((ytest-ypred)**2).sum()/((ytest-np.mean(y))**2).sum())

n,p=x.shape

adj_r2=1-((1-r2)*(n-1)/(n-p-1))

r2,adj_r2
from statsmodels.stats.outliers_influence import variance_inflation_factor
for i in range(x.shape[1]):

  print(x.columns[i],variance_inflation_factor(x.values, i))
# we eliminate features to see the change in adj R2. It is our decision to choose inbetween higher r2 and no multicollinearity.

# Higher r2 means better results, no multicollienarity means stable relationship between y and x variables.

x=df.drop(['mpg','origin','car_type','acc'],1)

x=pd.get_dummies(x,drop_first=True)

y=df.mpg

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=2)

lr=LinearRegression()

lr.fit(xtrain,ytrain)

ypred=lr.predict(xtest)

r2=1-(((ytest-ypred)**2).sum()/((ytest-np.mean(y))**2).sum())

n,p=x.shape

adj_r2=1-((1-r2)*(n-1)/(n-p-1))

print(r2,adj_r2)

for i in range(x.shape[1]):

  print(x.columns[i],variance_inflation_factor(x.values, i))