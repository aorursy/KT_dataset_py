import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
df = pd.read_csv("/kaggle/input/co2-emission-by-vehicles/CO2 Emissions_Canada.csv")
df.head()
df.info()
df.describe(include='all')
sns.distplot(df['CO2 Emissions(g/km)'])
corr = df.corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr,annot=True)
plt.figure(figsize=(25,5))
df.groupby(['Fuel Consumption City (L/100 km)'])['CO2 Emissions(g/km)'].mean().sort_values().plot(kind='bar')
plt.xticks(rotation=90, horizontalalignment='center', fontweight='light', fontsize='7')
plt.ylabel("CO2 Emissions(g/km)")
plt.figure(figsize=(20,5))
df.groupby(['Fuel Consumption Comb (mpg)'])['CO2 Emissions(g/km)'].mean().sort_values().plot(kind='bar')
plt.xticks(rotation=90, horizontalalignment='center', fontweight='light', fontsize='7')
plt.ylabel("CO2 Emissions(g/km)")
Ft = pd.get_dummies(df['Fuel Type'],drop_first=True,prefix='Fuel')
df = df.drop(['Fuel Type'],axis=1)
df = pd.concat([df,Ft],axis=1)
Tr = pd.get_dummies(df['Transmission'],drop_first=True)
df = df.drop(['Transmission'],axis=1)
df = pd.concat([df,Tr],axis=1)
df.head()
X = df.drop(['CO2 Emissions(g/km)','Fuel Consumption Comb (mpg)'],axis=1)
y = df['CO2 Emissions(g/km)']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.5,random_state=42)
cat_cols = ['Make','Model','Vehicle Class']
import category_encoders as ce
target_enc = ce.CatBoostEncoder(cols = cat_cols)
target_enc.fit(x_train[cat_cols],y_train)

train_enc = target_enc.transform(x_train[cat_cols])
test_enc = target_enc.transform(x_test[cat_cols])

x_train = x_train.drop(['Make','Model','Vehicle Class'],axis=1)
x_test = x_test.drop(['Make','Model','Vehicle Class'],axis=1)
x_train = pd.concat([x_train,train_enc],axis=1)
x_test = pd.concat([x_test,test_enc],axis=1)

x_train.head()
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
model = lr.fit(x_train,y_train)
y_pred = model.predict(x_test) 
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)
y_test
y_pred
from sklearn.metrics import mean_squared_error,mean_absolute_error
mae = mean_absolute_error(y_test,y_pred)
print('Absolute error:',mae)
mre = mean_squared_error(y_test,y_pred)
print('Squared error',mre)
#Linear
fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(121)
plt.scatter(y_test, y_pred, alpha=0.1, color = 'g')
ax2 = fig.add_subplot(122)
sns.distplot(y_test-y_pred)
#Ridge regression with multiple features by using the SCIKIT-LEARN library
from sklearn.linear_model import Ridge
ridge = Ridge()
model2 = ridge.fit(x_train,y_train)
y_pred2 = model2.predict(x_test)
r2_score(y_test,y_pred2)
y_test

y_pred2
mse = mean_squared_error(y_test,y_pred2)
mse
mae = mean_absolute_error(y_test,y_pred2)
mae
#Ridge
fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(121)
plt.scatter(y_test, y_pred2, alpha=0.1, color = 'g')
ax2 = fig.add_subplot(122)
sns.distplot(y_test-y_pred2)
from sklearn.linear_model import Lasso
lasso = Lasso()
model3=lasso.fit(x_train, y_train)
y_pred3 = model3.predict(x_test)
r2_score(y_test,y_pred3)
y_pred3
y_test
mse = mean_squared_error(y_test,y_pred3)
mse
mae = mean_absolute_error(y_test,y_pred3)
mae
#Lasso
fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(121)
plt.scatter(y_test, y_pred3, alpha=0.1, color = 'g')
ax2 = fig.add_subplot(122)
sns.distplot(y_test-y_pred3)
from sklearn.linear_model import ElasticNet
elasticNet= ElasticNet()
model4=elasticNet.fit(x_train, y_train)
y_pred4 = model4.predict(x_test)
r2_score(y_test,y_pred4)
y_pred4
y_test
mse = mean_squared_error(y_test,y_pred4)
mse
mae = mean_absolute_error(y_test,y_pred4)
mae
#ElasticNet
fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(121)
plt.scatter(y_test, y_pred4, alpha=0.1, color = 'g')
ax2 = fig.add_subplot(122)
sns.distplot(y_test-y_pred4)
from sklearn.ensemble import VotingRegressor
votingReg=VotingRegressor([('LinearRegression',lr),('lasso',lasso),('Ridge',ridge),('ElasticNet',elasticNet)]).fit(x_train, y_train)
y_pred = votingReg.predict(x_test)
y_pred
y_test
mse = mean_squared_error(y_test, y_pred)
mse

mae = mean_absolute_error(y_test, y_pred)
mae
fig = plt.figure(figsize=(16,8))
ax1 = fig.add_subplot(121)
plt.scatter(y_test, y_pred, alpha=0.1, color = 'g')
ax2 = fig.add_subplot(122)
sns.distplot(y_test-y_pred)