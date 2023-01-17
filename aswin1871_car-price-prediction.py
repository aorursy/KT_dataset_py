import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('../input/car-price/cars.csv')

df.head()
df.describe(include='all')
df = df.drop(['Model'],axis=1)
df.describe(include='all')
df.isnull().sum()
df = df.dropna(axis=0)
df.describe(include='all')
sns.distplot(df['Price'])
df.describe(include='all')
q = df['Price'].quantile(0.99)
df = df[df['Price']<q]

df.describe(include='all')
sns.distplot(df['Price'])
sns.distplot(df['Mileage'])
q = df['Mileage'].quantile(0.99)

df = df[df['Mileage']<q]

df.describe(include='all')
sns.distplot(df['Mileage'])
sns.distplot(df['Year'])
q = df['Year'].quantile(0.1)

df = df[df['Year']>q]

df.describe(include='all')
sns.distplot(df['Year'])
df[['EngineV']].max()
df[df['EngineV']>6.5].count()
df = df[df['EngineV']<6.5]
df.describe(include='all')
df
df = df.reset_index(drop=True)
df
df.describe(include='all')
f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True,figsize=(15,3))

ax1.scatter(df['Mileage'],df['Price'])

ax1.set_title('Milage vs Price')



ax2.scatter(df['EngineV'],df['Price'])

ax2.set_title('Engine Volume vs Price')



ax3.scatter(df['Year'],df['Price'])

ax3.set_title('Year vs Price')



log_price = np.log(df['Price'])

df['log_price'] = log_price
f,(ax1,ax2,ax3) = plt.subplots(1,3,sharey=True,figsize=(15,3))

ax1.scatter(df['Mileage'],df['log_price'])

ax1.set_title('Milage vs Log Price')



ax2.scatter(df['EngineV'],df['log_price'])

ax2.set_title('Engine Volume vs Log Price')



ax3.scatter(df['Year'],df['log_price'])

ax3.set_title('Year vs Log Price')



from statsmodels.stats.outliers_influence import variance_inflation_factor
variables = df[['Mileage','EngineV','Year']]



vif = pd.DataFrame()



variables.shape
vif['VIF'] = [variance_inflation_factor(variables.values,i) for i in range(variables.shape[1])]
vif['Features'] = variables.columns
vif
df = df.drop(['Year'],axis=1)
df
df_dummies = pd.get_dummies(df,drop_first=True)
df_dummies
df_dummies.columns
variables = df_dummies[[x for x in df_dummies.columns if x not in ['Price']]]



vif = pd.DataFrame()



variables.shape
vif['VIF'] = [variance_inflation_factor(variables.values,i) for i in range(variables.shape[1])]
vif['Features'] = variables.columns
vif
df = df_dummies
df
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
target = df[['log_price']]

inputs = df.drop(['log_price','Price'],axis=1)
inputs
# Scaling the data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(inputs)
scaled_input = scaler.transform(inputs)
# Train-test split

from sklearn.model_selection import train_test_split
input_train,input_test,target_train,target_test = train_test_split(scaled_input,target,test_size=0.2,random_state=365)
# Fitting a regression



reg.fit(input_train,target_train)
reg.score(input_train,target_train)
yhat = reg.predict(input_train)
plt.scatter(target_train,yhat)
sns.distplot(target_train-yhat)
y_hat_test = reg.predict(input_test)
plt.scatter(y_hat_test,target_test,alpha=0.2)

fig = plt.plot([7.5,8,11,12],[7.5,8,11,12],lw=4,c='orange')

plt.show()
# Performance Dataframe



df_pf = pd.DataFrame(np.exp(y_hat_test),columns=['Predicted'])
df_pf['Target'] = np.exp(target_test)
df_pf
target_test = target_test.reset_index(drop=True)

target_test
df_pf['Target'] = np.exp(target_test)
df_pf
df_pf['Residual'] = df_pf['Target'] - df_pf['Predicted']
df_pf
df_pf['Difference%'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf
df_pf.describe()