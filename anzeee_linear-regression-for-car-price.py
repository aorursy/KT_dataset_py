

import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.preprocessing import StandardScaler 

from sklearn.linear_model  import LinearRegression,Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV

from sklearn.model_selection import train_test_split



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

df=pd.read_csv('/kaggle/input/vehicle-dataset-from-cardekho/car data.csv')

df.head()
df.shape
df.info()
df['Owner'].unique()
df['Number_of_Year']=2020-df['Year'] #age of the car
df.head()
sns.pairplot(df)
sns.regplot('Year','Selling_Price',data=df)
sns.regplot('Number_of_Year','Selling_Price',data=df)
ax = sns.barplot(x="Seller_Type", y="Selling_Price", data=df)
ax = sns.barplot(x="Fuel_Type", y="Selling_Price", data=df)
ax = sns.barplot(x="Transmission", y="Selling_Price", data=df)
ax = sns.barplot(x="Owner", y="Selling_Price", data=df)
sns.regplot('Selling_Price','Kms_Driven',data=df)
df.columns
final=df[['Selling_Price', 'Present_Price', 'Kms_Driven',

       'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner', 'Number_of_Year']]

final_df=pd.get_dummies(final,drop_first=True)
final_df.head()
plt.figure(figsize=(10,10))

ax = sns.heatmap(final_df.corr(),annot=True)
y=df.Selling_Price

x=final_df.drop(['Selling_Price'],axis=1)
x.head()
y.head()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
reg=LinearRegression()

reg.fit(X_train,y_train)
print(reg.intercept_)

print(reg.coef_)
from sklearn.metrics import mean_squared_error

mean_squared_error(y_true=y_train, y_pred=reg.predict(X_train))
from sklearn.metrics import r2_score

r2_score(y_true=y_train, y_pred=reg.predict(X_train))
lasscv = LassoCV(alphas = None,cv =10, max_iter = 100000, normalize = True)

lasscv.fit(X_train, y_train)
alpha = lasscv.alpha_

alpha
lasso_reg = Lasso(alpha)

lasso_reg.fit(X_train, y_train)
lasso_reg.score(X_test, y_test)
alphas = np.random.uniform(low=0, high=10, size=(50,))

ridgecv = RidgeCV(alphas = alphas,cv=10,normalize = True)

ridgecv.fit(X_train, y_train)
ridgecv.alpha_
ridge_model = Ridge(alpha=ridgecv.alpha_)

ridge_model.fit(X_train, y_train)
ridge_model.score(X_test, y_test)
elasticCV = ElasticNetCV(alphas = None, cv =10)



elasticCV.fit(X_train, y_train)
elasticCV.alpha_
elasticCV.l1_ratio
elasticnet_reg = ElasticNet(alpha = elasticCV.alpha_,l1_ratio=0.5)

elasticnet_reg.fit(X_train, y_train)
elasticnet_reg.score(X_test, y_test)