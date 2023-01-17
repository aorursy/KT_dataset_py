# Libraries

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
df = pd.read_csv("../input/insurance.csv")
print (df.head())

print (df.info())
df.shape
sns.kdeplot(df[df['sex']=='female']['charges'], shade=True, label = 'Female charge')

sns.kdeplot(df[df['sex']=='male']['charges'], shade=True, label = 'Male charge')

sns.swarmplot(x='sex', y='charges', data=df)
#The impact of smoke on charges



df.groupby("smoker")['charges'].agg('mean').plot.bar()
sns.scatterplot(x='bmi', y='charges',hue='smoker',data=df)
sns.regplot(x='bmi',y='charges',data=df)
sns.lmplot(x='bmi',y='charges',hue='sex',data=df)
sns.scatterplot(x='age', y='charges', hue='sex',data=df)
sns.lineplot(x='children', y='charges',  estimator=np.median, data=df);
#sns.lineplot(x='children', y='charges', data=df);

#sns.scatterplot(x='children', y='charges', data=df)

df.groupby('children')['charges'].agg('median')
df.info()
df_dummies = pd.get_dummies(df)

df_dummies.head()
from sklearn.model_selection import train_test_split
X = df_dummies.drop('charges', axis= 1)

y = df_dummies.charges



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

lm.fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error



mean_absolute_error(y_test, lm.predict(X_test))
avg_charges = pd.Series([y_test.mean()]* y_test.shape[0])

avg_charges

mean_absolute_error(y_test, avg_charges)
from sklearn.linear_model import Lasso,Ridge, ElasticNet
ridge = Ridge()

ridge.fit(X_train, y_train)

mean_absolute_error(y_test, ridge.predict(X_test))
lasso = Lasso(alpha=0.1)

lasso.fit(X_train, y_train)

mean_absolute_error(y_test, lasso.predict(X_test))
elasticnet = ElasticNet()

elasticnet.fit(X_train,y_train)

mean_absolute_error(y_test, elasticnet.predict(X_test))
from sklearn.model_selection import GridSearchCV
params = {'alpha': [0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]}



params_elastic ={'alpha': [0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1],

                'l1_ratio': [0.1, 0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]}
# Ridge

ridge_cv = GridSearchCV(ridge, params, scoring = 'neg_mean_absolute_error')

ridge_cv.fit(X_train,y_train)



# Lasso

lasso_cv = GridSearchCV(lasso, params, scoring = 'neg_mean_absolute_error')

lasso_cv.fit(X_train,y_train)



# Elastic Net

elasticnet_cv = GridSearchCV(elasticnet, params_elastic, scoring = 'neg_mean_absolute_error')

elasticnet_cv.fit(X_train,y_train)
print (ridge_cv.best_params_)

print (lasso_cv.best_params_)

print (elasticnet_cv.best_params_)

print (mean_absolute_error(y_test, ridge_cv.predict(X_test)))

print (mean_absolute_error(y_test, lasso_cv.predict(X_test)))

print (mean_absolute_error(y_test, elasticnet_cv.predict(X_test)))