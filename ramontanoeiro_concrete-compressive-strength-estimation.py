# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import warnings

warnings.filterwarnings("ignore")



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
df = pd.read_csv("../input/concrete-dataset/Concrete_Data.csv")
df.head()
df.columns
df_new = df.rename(columns={'Cement (component 1)(kg in a m^3 mixture)':'Cement',

       'Blast Furnace Slag (component 2)(kg in a m^3 mixture)':'BFS',

       'Fly Ash (component 3)(kg in a m^3 mixture)':'Fly_Ash',

       'Water  (component 4)(kg in a m^3 mixture)':'Water',

       'Superplasticizer (component 5)(kg in a m^3 mixture)':'Superplasticizer',

       'Coarse Aggregate  (component 6)(kg in a m^3 mixture)':'Coarser_agg',

       'Fine Aggregate (component 7)(kg in a m^3 mixture)':'Fine_agg',

       'Age (day)':'Days',

       'Concrete compressive strength(MPa. megapascals)':'Comp_str'})
df_new.head()
df_new.columns
df_new.describe()
df_new.isnull().sum()
import matplotlib.pyplot as plt

import seaborn as sns
mask = np.zeros_like(df_new.corr())

mask[np.triu_indices_from(mask)] = True



f, ab = plt.subplots(figsize=(15,10))

sns.heatmap(df_new.corr(), annot=True, mask=mask)

sns.regplot(x='Cement',y='Comp_str', data=df_new)
sns.jointplot(x='BFS',y='Comp_str', kind='kde',data=df_new)
sns.jointplot(x='Fly_Ash',y='Comp_str',  kind='kde', data=df_new)
sns.regplot(x='Water',y='Comp_str', data=df_new)
sns.jointplot(x='Superplasticizer',y='Comp_str',kind='kde',data=df_new)
sns.regplot(x='Coarser_agg',y='Comp_str', data=df_new)
sns.regplot(x='Fine_agg',y='Comp_str', data=df_new)
sns.jointplot(x='Days',y='Comp_str',  kind='kde', data=df_new)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split, cross_val_score
features = ['Cement', 'BFS', 'Fly_Ash', 'Water', 'Superplasticizer', 'Coarser_agg',

       'Fine_agg', 'Days']

targets = ['Comp_str']



LR = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(df_new[features], df_new[targets], test_size=0.20, random_state=42)
model = LR.fit(X_train, y_train)

y_pred = model.predict(X_test)
from sklearn.metrics import mean_squared_error



plt.figure(figsize=(15,10))

x = np.linspace(0,80,1000)

y=x

plt.scatter(y_test, y_pred)

plt.plot(x,y)



print("The R^2 for the test data in this Linear Regression is: ", model.score(X_test,y_test))



print("The R^2 for the training data in this Linear Regression is: ", model.score(X_train,y_train))



RMSE = np.sqrt(mean_squared_error(y_test,y_pred))



print("The Root Mean Squared Error for this is: ",RMSE )

from sklearn import tree, svm

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV
params_SVR = [{'C':[1,10, 100, 250, 500, 750, 1000], 'max_iter':[3000, 4000, 5000, 6000]}]

params_DTR = [{'max_depth':[5,6,7,8]}]

params_RFR = [{'n_estimators':[200, 250, 300, 350, 400, 450, 500, 550, 600]}]



SVR = svm.LinearSVR()

DTR = tree.DecisionTreeRegressor()

RFR = RandomForestRegressor()
grid_SVR = GridSearchCV(SVR, params_SVR, cv=3, scoring='r2')

grid_DTR = GridSearchCV(DTR, params_DTR, cv=3, scoring='r2')

grid_RFR = GridSearchCV(RFR, params_RFR, cv=3, scoring='r2')
model_SVR = grid_SVR.fit(X_train, y_train.values.ravel())

model_DTR = grid_DTR.fit(X_train, y_train)

model_RFR = grid_RFR.fit(X_train, y_train)
print(grid_SVR.best_params_)

print(grid_DTR.best_params_)

print(grid_RFR.best_params_)
SVR_1 = svm.SVR(C=100, max_iter=400)

model_SVR1 = SVR_1.fit(X_train,y_train)

y_SVR1 = model_SVR1.predict(X_test)



DTR_1 = tree.DecisionTreeRegressor(max_depth=8)

model_DTR1 = DTR_1.fit(X_train,y_train)

y_DTR1 = model_DTR1.predict(X_test)



RFR_1 = RandomForestRegressor(n_estimators=300, criterion='mse')

model_RFR_1 = RFR_1.fit(X_train,y_train)

y_RFR1 = model_RFR_1.predict(X_test)
from sklearn.metrics import mean_squared_error



plt.figure(figsize=(15,10))

x = np.linspace(0,80,1000)

y=x

plt.scatter(y_test, y_SVR1)

plt.plot(x,y)



print("The R^2 for the test data in this SVR Regression is: ", model_SVR1.score(X_test,y_test))



print("The R^2 for the training data in this SVR Regression is: ", model_SVR1.score(X_train,y_train))



RMSE = np.sqrt(mean_squared_error(y_test,y_SVR1))



print("The Root Mean Squared Error for this is: ",RMSE )
from sklearn.metrics import mean_squared_error



plt.figure(figsize=(15,10))

x = np.linspace(0,80,1000)

y=x

plt.scatter(y_test, y_DTR1)

plt.plot(x,y)



print("The R^2 for the test data in this Decision Tree Regression is: ", model_DTR1.score(X_test,y_test))



print("The R^2 for the training data in this Decision Tree Regression is: ", model_DTR1.score(X_train,y_train))



RMSE = np.sqrt(mean_squared_error(y_test,y_DTR1))



print("The Root Mean Squared Error for this is: ",RMSE )

from sklearn.metrics import mean_squared_error



plt.figure(figsize=(15,10))

x = np.linspace(0,80,1000)

y=x

plt.scatter(y_test, y_RFR1)

plt.plot(x,y)



print("The R^2 for the test data in this Random Forest Regression is: ", model_RFR_1.score(X_test,y_test))



print("The R^2 for the training data in this Random Forest Regression is: ", model_RFR_1.score(X_train,y_train))



RMSE = np.sqrt(mean_squared_error(y_test,y_RFR1))



print("The Root Mean Squared Error for this is: ",RMSE )