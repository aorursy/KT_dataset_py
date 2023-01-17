import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))
data = pd.read_csv("../input/housing.csv")
data.head(6)
data.shape
data.info()
data.dtypes
#univariate analysis

data.describe(include='all')
#Histogram for all the features

data.hist(bins=20, figsize=(20,10), grid=False)
#Boxplot for all the variables

fig, ax= plt.subplots(nrows=2, ncols=4, figsize=(20,10))

sns.boxplot(y= data['housing_median_age'], ax=ax[0,0])

sns.boxplot(y= data['total_rooms'], ax= ax[0,1])

sns.boxplot(y= data['total_bedrooms'], ax= ax[0,2])

sns.boxplot(y= data['population'], ax= ax[0,3])

sns.boxplot(y= data['households'], ax= ax[1,0])

sns.boxplot(y= data['median_income'], ax= ax[1,1])
#Bivariate Analysis
sns.pairplot(data.head(200))
plt.figure(figsize=(6,5))

plt.scatter(data['longitude'], data['latitude'], alpha=0.3)
#shows the density of california

data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.3,label="population", figsize=(15,8),

          c="median_house_value", cmap=plt.get_cmap("jet"))

plt.legend()
plt.figure(figsize=(10,8))

sns.heatmap(data.corr(),annot=True)
data.corr()['median_house_value'].sort_values(ascending=False)
data.columns
X= data.iloc[:,[2,3,4,5,6,7,9]]

Y= data.iloc[:,8]
X.isna().sum()
from sklearn.preprocessing import Imputer

imputer= Imputer(missing_values=np.nan, strategy='median', axis=0)

imputer= imputer.fit(X[['total_bedrooms']])

X[['total_bedrooms']]= imputer.transform(X[['total_bedrooms']])
X.isna().sum()
Y.isna().sum()
X['ocean_proximity'].value_counts()
X= pd.get_dummies(data=X, columns=['ocean_proximity'], prefix='ocean_proximity')

X= X.drop('ocean_proximity_ISLAND', axis=1)

X.head()
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test= train_test_split(X,Y, test_size=0.3, random_state=0)
X_train.columns
#Lets try to fit the data taking all the variables

from sklearn.linear_model import LinearRegression

regressor1= LinearRegression()



print(regressor1.fit(X_train, Y_train))

print(regressor1.coef_)

print(X_train.columns)

print(regressor1.intercept_)



from sklearn import metrics

print(metrics.r2_score(Y_test, regressor1.predict(X_test)))

print(np.sqrt(metrics.mean_squared_error(Y_test, regressor1.predict(X_test))))
#Removing columns based on low coffecient values

X_train1= X_train.iloc[:,[0,4,5,6,7,8,9]]

X_test1= X_test.iloc[:,[0,4,5,6,7,8,9]]



regressor2= LinearRegression()

print(regressor2.fit(X_train1, Y_train))

print(regressor2.coef_)

print(X_train1.columns)

print(regressor2.intercept_)



from sklearn import metrics

print(metrics.r2_score(Y_test, regressor2.predict(X_test1)))

print(np.sqrt(metrics.mean_squared_error(Y_test, regressor2.predict(X_test1))))
import statsmodels.formula.api as sm

regressor1_ols= sm.OLS(endog=Y_train, exog=X_train1).fit()

regressor1_ols.summary()
#Building Decision Tree



from sklearn.tree import DecisionTreeRegressor

dtree= DecisionTreeRegressor(max_depth=6, random_state=0)

dtree.fit(X_train,Y_train)



print(dtree.feature_importances_)



from sklearn import metrics

print(metrics.r2_score(Y_test, dtree.predict(X_test)))

print(np.sqrt(metrics.mean_squared_error(Y_test, dtree.predict(X_test))))
#Using grid search CV for best parameters



from sklearn.model_selection import GridSearchCV

dt= DecisionTreeRegressor(random_state=5)

params= [{"max_depth":[4,8,12,16,20],"max_leaf_nodes":range(2,20)}]

grid= GridSearchCV(estimator=dt, param_grid=params, cv=3, refit=True)
grid.fit(X_train, Y_train)

grid_predictions= grid.predict(X_test)



print('Accuracy Score:')

print(metrics.r2_score(Y_test, grid_predictions))
print(grid.best_params_)

print(grid.best_score_)
from sklearn.externals.six import StringIO  

from IPython.display import Image  

from sklearn.tree import export_graphviz

import pydot
# Create DOT data

'''dot_data = StringIO()

export_graphviz(dtree, out_file=dot_data, class_names=['median_house_value'], filled=True, rounded=True)

#out_file is the location where we are displaying it and it should be string IO object

#Basically my dot_data have the output of the graph

#feature_names=['Level']



#Draw Graph

graph = pydot.graph_from_dot_data(dot_data.getvalue())



#Show Graph

Image(graph[0].create_png())'''
#Building Random Forest

from sklearn.ensemble import RandomForestRegressor

rf= RandomForestRegressor(random_state=10)

rf_params= [{'n_estimators':range(1,25)}]

rf_grid= GridSearchCV(estimator=rf, param_grid=rf_params, refit=True, cv=3)
rf_grid.fit(X_train, Y_train)
print(rf_grid.best_params_)

print(rf_grid.best_estimator_)

print(rf_grid.best_score_)
rtree= rf_grid.best_estimator_

rtree.fit(X_train, Y_train)



metrics.r2_score(Y_test, rtree.predict(X_test))