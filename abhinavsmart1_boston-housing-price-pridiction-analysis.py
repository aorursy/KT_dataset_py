import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import scikitplot as skplt

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_val_score

from collections import Counter

from IPython.core.display import display, HTML

#sns.set_style('darkgrid')
from sklearn.datasets import load_boston

boston_dataset = load_boston()

housing = pd.DataFrame(boston_dataset.data, columns = boston_dataset.feature_names)
housing.head()
housing['MEDV'] = boston_dataset.target
housing.head()
housing.tail()
housing.info()
housing.isnull().any()
housing.isnull().sum()
housing.describe()
housing.hist(bins=40,figsize=(20,15))
X = housing.iloc[:, 0:13].values

y = housing.iloc[:, 13].values.reshape(-1,1)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
print(f"Shape of X_train: {X_train.shape}\nShape of X_test: {X_test.shape}\nShape of y_train: {y_train.shape}\nShape of y_test {y_test.shape}\n")
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

split.get_n_splits(X, y)

for train_index,test_index in split.split(housing,housing['CHAS']):

    print("TRAIN:", train_index, "\nTEST:", test_index)

   

    strat_X_train = housing.loc[train_index]

    strat_X_test = housing.loc[test_index]
# A Category in Train Data

strat_X_train['CHAS'].value_counts()
# A Category in Train Data

strat_X_test['CHAS'].value_counts()
# Ratio of different values assigned in a category in train data 

(376/28)
# Ratio of different values assigned in a category in test data 

95/7
strat_X_train.describe()
strat_X_train.info()
#Looking for correlation



corr_matrix = housing.corr()

#Plot figsize

corr_matrix['MEDV'].sort_values(ascending=False)
corr = housing.corr()

#Plot figsize

fig, ax = plt.subplots(figsize=(18, 18))

#Generate Heat Map, allow annotations and place floats in map

sns.heatmap(corr, cmap='RdBu', annot=True, fmt=".3f")

#Apply xticks

plt.xticks(range(len(corr.columns)), corr.columns);

#Apply yticks

plt.yticks(range(len(corr.columns)), corr.columns)

#show plot

plt.show()
sns.pairplot(housing)

plt.show()
from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

my_pipeline= Pipeline([('imputer', SimpleImputer(strategy="median")),('std_scalar',StandardScaler())])
housing_tr= my_pipeline.fit_transform(housing)

housing_tr
from sklearn.linear_model import LinearRegression

regression_linear = LinearRegression()

regression_linear.fit(X_train, y_train)
from sklearn.metrics import r2_score



# Predicting Cross Validation Score the Test set results

cv_l = cross_val_score(estimator = regression_linear, X = X_train, y = y_train, cv = 10)



# Predicting R2 Score the Train set results

y_train_predict_l = regression_linear.predict(X_train)

r2_score_train_l = r2_score(y_train, y_train_predict_l)



# Predicting R2 Score the Test set results

y_test_predict_l = regression_linear.predict(X_test)

r2_score_test_l = r2_score(y_test, y_test_predict_l)



# Predicting RMSE the Test set results

rmse_l = (np.sqrt(mean_squared_error(y_test, y_test_predict_l)))

print("CV_mean: ", cv_l.mean())

print("CV_Std: ", cv_l.std())

print('R2_score (train): ', r2_score_train_l)

print('R2_score (test): ', r2_score_test_l)

print("RMSE: ", rmse_l)
skplt.estimators.plot_learning_curve(regression_linear, X, y,cv=10)

plt.show()
# Fitting the Decision Tree Regression Model to the dataset

from sklearn.tree import DecisionTreeRegressor

regression_dt = DecisionTreeRegressor(random_state = 0)

regression_dt.fit(X_train, y_train)
from sklearn.metrics import r2_score



# For Cross Validation Score

cv_dt = cross_val_score(estimator = regression_dt, X = X_train, y = y_train, cv = 10)



# For R2 Score the Train set results

y_train_predict_dt = regression_dt.predict(X_train)

r2_score_train_dt = r2_score(y_train, y_train_predict_dt)



# For R2 Score the Test set results

y_test_predict_dt = regression_dt.predict(X_test)

r2_score_test_dt = r2_score(y_test, y_test_predict_dt)



# For RMSE the Test set results

rmse_dt = (np.sqrt(mean_squared_error(y_test, y_test_predict_dt)))

print('CV_mean: ', cv_dt.mean())

print('CV_Std: ', cv_dt.std())

print('R2_score (train): ', r2_score_train_dt)

print('R2_score (test): ', r2_score_test_dt)

print("RMSE: ", rmse_dt)
skplt.estimators.plot_learning_curve(regression_dt, X, y ,cv=10)

plt.show()
# Fitting the Random Forest Regression to the dataset

from sklearn.ensemble import RandomForestRegressor

regression_rf = RandomForestRegressor(n_estimators = 500, random_state = 0)

regression_rf.fit(X_train, y_train.ravel())
from sklearn.metrics import r2_score



# For Cross Validation Score

cv_rf = cross_val_score(estimator = regression_rf, X = X_train, y = y_train.ravel(), cv = 10)



# For R2 Score the Train set results

y_train_predict_rf = regression_rf.predict(X_train)

r2_score_train_rf = r2_score(y_train, y_train_predict_rf)



# For R2 Score the Test set results

y_test_predict_rf = regression_rf.predict(X_test)

r2_score_test_rf = r2_score(y_test, y_test_predict_rf)



# For RMSE the Test set results

rmse_rf = (np.sqrt(mean_squared_error(y_test, y_test_predict_rf)))

print('CV_mean: ', cv_rf.mean())

print('CV_Std: ', cv_rf.std())

print('R2_score (train): ', r2_score_train_rf)

print('R2_score (test): ', r2_score_test_rf)

print("RMSE: ", rmse_rf)
skplt.estimators.plot_learning_curve(regression_rf, X, y.ravel(), cv=10)

plt.show()
models = [('Linear Regression', rmse_l, r2_score_train_l, r2_score_test_l, cv_l.mean(),cv_l.std()),

          ('Decision Tree Regression', rmse_dt, r2_score_train_dt, r2_score_test_dt, cv_dt.mean(),cv_dt.std()),

          ('Random Forest Regression', rmse_rf, r2_score_train_rf, r2_score_test_rf, cv_rf.mean(),cv_rf.std())   

         ]
predict = pd.DataFrame(data = models, columns=['Model', 'RMSE', 'R2_Score(train)', 'R2_Score(test)', ' Cross-Validation_mean',' Cross-Validation_Std'],)

predict
fig, axes = plt.subplots(2,1, figsize=(18,12))



predict.sort_values(by=['R2_Score(train)'], ascending=False, inplace=True)



sns.barplot(x='R2_Score(train)', y='Model', data = predict, palette='muted', ax = axes[0])

axes[0].set_xlabel('R2 Score (Train)', size=16)

axes[0].set_ylabel('Model')

axes[0].set_xlim(0,1.0)



predict.sort_values(by=['R2_Score(test)'], ascending=False, inplace=True)



sns.barplot(x='R2_Score(test)', y='Model', data = predict, palette='Blues_d', ax = axes[1])

axes[1].set_xlabel('R2 Score (Test)', size=16)

axes[1].set_ylabel('Model')

axes[1].set_xlim(0,1.0)



plt.show()
predict.sort_values(by=['RMSE'], ascending=False, inplace=True)

fig, axe = plt.subplots(1,1, figsize=(10,6))

sns.barplot(x='Model', y='RMSE', data=predict, ax = axe)

axe.set_xlabel('Model', size=16)

axe.set_ylabel('RMSE', size=16)



plt.show()
