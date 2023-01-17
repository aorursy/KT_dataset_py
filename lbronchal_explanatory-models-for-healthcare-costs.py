import warnings
import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
data = pd.read_csv('../input/insurance.csv')
data.head()
data.isnull().any()
data.describe()
data.info()
data['sex'] = pd.Categorical(data['sex'])
data['smoker'] = pd.Categorical(data['smoker'])
data['region'] = pd.Categorical(data['region'])
data.info()
plt.figure(figsize=(18, 6))
plt.subplot(131)
sns.distplot(data['age']).set_title("Age")
plt.subplot(132)
sns.distplot(data['bmi']).set_title("Bmi")
plt.subplot(133)
sns.distplot(data['charges']).set_title("Charges")
plt.show()
corr = data.corr()
sns.heatmap(corr, annot=True, linewidths=.5, fmt= '.3f', cmap="YlGnBu")
plt.show()
sns.pairplot(data, kind="reg")
plt.show()
plt.figure(figsize=(18, 6))
plt.subplot(131)
sns.boxplot(x='sex', y='charges', data=data)
plt.subplot(132)
sns.boxplot(x='region', y='charges', data=data)
plt.subplot(133)
sns.boxplot(x='smoker', y='charges', data=data)
plt.show()
data.groupby('sex')['charges'].mean()
data.groupby('smoker')['charges'].mean()
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler
y = data['charges']
X = data.drop('charges', axis=1)
X = pd.get_dummies(X, drop_first=True, prefix = ['sex', 'smoker', 'region'])

scaler = MinMaxScaler()
X[['age', 'bmi', 'children']] = scaler.fit_transform(X[['age', 'bmi', 'children']])
np.random.seed(1)
X_2 = sm.add_constant(X)
model_lr = sm.OLS(y, X_2)
linear = model_lr.fit()
print(linear.summary())
from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state = 100)
y = data['charges']
X = data.drop('charges', axis=1)
X['sex'] = X['sex'].cat.codes
X['smoker'] = X['smoker'].cat.codes
X['region'] = X['region'].cat.codes
model.fit(X, y)
sns.barplot(x=X.columns, 
            y=model.feature_importances_, 
            order=X.columns[np.argsort(model.feature_importances_)[::-1]])
plt.show()
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics.regression import mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
model_rf = RandomForestRegressor(n_estimators=200, random_state=1)
scores = cross_val_score(model_rf, X_train, y_train, cv=5, scoring="neg_mean_squared_error")
scores = np.sqrt(-scores)
print("validation RMSE: {:0.4f} (+/- {:0.4f})".format(np.mean(scores), np.std(scores)))
model_rf.fit(X_train, y_train)
y_pred = model_rf.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
rmse
y_train_pred = model_rf.predict(X_train)
sns.regplot(x=y_train, y=y_train_pred)
plt.title("Predicted vs Real")
plt.show()
importances = model_rf.feature_importances_
indices = np.argsort(importances)[::-1]

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices])
plt.xlim([-1, X.shape[1]])
plt.show()