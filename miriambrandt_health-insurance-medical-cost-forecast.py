import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv("../input/insurance.csv")
df.shape
df.info()
df["charges"].describe()
sns.distplot(df["charges"], bins=20, fit=norm)
plt.show()
df["age"].describe()
sns.distplot(df["age"], bins=10, fit=norm)
plt.show()
sns.regplot(x="age", y="charges", data=df);
plt.show()
df["bmi"].describe()
sns.distplot(df["bmi"], bins=10, fit=norm)
plt.show()
sns.regplot(x="bmi", y="charges", data=df);
plt.show()
df["children"].value_counts()
sns.countplot(x="children", data=df);
plt.show()
sns.boxplot(x="children", y="charges", data=df);
plt.show()
corr = df.corr()
fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, vmax=.8, square=True);
plt.show()
print(corr.loc["charges"].sort_values(ascending=False).drop("charges"))
df["sex"].value_counts()
sns.countplot(x="sex", data=df);
plt.show()
sns.boxplot(x="sex", y="charges", data=df);
plt.show()
df["smoker"].value_counts()
sns.countplot(x="smoker", data=df);
plt.show()
sns.boxplot(x="smoker", y="charges", data=df);
plt.show()
df["region"].value_counts()
sns.countplot(x="region", data=df);
plt.show()
sns.boxplot(x="region", y="charges", data=df);
plt.show()
df["children"] = df["children"].astype("str")
df = pd.get_dummies(df, drop_first=True)
df.shape
df.info()
X = df.drop("charges", axis=1)
X = X.values
y = df["charges"]
y = y.values
linreg = LinearRegression()
dtr = DecisionTreeRegressor()
rfr = RandomForestRegressor()
score_lin = round(np.mean(cross_val_score(linreg, X, y, cv=5)),3)
print(score_lin)
param_grid_dtr = {'max_depth': [2,3,4], 'max_leaf_nodes': [13,14,15]}

grid_dtr = GridSearchCV(dtr, param_grid_dtr, cv=5)

grid_dtr.fit(X,y)

print(grid_dtr.best_params_)
score_dtr = round(np.mean(cross_val_score(grid_dtr, X, y, cv=5)),3)
print(score_dtr)
param_grid_rfr = {'n_estimators': [16,17,18,19], 'max_depth': [2,3,4], 'max_leaf_nodes': [19,20,21,22,23]}


grid_rfr = GridSearchCV(rfr, param_grid_rfr, cv=5)

grid_rfr.fit(X,y)

print(grid_rfr.best_params_)
score_rfr = round(np.mean(cross_val_score(grid_rfr, X, y, cv=5)),3)
print(score_rfr)
print("Linear Regression:" + str(score_lin))
print("Decision Tree Regressor:" + str(score_dtr))
print("Random Forest Regressor:" + str(score_rfr))