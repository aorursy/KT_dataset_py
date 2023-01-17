#importing all the libraries which are needed

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import seaborn as sns

col_names = ['mpg','cylinders','displacement','horsepower','weight','acceleration','model year','origin','car name']
df = pd.read_table("../input/auto-mpg.data", header=None, delim_whitespace=True)
df.columns = col_names
df.head()
print(df.shape)
print(df.dtypes)
df['horsepower'].isnull().sum()
df['horsepower'].unique()
df = df[df['horsepower'] != '?']
print(df.shape)
df['horsepower'] = df['horsepower'].astype('float')
print(df.dtypes)
df.isnull().sum()
df.describe()
df['mpg'].describe()
sns.distplot(df['mpg'])
sns.boxplot(x='model year', y='mpg', data=df)
g = sns.FacetGrid(data=df, col='cylinders')
g.map(plt.scatter, 'mpg','acceleration').add_legend()
sns.distplot(df['acceleration'])
sns.distplot(df['cylinders'])
sns.distplot(df['displacement'])
sns.distplot(df['weight'])
sns.distplot(df['horsepower'])
clist = col_names[1:8]
print("variables:", clist)
cdict = {}

for cname in clist:
    cdict[cname] = np.float(np.corrcoef(df['mpg'], df[cname])[0,1])
print("\n", cdict)
corrmatrix = df.corr()
sns.heatmap(corrmatrix, square=True)
from sklearn.metrics import mean_squared_error

features_list_m1 = ['cylinders', 'displacement', 'horsepower','weight', 'acceleration', 'model year', 'origin']
X1 = df[features_list_m1]
y1 = df['mpg']
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.33, random_state=24)
regr1 = linear_model.LinearRegression()
regr1.fit(X1_train, y1_train)
predicted_values_m1 = regr1.predict(X1_test)
residuals = y1_test - predicted_values_m1
print(r2_score(y1_test, predicted_values_m1))
r_squared = r2_score(y1_test, predicted_values_m1)
adjusted_r = 1 - (1 - r_squared) * (len(y1_test) - 1) / (len(y1_test) - len(regr1.coef_) - 1)

print(adjusted_r)
plt.scatter(residuals, predicted_values_m1)
plt.xlabel("Residuals")
plt.ylabel("Predictions")
plt.title("Residuals v/s Predictions")
sns.distplot(residuals)
m1_rmse = np.sqrt(mean_squared_error(y1_test, predicted_values_m1))
print("RMSE: ",m1_rmse)
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(random_state=12)
forest_reg.fit(X1_train, y1_train)
print("Training accuracy: ", forest_reg.score(X1_train, y1_train))

y_pred = forest_reg.predict(X1_test)

print("Testing accuracy: ", forest_reg.score(X1_test, y1_test))

forest_rmse = np.sqrt(mean_squared_error(y_pred, y1_test))
print("RMSE: ",forest_rmse)
