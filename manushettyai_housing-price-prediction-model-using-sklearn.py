import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data_df_train = pd.read_csv('../input/train.csv')
data_df_train.columns
y = data_df_train['SalePrice']
y.describe()
sns.distplot(y, bins=40)
corr_map = data_df_train.corr()

plt.subplots(figsize=(15, 15))

sns.heatmap(corr_map)
data_df_train.corr().nlargest(10, 'SalePrice')['SalePrice'].index
cols = data_df_train.corr().nlargest(8, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(data_df_train[cols].values.T)

hm = sns.heatmap(cm, cbar=True, annot=True, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
x = data_df_train[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',

       'TotalBsmtSF', '1stFlrSF','FullBath', 'TotRmsAbvGrd']]
x.head()
x_min = x.min()

x_range = x.max()- x_min

x_scaled = x/x_range
plt.figure(figsize=(15,10))

sns.heatmap(x.isnull(), yticklabels = False, cbar = False, cmap="Blues")
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
from sklearn.linear_model import LinearRegression

classifier = LinearRegression()

classifier.fit(X_train, y_train)
y_predict_test = classifier.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_predict_test})

df1 = df.head(25)
df1.plot(kind='bar',figsize=(10,8))

plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')

plt.show()
from sklearn import metrics

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_predict_test))  

print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_predict_test))  

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_predict_test)))