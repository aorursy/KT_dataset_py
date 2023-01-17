import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
import seaborn as sns
import matplotlib.pyplot as plt
print(check_output(["ls", "../input"]).decode("utf8"))
df = pd.read_csv('../input/train.csv')
pd.set_option('display.max_columns', None)
df.head()
df.info()
df.describe()
plt.figure(figsize=(26,19))
sns.heatmap(df.corr(),cmap=sns.cm.rocket_r,annot=True)
considered_train_values = df[['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','SalePrice']]
considered_train_values.head()
sns.heatmap(considered_train_values.isnull(),cbar=False,yticklabels=False)
considered_train_values.isnull().any()
plt.figure(figsize=(10,6))
plt.plot(considered_train_values['SalePrice'])

plt.figure(figsize=(15,6))
sns.distplot(considered_train_values['SalePrice'],bins=200)
X = considered_train_values.drop('SalePrice',axis=1)
y = considered_train_values['SalePrice']
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_scaled = ss.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.33, random_state=42)
from sklearn.neighbors import KNeighborsRegressor
knn = KNeighborsRegressor(weights='distance')
knn.fit(X_train,y_train)
predictions = knn.predict(X_test)
from sklearn.metrics import mean_squared_error
mean_squared_error(y_test,predictions)

knn.score(X_test,y_test)
knn_no_weight = KNeighborsRegressor()
knn_no_weight.fit(X_train,y_train)
knn_no_weight.score(X_test,y_test)