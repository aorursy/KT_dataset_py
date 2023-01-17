import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv('../input/mycsgo-data/model_data.csv')

df
df.describe()
df.info()
from scipy import stats

z = np.abs(stats.zscore(df.iloc[:,1:]))

df = df[(z < 3).all(axis=1)]

df
plt.style.use('fivethirtyeight')

sns.kdeplot(df['Kills'])

plt.title("Kills")

plt.show()
sns.heatmap(df.corr(),square = True,cbar=True,annot=True,annot_kws={'size': 8})
sns.pairplot(df)
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

y = df.iloc[:, 2]

x = df.iloc[:, [1, 3, 4, 5, 6]]

scaler = StandardScaler()

scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
from sklearn.metrics import r2_score, mean_squared_error

from sklearn.linear_model import LinearRegression,Ridge,Lasso

model = LinearRegression(normalize = True)

model.fit(x_train, y_train)

y_pred = model.predict(x_test)



print(model.score(x_test, y_test))

print(mean_squared_error(y_pred, y_test))
sns.residplot(y_test, y_pred, color="orange", scatter_kws={"s": 2})

plt.show()