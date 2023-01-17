import numpy as np

import seaborn as sns

import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv("../input/warm-up-dataset/50_Startups.csv")
df.head(5)
df.info()
df.shape
df.isna().sum()
df.corr()

corr = df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);
sns.scatterplot(x="R&D Spend", y="Profit", data=df, color="blue");
sns.distplot(df["R&D Spend"], color="blue");

sns.distplot(df["Profit"], color="red");
df.hist(figsize = (20,20))

plt.show()
df.describe()

#Varyansın karesi alındığında standart sapma elde edilir.
df["State"].unique()
df['State'] = pd.Categorical(df['State'])
df_State = pd.get_dummies(df['State'],)
df_State
df_State.colums = ['California', 'Florida', 'New York']

df_State.head()
df = pd.concat([df, df_State], axis=1)

df.drop(["Florida", "State"], axis=1, inplace = True)

df.head()
X =df[['R&D Spend', 'Administration', 'Marketing Spend']]

y = df['Profit']
X
y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state = 0)
X_train
X_test
y_train
y_test
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
model = lm.fit(X_train, y_train)
y_predict = lm.predict(X_test)

y_predict
df_predict = pd.DataFrame({'Profit': y_test, 'Predicted Profit': y_predict})

df_predict
from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

import math
MAE = mean_absolute_error(y_test, y_predict)

MAE
MSE = mean_squared_error(y_test, y_predict)

MSE
RMSE = math.sqrt(MSE)

RMSE
model.score(X, y)