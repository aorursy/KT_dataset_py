import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt
startups = pd.read_csv("../input/50_Startups.csv", sep = ",")

df = startups.copy()
startups.head()
startups.info()
startups.shape
startups.isnull().sum()
startups.corr()
sns.heatmap(startups.corr())
sns.set(rc={'figure.figsize':(10,7)}) # oluşacak grafiklerin uzunluğunu ve genişliğini belirliyorum.



sns.scatterplot(startups["R&D Spend"], startups["Profit"]);  # maaş ve tecrübe değerlerinin dağılımını görüntüleyelim.
plt.hist(df["Profit"],bins=None, label = "Profit")

plt.xlabel("Oranlar")

plt.ylabel("Dağılımlar")

plt.legend()

plt.title("Kar Değerleri")

plt.show()
df.describe().T
x = np.array(df["State"])

np.unique(x)
dfDummies = pd.get_dummies(df["State"])

dfDummies.head(5)
dfDummies["California"]
dfDummies["Florida"]
dfDummies["New York"]
newDF = df.drop("State", axis = 1)
newDF["California"] = dfDummies["California"] #dummyden gelen değer atanacak

newDF["New York"] = dfDummies["New York"] #dummyden gelen değer atanacak

newDF.head(5)
X = newDF.drop("Profit", axis = 1)

y = newDF["Profit"]
X #bağımsız değişken
y #bağımlı değişken (profit)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train
X_test
y_train
y_test
from sklearn.linear_model import LinearRegression



linear_regresyon = LinearRegression()
model = linear_regresyon.fit(X, y)
y_pred = linear_regresyon.predict([[100000,70000,200000,1,0]])

y_pred
newDF["tahminProfit"] = model.predict(X);

newDF.head(10)
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error

import math

MSE = mean_squared_error(newDF["Profit"], newDF["tahminProfit"])

RMSE = math.sqrt(MSE)

MAE = mean_absolute_error(newDF["Profit"], newDF["tahminProfit"])
MSE
RMSE
MAE
linear_regresyon.score(X,y)