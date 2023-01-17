import pandas as pd

import numpy as np

import seaborn as sns

from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
df = pd.read_csv("../input/startups/50_Startups.csv").copy()
df.head()

df.info()
df.shape
df.isna().sum()
df.corr() 

#korelasyon degeri cogunlukla 1 e yakin. Bu da aralarında guclu bir iliksi olduğunu gösterir
corr = df.corr()

sns.heatmap(corr, 

            xticklabels=corr.columns.values,

            yticklabels=corr.columns.values);
sns.scatterplot(x = "R&D Spend", y = "Profit", data = df);

#tabloya bakinca lineer bir dağılım var. korlelasyonu guclu.
sns.distplot(df["Profit"], bins=16, color="black");

sns.distplot(df["Marketing Spend"], bins=16, color="black");
sns.distplot(df["Administration"], bins=16, color="black");
sns.distplot(df["R&D Spend"], bins=16, color="black");
#bağımlo değişken ile bağımsız değişken arasındaki ilişkinin gösterimi

sns.pairplot(df, x_vars=['R&D Spend','Administration','Marketing Spend'], y_vars='Profit', size=7, aspect=0.7)
df.describe().T

df["State"].unique()
df['State'] = pd.Categorical(df['State'])

dfDummies = pd.get_dummies(df['State'])

dfDummies
df = pd.concat([df, dfDummies], axis=1)

df.drop(["State","New York" ], axis = 1, inplace = True)

df.head()
X = df.drop("Profit", axis = 1)

y = df["Profit"]
X.head()
y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 3512, shuffle=1)
X_train.head()

X_test.head()
y_test.head()
y_train.head()
lm = LinearRegression()
model = lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)
df_a = pd.DataFrame({'Gercek': y_test, 'Tahmin': y_pred})

df_a

#gercek deger ile tahmin edilen deger arasinda cok fazla fark yok. bu da makinenin başarısınin yuksek oldugunu gosterir.
from sklearn.metrics import mean_squared_error



MSE = mean_squared_error(y_test, y_pred)

MSE
from sklearn.metrics import mean_absolute_error



MSA = mean_absolute_error(y_test, y_pred)

MSA
import math



RMSE = math.sqrt(MSE)

RMSE

model.score(X, y)
import statsmodels.api as stat

stmodel = stat.OLS(y, X).fit()

stmodel.summary()