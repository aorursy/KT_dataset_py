import pandas as pd
df_train = pd.read_csv("../input/atividade-regressao-PMR3508/train.csv",

                          engine='python',

                          na_values="?"

                         )
df_train.shape
df_train.head()
df_test = pd.read_csv("../input/atividade-regressao-PMR3508/test.csv",

                          engine='python',

                          na_values="?"

                         )
df_test.shape
df_test.head()
df_train.isnull().sum().sort_values(ascending = False).head(5)
df_test.isnull().sum().sort_values(ascending = False).head(5)
import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
correlation = df_train.corr()



plt.figure(figsize=(16,16))

matrix = np.triu(correlation)

sns.heatmap(correlation, annot=True, mask = matrix, vmin = -0.5, vmax = 0.5, center = 0, cmap= 'coolwarm')
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()



#features escolhidas para modelagem

features = ["median_income", "latitude", "total_rooms", "median_age"]



#separação dos dados em features e labels

df_x = df_train[features]

df_y = df_train["median_house_value"]



#somente os dados de features devem ser normalizadas

df_x_normalized = pd.DataFrame(scaler.fit_transform(df_x))



df_test_normalized = pd.DataFrame(scaler.fit_transform(df_test))



#Restore feature names

df_x_normalized.columns = df_x.columns

df_test_normalized.columns = df_test.columns
df_x_normalized.head()
df_test_normalized.head()
from sklearn.linear_model import LinearRegression
linear_regression = LinearRegression().fit(df_x_normalized, df_y)
r_sq = linear_regression.score(df_x_normalized, df_y)



print('coefficient of determination:', r_sq)
from sklearn.preprocessing import PolynomialFeatures
x_polinomial = PolynomialFeatures(degree=2, include_bias=False).fit_transform(df_x_normalized)
polinomial_regression = LinearRegression().fit(x_polinomial, df_y)
r_sq = polinomial_regression.score(x_polinomial, df_y)



print('coefficient of determination:', r_sq)
features = ["median_income", "latitude", "total_rooms", "median_age"]

df_x_test = df_test_normalized[features]
#Realização da predição

y_pred = linear_regression.predict(df_x_test)



df_pred = pd.DataFrame({'Id': df_test["Id"], 'median_house_value': y_pred})



#Salvar resultados

df_pred.to_csv("submission.csv", index = False)