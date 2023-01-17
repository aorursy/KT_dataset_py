import pandas as pd

from sklearn.model_selection import train_test_split

import math

from sklearn import metrics

from sklearn import linear_model

import matplotlib.pyplot as plt

import seaborn as sns
df = pd.read_csv('/kaggle/input/world-happiness-report/2020.csv')

df.head()

columns = ['Country name', 'Ladder score', 'Logged GDP per capita', 'Social support', 

           'Healthy life expectancy', 'Freedom to make life choices', 'Generosity',

           'Perceptions of corruption', 'Dystopia + residual']

new_df = df[columns]

new_df.head()
new_df.describe()
plt.figure(figsize=(10, 10))

sns.heatmap(new_df.corr(), annot=new_df.corr(), cmap='Reds')

plt.show()
sns.pairplot(new_df)

plt.show()
x = new_df['Ladder score'].values

y = new_df['Logged GDP per capita'].values
x = x.reshape((-1, 1))
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.15, random_state=0)
model = linear_model.LinearRegression().fit(x_train, y_train)
alpha, beta = model.coef_[0], model.intercept_

print('Alpha = ', alpha)

print('Beta = ', beta)
y_pred_train = model.predict(x_train)



plt.figure(figsize=(10, 7))

plt.scatter(x_train, y_train, color='blue', edgecolor='k', alpha=.75, label="Dados de treinamento")

plt.plot(x_train, y_pred_train, color='red', linewidth=1, label=r"$y_{i}$ = $\alpha+\beta x_{i}$")

plt.xlabel('Índice de felicidade')

plt.ylabel('PIB per capta')

plt.legend()

plt.show()
y_pred_test = model.predict(x_test)



plt.figure(figsize=(10, 7))

plt.scatter(x_train, y_train, color='blue', edgecolor='k', alpha=.75, label="Dados de treinamento")

plt.scatter(x_test, y_test, color='green', edgecolor='k', alpha=.75, label="Dados de teste")

plt.plot(x_train, y_pred_train, color='red', linewidth=1, label=r"$y_{i}$ = $\alpha+\beta x_{i}$")

plt.xlabel('Índice de felicidade')

plt.ylabel('PIB per capta')

plt.legend()

plt.show()
plt.figure(figsize=(10, 7))

plt.scatter(y_test, y_pred_test, color='k', edgecolor='k', alpha=.75)

plt.xlabel(r'$y_{i}$ real')

plt.ylabel(r'$y_{i}$ previsto')

plt.plot([min(y_test), max(y_test)], [min(y_pred_test), max(y_pred_test)], label='Diagonal secundária', linestyle='--', alpha=.75)



plt.legend()

plt.show()
print("MAE: %.3f"%metrics.mean_absolute_error(y_test, y_pred_test))

print("MSE: %.3f"%metrics.mean_squared_error(y_test, y_pred_test))

print("RMSE: %.3f"%math.sqrt(metrics.mean_squared_error(y_test, y_pred_test)))
