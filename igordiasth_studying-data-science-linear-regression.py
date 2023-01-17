# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
# Reading the dataset file and replace the commam for dot

df = pd.read_csv('../input/Consumo_cerveja.csv', decimal=',')
# Update the name of the columns

df.columns=["date", "temp_medio", "temp_min", "temp_max", "rain", "weekend", "consumo"]
# Removing the NAN

df["consumo"] = df["consumo"].astype(float)

df = df.dropna()
# Displaying the first 5 rows of the DataFrame

df.head()
# Number of rows and columns

df.shape
# Summary of information in all columns

df.describe().round(2)
# Correlation Matrix

df.corr().round(4)
fig, ax = plt.subplots(figsize=(20, 6))



ax.set_title('Beer consumption', fontsize=20)

ax.set_ylabel('Liters', fontsize=16)

ax.set_xlabel('Days', fontsize=16)

ax = df['consumo'].plot(fontsize=16)
fig, ax = plt.subplots(figsize=(20, 6))



ax.set_title('Average Temperature', fontsize=20)

ax.set_xlabel('Days', fontsize=16)

ax.set_ylabel('Graus Celsius', fontsize=16)

ax = df['temp_medio'].plot(fontsize=16)
ax = sns.boxplot(x = 'weekend', y = 'consumo', data = df, orient = 'v', width = 0.5)

ax.figure.set_size_inches(12, 6)

ax.set_title('Beer Consumption', fontsize=20)

ax.set_ylabel('Liters', fontsize=16)

ax.set_xlabel('Weekend', fontsize=16)

ax
ax = sns.pairplot(df, y_vars = 'consumo', x_vars = ['temp_min', 'temp_medio', 'temp_max', 'rain', 'weekend'])

ax.fig.suptitle('Dispersion between the variables', fontsize=20, y=1.10)

ax
ax = sns.pairplot(df, y_vars='consumo', x_vars=['temp_min', 'temp_medio', 'temp_max', 'rain', 'weekend'], kind='reg')

ax.fig.suptitle("Dispersion between the variables", fontsize=20, y=1.10)

ax
ax = sns.lmplot(x = 'temp_max', y = 'consumo', data = df, hue='weekend', markers=['o', '*'], legend=False)

ax.fig.suptitle("Regression Line - Consumption X Temperature", fontsize=20, y=1.10)

ax.set_xlabels("Max Temperature (Celsius)", fontsize=16)

ax.set_ylabels("Beer Consumption (Liters)", fontsize=16)

ax.add_legend(title="Weekend")

ax
from sklearn.model_selection import train_test_split
y = df['consumo']
X = df[['temp_max', 'rain', 'weekend']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2811)
X_train.shape
X_test.shape
X.shape[0] * 0.3
X.shape[0] * 0.7
from sklearn.linear_model import LinearRegression

from sklearn import metrics
# Instanciando a classe LinearRegression()

model = LinearRegression()
# Method fit() object model

model.fit(X_train, y_train)
print("R2 = {}".format(model.score(X_train, y_train).round(2)))
y_predict = model.predict(X_test)
print("R2 = %s" % metrics.r2_score(y_test, y_predict).round(2))
temp_max = 40

rain = 0

weekend = 1

entrance = [[temp_max, rain, weekend]]



print('Consumption: {0:.2f} liters'.format(model.predict(entrance)[0]))
entrada = X_test[0:1]

entrada
# Consumption average in liters in a day of maximum temperature of 30.5, with precipitation of rain of 12.2 mm and is not weekend

model.predict(entrada)[0]