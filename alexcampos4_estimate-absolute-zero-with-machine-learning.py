# Importing Libraries
import pylab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

sns.set_style('whitegrid')
df = pd.read_csv("../input/absolute_zero.csv")
df
df.columns = df.iloc[0]
df.drop(0, inplace=True)
df = df[['Temperature  (°C)', 'Pressure (mm Hg)']]
df.columns = ['Temp', 'Pressure']
df
df.dtypes
df = df.replace('–','-',   regex=True).apply(pd.to_numeric)
df.dtypes
fig = plt.figure(figsize=(12,6))

sns.regplot('Temp', 'Pressure', df, color='b', fit_reg=False)

plt.xlabel('Temperature (ºC)', fontsize=16)
plt.ylabel('Pressure (mm Hg)', fontsize=16)
plt.title('Pressure vs Temperature', fontsize=22)

pylab.xlim([-400, 200])
pylab.ylim([0, 1100])
df['Pressure'][5] = 500
df
fig = plt.figure(figsize=(12,6))

sns.regplot('Temp', 'Pressure', df, color='g')

plt.xlabel('Temperature (ºC)', fontsize=16)
plt.ylabel('Pressure (mm Hg)', fontsize=16)
plt.title('Pressure vs Temperature', fontsize=22)

pylab.xlim([-400, 200])
pylab.ylim([0, 1100])
from sklearn.linear_model import LinearRegression

lm = LinearRegression()
X = df['Pressure'][:, None]
y = df['Temp']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=101)
lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
y_test = y_test.as_matrix()

for i in range(len(predictions)):
    print(f'True vs Predicted: {y_test[i]} | {round(predictions[i],2)}')
lm.predict([[0]])
lm2 = LinearRegression()
lm2.fit(X, y)
lm2.predict([[0]])
