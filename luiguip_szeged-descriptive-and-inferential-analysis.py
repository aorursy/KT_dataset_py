# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

from scipy.stats import zscore

import statsmodels.api as sm



from sklearn.metrics import mean_squared_error, r2_score

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

%matplotlib inline
df = pd.read_csv('/kaggle/input/szeged-weather/weatherHistory.csv')

df.info()
df.head()
df = df.set_index(pd.to_datetime(df['Formatted Date']))

df = df.drop('Formatted Date', axis=1)

df.head()
df = df[['Humidity', 'Temperature (C)', 'Apparent Temperature (C)', 'Wind Speed (km/h)']]

df = df.apply(zscore)

df.head()
sns.set()

fig, axs = plt.subplots(4, 1, figsize=(16, 24))

count = 0

for column in ['Humidity', 'Temperature (C)', 'Apparent Temperature (C)', 'Wind Speed (km/h)']:

    axs[count].hist(df[column])

    axs[count].set_xlabel(column)

    axs[count].set_ylabel('count')

    axs[count].set_title('Histogram of {}'.format(column))

    count += 1

plt.show()
sns.set(style="darkgrid")

g = sns.jointplot("Humidity", "Apparent Temperature (C)", data=df, kind="reg", color="m", height=7)
g = sns.jointplot("Temperature (C)", "Apparent Temperature (C)", data=df, kind="reg", color="m", height=7)
g = sns.jointplot("Wind Speed (km/h)", "Apparent Temperature (C)", data=df, kind="reg", color="m", height=7)
def model_summary(x_column, y_column):

    X = sm.add_constant(df[x_column])

    y = df[y_column]

    estimative = sm.OLS(y, X)

    model = estimative.fit()

    return model.summary()
model_summary('Temperature (C)', 'Apparent Temperature (C)')
model_summary('Humidity', 'Apparent Temperature (C)')
model_summary('Wind Speed (km/h)', 'Apparent Temperature (C)')
X = df[['Temperature (C)', 'Humidity']]

y = df['Apparent Temperature (C)']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

linear_regression = LinearRegression().fit(X_train, y_train)

y_pred_test = linear_regression.predict(X_test)
linear_regression.score(X_train,y_train)
linear_regression.score(X_test,y_test)
mean_squared_error(y_test, y_pred_test)
fig = plt.figure(figsize=(10, 10))

plt.scatter(X_test['Temperature (C)'], y_test,  color='black')

plt.plot(X_test['Temperature (C)'], y_pred_test, color='blue', linewidth=3)

plt.xlabel('Temperature (C)')

plt.ylabel("y")

plt.show()