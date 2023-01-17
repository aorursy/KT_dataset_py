# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

pd.set_option('display.precision', 2)
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv('../input/videogamesales/vgsales.csv')
df.head()
df.shape
df.columns
df.info()
df["Year"].fillna( method ='ffill', inplace = True) 
df.info()
df.describe()
df["Genre"].value_counts()
df.sort_values(by='Global_Sales', ascending=False).head()
df.plot(x ='Global_Sales', y='Genre', kind = 'scatter')
q_low = df["Global_Sales"].quantile(0.25)
q_hi  = df["Global_Sales"].quantile(0.75)

df_filtered = df[(df["Global_Sales"] > q_low) & (df["Global_Sales"] < q_hi)]
df_filtered.plot(x ='Global_Sales', y='Genre', kind = 'scatter')
X = pd.DataFrame(df_filtered, columns=['Platform', 'Year', 'Genre'])
y = df_filtered['Global_Sales']

# Categorize
X['Platform'] = pd.Categorical(X['Platform'])
X['Platform_code'] = X['Platform'].cat.codes

X['Genre'] = pd.Categorical(X['Genre'])
X['Genre_code'] = X['Genre'].cat.codes

X = pd.DataFrame(X, columns=['Year', 'Platform_code', 'Genre_code'])

X.describe()
X.hist(X.columns, figsize=(10, 10));
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
from sklearn.linear_model import LinearRegression

model = LinearRegression()

model.fit(X_train, y_train)

y_train_prediction = model.predict(X_train)
y_test_prediction = model.predict(X_test)
import matplotlib.pyplot as plt
%matplotlib inline

plt.figure(figsize=(20, 8))
plt.bar(X.columns, model.coef_)
from sklearn.metrics import mean_squared_error, mean_absolute_error

print(f'Train MSE: {mean_squared_error(y_train, y_train_prediction)}')
print(f'Test MSE: {mean_squared_error(y_test, y_test_prediction)}')

print(f'Train MAE: {mean_absolute_error(y_train, y_train_prediction)}')
print(f'Test MAE: {mean_absolute_error(y_test, y_test_prediction)}')
y.mean()
from sklearn.model_selection import cross_val_score

result = cross_val_score(estimator=LinearRegression(), X=X, y=y, scoring='neg_mean_absolute_error', cv=5)
result
print(f'Среднее MAE равно {-result.mean()}, стандартное отклонение MAE равно {result.std()}')