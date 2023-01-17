import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_excel('../input/random-time-series-data/ts_data.xlsx')
df.head()
df.shape
df.isnull().sum()
df['Value'].plot()
from statsmodels.tsa.stattools import adfuller

def adfuller_test(Value):
    result = adfuller(Value)
    labels = ['ADF test statistics', 'P-value', '#Lags used', 'Number of observation used']
    for value, label in zip(result, labels):
        print(label+' : '+str(value))
    if result[1] <= 0.05:
        print('Strong evidence against the null hypothesis (Ho), Reject the null hypothesis, Data has no unit root and is stationary')
    else:
        print('Weak evidence against the null hypothesis (Ho), time series has a unit root, indicating it is non stationary. ')
        
        
adfuller_test(df['Value'])
from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(df['Value'], lags=10)
df['Value_shifted_1'] = df['Value'].shift(1)
df.head(3)
df.drop('Time', axis = 1, inplace = True)
df.head(3)
df.dropna(inplace = True)
df.head(3)
X = df.Value_shifted_1.values
y = df.Value.values
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2)
lr1 = LinearRegression()
model = lr1.fit(X_train, y_train)
y1_pred = lr1.predict(X_test)
plt.plot(y_test[-10:], label="Actual Values", color = 'Green', marker = 'o')
plt.plot(y1_pred[-10:], label="Predicted Values", color = 'Red', marker = 'x')
plt.legend()
plt.show()
from sklearn.metrics import r2_score

print(r2_score(y_test, y1_pred).round(4))