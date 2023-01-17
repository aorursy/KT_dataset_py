import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv('../input/perrin-freres-monthly-champagne-sales/Perrin Freres monthly champagne sales millions.csv')

df.head()
df.drop([105,106], axis = 0, inplace = True)
df.columns = ['Month', 'Sales']
df.head()
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace = True)
df.head()
plt.figure(figsize = (12, 5))

plt.plot(df['Sales'])
plt.show()
from statsmodels.tsa.stattools import adfuller

def adfuller_test(sales):
    result = adfuller(sales)
    labels = ['ADF test statistics', 'P-value', '#Lags used', 'Number of observation used']
    for value, label in zip(result, labels):
        print(label+' : '+str(value))
    if result[1] <= 0.05:
        print('Strong evidence against the null hypothesis (Ho), Reject the null hypothesis, Data has no unit root and is stationary')
    else:
        print('Weak evidence against the null hypothesis (Ho), time series has a unit root, indicating it is non stationary. ')
        
        
adfuller_test(df['Sales'])
df['seasional_first_difference'] = df['Sales'] - df['Sales'].shift(12)
df
plt.figure(figsize = (12, 5))

plt.plot(df['seasional_first_difference'])
plt.show()
adfuller_test(df['seasional_first_difference'].dropna())