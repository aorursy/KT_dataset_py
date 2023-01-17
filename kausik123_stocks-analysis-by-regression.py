import pandas as pd

import datetime

import pandas_datareader.data as web

from pandas import Series, DataFrame





start = datetime.datetime(2012, 1, 1)

end = datetime.datetime(2019, 9, 1)



df = web.DataReader("AAPL", 'yahoo', start, end)

df.tail()
close_px = df['Adj Close']

mavg = close_px.rolling(window=100).mean()

mavg.tail(10)
%matplotlib inline

import matplotlib.pyplot as plt

from matplotlib import style



# Adjusting the size of matplotlib

import matplotlib as mpl

mpl.rc('figure', figsize=(8, 7))

mpl.__version__



# Adjusting the style of matplotlib

style.use('ggplot')



close_px.plot(label='AAPL')

mavg.plot(label='mavg')

plt.legend()
rets = close_px / close_px.shift(1) - 1

rets.plot(label='return')
dfcomp = web.DataReader(['AAPL', 'GE', 'GOOG', 'IBM', 'MSFT'],'yahoo',start=start,end=end)['Adj Close']

dfcomp.tail()
import seaborn as sns

retscomp = dfcomp.pct_change()



corr = retscomp.corr()

print(corr)

sns.heatmap(corr, annot=True)
sns.pairplot(retscomp);
plt.imshow(corr, cmap='hot', interpolation='none')

plt.colorbar()

plt.xticks(range(len(corr)), corr.columns)

plt.yticks(range(len(corr)), corr.columns);
plt.scatter(retscomp.mean(), retscomp.std())

plt.xlabel('Expected returns')

plt.ylabel('Risk')

for label, x, y in zip(retscomp.columns, retscomp.mean(), retscomp.std()):

    plt.annotate(

        label, 

        xy = (x, y), xytext = (20, -20),

        textcoords = 'offset points', ha = 'right', va = 'bottom',

        bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),

        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
df.columns
dfreg = df[["Adj Close","Volume"]].copy()

dfreg['HL_PCT'] = (df['High'] - df['Low'])/df['Close'] * 100

dfreg['Change_PCT'] = (df['Close'] - df['Open'])/df['Open'] * 100

dfreg.head()
import math

import numpy as np

from sklearn import preprocessing



# Drop missing value

dfreg.fillna(value=-99999, inplace=True)



# We want to separate 1 percent of the data to forecast

forecast_out = int(math.ceil(0.01 * len(dfreg)))



# Separating the label here, we want to predict the AdjClose

forecast_col = 'Adj Close'

dfreg['label'] = dfreg[forecast_col].shift(-forecast_out)

X = np.array(dfreg.drop(['label'], 1))



# Scale the X so that everyone can have the same distribution for linear regression

X = preprocessing.scale(X)



# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation

X_lately = X[-forecast_out:]

X = X[:-forecast_out]



# Separate label and identify it as y

y = np.array(dfreg['label'])

y = y[:-forecast_out]
from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor



from sklearn.linear_model import Ridge

from sklearn.preprocessing import PolynomialFeatures

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from sklearn.pipeline import make_pipeline



X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)



# Linear regression

clfreg = LinearRegression(n_jobs=-1)

clfreg.fit(X_train, y_train)

# Quadratic Regression 2

clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())

clfpoly2.fit(X_train, y_train)



# Quadratic Regression 3

clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())

clfpoly3.fit(X_train, y_train)
# KNN Regression

clfknn = KNeighborsRegressor(n_neighbors=2)

clfknn.fit(X_train, y_train)
confidencereg = clfreg.score(X_test, y_test)

confidencepoly2 = clfpoly2.score(X_test,y_test)

confidencepoly3 = clfpoly3.score(X_test,y_test)

confidenceknn = clfknn.score(X_test, y_test)
print("# results")

print('The linear regression confidence is {}'.format(confidencereg))

print('The quadratic regression 2 confidence is {}'.format(confidencepoly2))

print('The quadratic regression 3 confidence is {}'.format(confidencepoly3))

print('The KNN confidence is {}'.format(confidenceknn))
forecast_set = clfreg.predict(X_lately)

dfreg['Forecast'] = np.nan

forecast_set
last_date = dfreg.iloc[-1].name

last_unix = last_date

next_unix = last_unix + datetime.timedelta(days=1)



for i in forecast_set:

    next_date = next_unix

    next_unix += datetime.timedelta(days=1)

    dfreg.loc[next_date] = [np.nan for _ in range(len(dfreg.columns)-1)]+[i]

dfreg['Adj Close'].tail(500).plot()

dfreg['Forecast'].tail(500).plot()

plt.legend(loc=4)

plt.xlabel('Date')

plt.ylabel('Price')

plt.show()