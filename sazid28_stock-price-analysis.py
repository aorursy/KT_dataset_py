import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
data = pd.read_csv("../input/WIKI-PRICES.csv")
data.head()
data.columns
cor = data.corr()
print(cor)
data.describe()
data.info()
x = data[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume',
       'ex-dividend', 'split_ratio', 'adj_open', 'adj_high', 'adj_low',
       'adj_close']]
x.head()
y = data[["adj_volume"]]
y.head()
from sklearn.cross_validation import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(x,y,random_state=1)
X_train.shape
X_test.shape
Y_train.shape
Y_test.shape
from sklearn.linear_model import LinearRegression
lin = LinearRegression()
lin.fit = (X_train,Y_train)
dt = data[["date"]]
dt
op = data[["open"]]
op
import seaborn as sns
sns.pairplot(data,x_vars=["dt"],y_vars=["op"],size=7,aspect=0.7)
dta = np.reshape(dt,(len(dt),1))
print(dta)