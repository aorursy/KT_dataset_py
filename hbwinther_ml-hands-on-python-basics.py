1 + 2
a = 1

b = 2.5

a + b
a = 'Hello World'

a
a = ['1', 2, b, 1+2]

a
a[1:]
import numpy as np                     # array goodnes

from pandas import DataFrame, read_csv # excel for python

from matplotlib import pyplot as plt   # plotting library



%matplotlib inline

import seaborn as sns

plt.style.use('fivethirtyeight')       # nice colors

plt.xkcd()

plt.rc('font',family='DejaVu Sans')
a = np.array([2, b, 1+2])

print('mean:', np.mean(a), 'sd:', np.std(a), 'median:', np.median(a))

print('min:', np.min(a), 'max:', np.max(a), 'sum:', np.sum(a))
df_iris = read_csv('../input/Iris.csv')

df_iris.head()
df_iris.describe()
sns.pairplot(data=df_iris, hue='Species', diag_kind='kde')
X_iris = df_iris.drop('Species', axis=1)

Y_iris = df_iris['Species']
X = X_iris['SepalLengthCm']

Y = X_iris['PetalLengthCm']



mask = Y_iris == 'Iris-virginica'

X = X[mask]

Y = Y[mask]



plt.scatter(X, Y, label='Iris-virginica')

plt.xlabel('sepal length (cm)')

plt.ylabel('petal length (cm)')

plt.legend()