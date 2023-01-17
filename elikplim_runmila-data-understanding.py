import pandas

#names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']

data = pandas.read_csv('../input/diabetes.csv')

print(data.shape)
data.head(10)
data.dtypes
data.describe()
data.groupby('Outcome').size()
%matplotlib inline

import matplotlib.pyplot as plt



data.hist()

plt.show()