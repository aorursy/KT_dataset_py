import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

from collections import Counter
# import csv file

data = pd.read_csv('../input/chopstick-effectiveness.csv')

data.head()
# show simple descriptive statistics

data.describe()
data = data.drop('Individual', axis=1)
# change name of columns

data.rename(columns={'Food.Pinching.Efficiency': 'efficiency', 'Chopstick.Length': 'len_chop'}, inplace=True)

data.head()
plt.plot(data['len_chop'], data['efficiency'])

plt.show()
# There are six length

plt.hist(data['len_chop'])
Counter(data['len_chop'])
plt.hist(data['efficiency'])

plt.show()