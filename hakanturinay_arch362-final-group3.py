# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
wind_multi = pd.read_csv('../input/izmir-wind-speed/weatherHistory.csv')

wind_multi.head(10)
wind_multi.info()
wind_multi.describe()
import matplotlib.pyplot as plt

dataset = pd.read_csv('../input/izmir-wind-speed/weatherHistory.csv')

X = dataset.iloc[:, :-1].values

y = dataset.iloc[:, 3].values

print(X)

print(y)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.pairplot(wind_multi, vars=['Wind Speed (km/h)','Wind Bearing (degrees)'])
X= wind_multi[['Formatted Date','Wind Speed (km/h)','Wind Bearing (degrees)']]

X.head()
y= wind_multi['Wind Speed (km/h)']

y.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7,random_state=100 )
print(X_train.shape)

print(y_train.shape)

print(X_test.shape)

print(y_test.shape)
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
plt.figure(figsize=(5,5))

sns.heatmap(wind_multi.corr(), annot=True)