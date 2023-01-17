
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
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df = pd.read_csv('../input/heart.csv')
df.head()
y = df.age
x = df.drop('age', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y,test_size=0.3)

print("\nx_train:\n")
print(x_train.head())
print(x_train.shape)

print("\nx_train:\n")
print(x_test.head())
print(x_test.shape)

from sklearn.model_selection import train_test_split
df['target'].describe()
#histogram
sns.distplot(df['target']);
#skewness and kurtosis
print("Skewness: %f" % df['target'].skew())
print("Kurtosis: %f" % df['target'].skew())
# scatter plot age/target
var = 'age'
data = pd.concat([df['target'], df[var]], axis=1)
data.plot.scatter(x=var, y='target', ylim=(0,1));
# scatter plot sex/target
var = 'sex'
data = pd.concat([df['target'], df[var]], axis=1)
data.plot.scatter(x=var, y='target', ylim=(0,1));
# heatmap
corelmat = df.corr()
plt.figure(figsize = (8,6))
sns.heatmap(corelmat)
plt.show()
sns.boxplot(x = 'target', y = 'age', data = df)
sns.boxplot(x = 'target', y = 'chol', data = df)
sns.boxplot(x = 'target', y = 'thalach', data = df)
sns.boxplot(x = 'target', y = 'oldpeak', data = df)
sns.boxplot(x = 'target', y = 'slope', data = df)
sns.boxplot(x = 'target', y = 'ca', data = df)
sns.boxplot(x = 'target', y = 'trestbps', data = df)
#convert categorical variable into dummy
df = pd.get_dummies(df)
df.info()
# End