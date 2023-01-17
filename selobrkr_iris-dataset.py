# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np #for linear algebra

import pandas as pd #for data processing



import matplotlib.pyplot as plt #for graphs

%matplotlib inline

import seaborn as sns #for graphs
#first things first, let's read the file 

df = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')

df.head()
#let's check how many rows&columns we have and if we have any null values

df.info()
#checking if target classes are well balanced

sns.countplot(x='species',data=df)
g = sns.PairGrid(data=df,hue='species')

g.map_upper(plt.scatter)

g.map_lower(plt.scatter)

g.map_diag(plt.hist)

g = g.add_legend()
df.corr()
df[df['species'] == 'Iris-setosa'].mean()
df[df['species'] == 'Iris-versicolor'].mean()
df[df['species'] == 'Iris-virginica'].mean()
from sklearn.model_selection import train_test_split
X = df.drop('species',axis=1)

y = df['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()

logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))