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
df = pd.read_csv('/kaggle/input/60k-stack-overflow-questions-with-quality-rate/data.csv')

df.head()
df.tail()
df=df.drop(['Id','CreationDate'],axis=1)

df
df.isnull().sum()
df.dtypes
x = df.drop(columns=['Y'])

x
y = df['Y']

y
from sklearn import preprocessing 

label_encoder = preprocessing.LabelEncoder()  

x= x.apply(label_encoder.fit_transform)

x
y= label_encoder.fit_transform(y)

y
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)
from sklearn.linear_model import LassoCV

from sklearn.datasets import make_regression

#x_train, x_test = make_regression(x.all(x,y),noise=4, random_state=0)

reg = LassoCV(cv=5, random_state=0).fit(x_train, y_train)

reg.score(x_train, y_train)