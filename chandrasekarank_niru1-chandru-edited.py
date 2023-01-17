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
df=pd.read_csv('/kaggle/input/symptoms-and-covid-presence/Covid Dataset.csv')
df.tail()
df.isnull().any()

df.describe()
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
sns.countplot(df['COVID-19'])
from sklearn.preprocessing import LabelEncoder

l=LabelEncoder()
df=df.apply(l.fit_transform).astype(int)
cor=df.corr()

cor
df.dtypes
sns.heatmap(cor)
x=df.iloc[:,list(range(0, 20))]

y=df.iloc[:,[20]]
x.head()
y.head()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=6,test_size=0.2)
print(x_train.shape)

print(x_test.shape)
from sklearn.linear_model import LogisticRegression

lr= LogisticRegression()
from sklearn.model_selection import cross_val_score

#help(cross_val_score)

scores = cross_val_score(lr, x, np.array(y).reshape(-1),

                              cv=5, scoring = 'f1')



print("F1 scores:\n", scores)
lr.fit(x_train,y_train)
print("Enter the following input : ")

cols = x.columns

inp = []

inp = [0, 1, 1,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]



# take this comment out if you want to enter manually

# for i in range(len(cols)):

#     print("{}".format(cols[i]))

#     inp.append(int(input()))



inp = np.array(inp)

print(inp)

res = lr.predict(inp.reshape(-1,20))

if res == 1:

    print("Yes you have covid")

else:

    print("No you dont have covid")