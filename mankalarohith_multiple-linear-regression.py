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
import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from sklearn.metrics import accuracy_score,r2_score

import pandas as pd

from ipywidgets import interact

from sklearn.model_selection import train_test_split
df=pd.read_csv("../input/fish-market/Fish.csv")

df.head()
x=df.iloc[:,2:]

y=df.iloc[:,1]

x
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.65)
np.random.seed(16)

np.random.permutation(6)
MLmodel=LinearRegression()

MLmodel.fit(x_train,y_train)
m=MLmodel.coef_

c=MLmodel.intercept_
x=[22,22,22,10,4]

y_pred=MLmodel.predict([x])

y_pred
ytest_predict=MLmodel.predict(x_test)

r2_score(ytest_predict,y_test)
def FishWeightPredict(Len1,Len2,Len3,Height,Width):

    y_pred=MLmodel.predict([[Len1,Len2,Len3,Height,Width]])

    print("Fish Weight is:",y_pred[0])
Len1_min = df.iloc[:, 2].min()

Len1_max = df.iloc[:, 2].max()

Len2_min = df.iloc[:, 3].min()

Len2_max = df.iloc[:, 3].max()

Len3_min = df.iloc[:, 4].min()

Len3_max = df.iloc[:, 4].max()

Height_min = df.iloc[:, 5].min()

Height_max = df.iloc[:, 5].max()

Width_min = df.iloc[:, 6].min()

Width_max = df.iloc[:, 6].max()
FishWeightPredict(20,20,20,10,4)

interact(FishWeightPredict, Len1 = (Len1_min, Len1_max),

         Len2= (Len2_min, Len2_max), Len3  = (Len3_min, Len3_max),Width=(Width_min,Width_max),Height=(Height_min,Height_max))