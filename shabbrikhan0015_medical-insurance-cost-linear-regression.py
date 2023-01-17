# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra
import pandas as pd
data_train_file = "../input/insurance.csv"
df =pd.read_csv(data_train_file)
df.head()
import matplotlib.pyplot as plt
%matplotlib inline
plt.scatter("age","charges",data=df)

import seaborn as sns
sns.countplot("children",data=df)
sns.countplot("smoker",data=df)
sns.countplot("region",data = df)
plt.scatter("smoker","charges",data =df)
df.head()
def smoker(yes):
    if yes =="yes":
        return 1
    else:
        return 0
df["smoker"]=df["smoker"].apply(smoker)
def sex(s):
    if s =="male":
        return 1
    else:
        return 0
df["sex"]=df["sex"].apply(sex)

df.head()
x = df.drop(["charges","region"],axis =1)
y =df["charges"]
from sklearn.cross_validation import train_test_split
#split data into training and testing sets
from sklearn.linear_model import LinearRegression
x1,x2,y1,y2 = train_test_split(x,y,test_size = 0.3)
model = LinearRegression()
model.fit(x1,y1)
pre = model.predict(x2)
print ('acc : ',model.score(x2,y2))
print ('intercept : ',model.intercept_)
print ('coefficient : ',model.coef_)


