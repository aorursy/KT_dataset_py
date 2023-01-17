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
train_df=pd.read_csv("/kaggle/input/autompg-dataset/auto-mpg.csv")
train_df
#fetch the data
import seaborn as sns
from matplotlib import pyplot as plt
#plt.figure(figsize=(16,9))
sns.barplot(x="origin",y="mpg",data=train_df)
#concludes that origin 3 has highest mpg

sns.distplot(train_df["mpg"])
# graph shows that data is slightly skewed to right
sns.barplot(x="model year",y="mpg",data=train_df)
# shows model year 80 has highest mpg
from sklearn import preprocessing
label_encoder=preprocessing.LabelEncoder()
train_df["car name"]=label_encoder.fit_transform(train_df["car name"])
#Label Encoding of car names
train_df.skew()
#checking skewness
train_df.horsepower.unique()
#checking the values to know why horsepower datatype has object
train_df=train_df[train_df["horsepower"] != '?']
#train_df having values that are only not equal to ?
train_df["horsepower"].unique()
#no values except int
train_df.horsepower=train_df.horsepower.astype('float')
#setting horsepower datatype to float
train_df.dtypes
def scaled(col):
         res = (col - col.min(axis=0)) / (col.max(axis=0) - col.min(axis=0))
         return res
#implementing a function that performs MinMaxScaler

train_df["mpg"]=scaled(train_df["mpg"])
train_df["displacement"]=scaled(train_df["displacement"])
train_df["weight"]=scaled(train_df["weight"])
train_df["acceleration"]=scaled(train_df["acceleration"])
train_df["horsepower"]=scaled(train_df["horsepower"])
#getting the scaled values


target=train_df["mpg"]
del train_df["mpg"]
#setting the target value
train_df
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(train_df,target,random_state=0)

X_train
y_train
X_test
y_test
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=2,include_bias=False)
poly.fit(X_train)
X_train=pd.DataFrame(poly.transform(X_train))
X_test=pd.DataFrame(poly.transform(X_test))

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_train,y_train)

print(reg.score(X_test,y_test)*100)
print(reg.score(X_train,y_train)*100)
