# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('/kaggle/input/insurance/insurance.csv')
data=df.copy()
data.shape
data.head(10)
data.info()
data.isnull().sum()
data.sample(50)
data.region.value_counts()
data.columns

data.sex.value_counts()
data.smoker.value_counts().plot.bar()
data.corr().T
f,ax=plt.subplots(figsize=(8,8))

sns.heatmap(data.corr(),annot=True,fmt='.2f',ax=ax)

plt.show()
data.corr()['charges'].sort_values()
from sklearn.preprocessing import LabelEncoder

#sex

le = LabelEncoder()

le.fit(data.sex.drop_duplicates()) 

data.sex = le.transform(data.sex)

# smoker or not

le.fit(data.smoker.drop_duplicates()) 

data.smoker = le.transform(data.smoker)

#region

le.fit(data.region.drop_duplicates()) 

data.region = le.transform(data.region)
data.head()
f= plt.figure(figsize=(12,5))



ax=f.add_subplot(121)

sns.distplot(data[(data.smoker == 1)]["charges"],color='c',ax=ax)

ax.set_title('Distribution of charges for smokers')



ax=f.add_subplot(122)

sns.distplot(data[(data.smoker == 0)]['charges'],color='b',ax=ax)

ax.set_title('Distribution of charges for non-smokers')

plt.show()
sns.catplot(x="smoker", kind="count",hue = 'sex', palette="pink", data=data)

plt.figure(figsize=(12,5))

plt.title("Box plot for charges of women")

sns.boxplot(y="charges", x="smoker", data =  data[(data.sex == 1)])
plt.figure(figsize=(12,5))

plt.title("Distribution of bmi")

ax = sns.distplot(data["bmi"], color = 'm')
plt.figure(figsize=(12,5))

plt.title("Distribution of charges for patients with BMI greater than 30")

ax = sns.distplot(data[(data.bmi >= 30)]['charges'], color = 'm')
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import r2_score,mean_squared_error
x
x = data.drop(['charges'], axis = 1)

y = data.charges



x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state = 42)

lr = LinearRegression().fit(x_train,y_train)



print(lr.score(x_test,y_test))
pre=x.iloc[32:33,:]
pre
lr.predict([[17,1,26,1,0,2]])
lr.predict(pre)