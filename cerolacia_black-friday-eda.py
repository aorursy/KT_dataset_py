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
train=pd.read_csv("../input/black-friday-sales-prediction/train_oSwQCTC (1)/train.csv")
train.head()
test=pd.read_csv("../input/black-friday-sales-prediction/test_HujdGe7 (1)/test.csv")
test.head()
train.shape
train.columns
m=train['Gender'].value_counts()
m
import matplotlib.pyplot as plt
import seaborn as sns
x=train['Age'].value_counts()
plt.bar(x.index,x)
plt.show()

labels = ['Male', 'Female']
colors = ['cyan', 'lightblue']
explode = [0, 0.1]

plt.pie(m, colors = colors, labels = labels, shadow = True, explode = explode, autopct = '%.2f%%')
plt.title('A Pie Chart representing the gender gap', fontsize = 20)
plt.legend()
plt.show()
train[['Product_Category_1','Product_Category_2','Product_Category_3']].groupby(train['Gender']).mean()
train[['Product_Category_1','Product_Category_2','Product_Category_3']].groupby(train['User_ID']).count()
train.groupby('User_ID').Product_ID.count()
train.info()
fig,ax=plt.subplots(figsize=(8,8))
sns.heatmap(train.corr(),annot=True)
plt.show()
sns.pairplot(train)
plt.show()
train.describe()
from scipy import stats
from scipy.stats import norm

# plotting a distribution plot for the target variable
plt.rcParams['figure.figsize'] = (7, 7)
sns.distplot(train['Purchase'], color = 'pink', fit = norm)

# fitting the target variable to the normal curve 
mu, sigma = norm.fit(train['Purchase']) 
print("The mu {} and Sigma {} for the curve".format(mu, sigma))

plt.title('A distribution plot to represent the distribution of Purchase')
plt.legend(['Normal Distribution ($mu$: {}, $sigma$: {}'.format(mu, sigma)], loc = 'best')
plt.show()

# plotting the QQplot
stats.probplot(train['Purchase'], plot = plt)
plt.show()
sns.boxplot(train['Purchase'])
fig,ax = plt.subplots(figsize=(20,4),ncols=2,nrows=1)
sns.barplot(x="Gender",y="Purchase",hue="Marital_Status",estimator=np.mean,data=train,ax=ax[0])
sns.countplot(x="Gender",hue="Marital_Status",data=train,ax=ax[1])
fig,ax = plt.subplots(figsize=(20,4),ncols=2,nrows=1)
sns.barplot(x="Stay_In_Current_City_Years",y="Purchase",hue="City_Category",order=["0","1","2","3","4+"],estimator=np.mean,data=train,ax=ax[0])
sns.countplot(x="Stay_In_Current_City_Years",hue="City_Category",order=["0","1","2","3","4+"],data=train,ax=ax[1])
fig,ax = plt.subplots(figsize=(20,4),ncols=2,nrows=1)
sns.violinplot(x="City_Category",y="Occupation",data=train,ax=ax[0])
sns.lineplot(x="Occupation",y="Purchase",data=train,ax=ax[1])
fig,ax = plt.subplots(figsize=(20,4),ncols=2,nrows=1)
sns.violinplot(x="Age",y="Purchase",order=["0-17","18-25","26-35","36-45","46-50","51-55","55+"],data=train,ax=ax[0])
sns.violinplot(x="Age",y="Occupation",order=["0-17","18-25","26-35","36-45","46-50","51-55","55+"],data=train,ax=ax[1])
