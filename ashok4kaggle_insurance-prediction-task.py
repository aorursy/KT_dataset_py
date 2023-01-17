#importing the relevant library

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
%matplotlib inline
#reading the data

raw_data = pd.read_csv('../input/insurance/insurance.csv')
raw_data.head()
# Looking for no. of missing value in each column
raw_data.isnull().sum()
#checking for missing value by heatmap

sns.heatmap(raw_data.isnull(),cbar=True)
# Some other tools to check for missing value
from missingno import matrix,heatmap
matrix(raw_data)
# Number of male and female patient  
raw_data['sex'].value_counts()
raw_data['sex'].value_counts().plot.bar(color='y')
raw_data['age'].value_counts().sort_values(ascending=False)
fig = plt.figure(figsize=(12,6))
sns.countplot(raw_data['age'],hue=raw_data['smoker'],palette=['red','green'],)
#Check Point
df = raw_data.copy()
# Encoding the data with map function

df['sex'] = raw_data['sex'].map({'female':0,'male':1})
df['smoker'] = raw_data['smoker'].map({'yes':1,'no':0})
df['region'] = raw_data['region'].map({'southeast':0,'southwest':1,'northwest':2,'northeast':3})
raw_data['region'].value_counts()
df.head()
sns.heatmap(df.corr(),cmap='Wistia',annot=True)
# Check Point 2
df2 = raw_data.copy()
df2.head()
## Label Encoding

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
df2['region'] = lb.fit_transform(df2['region'])
df2['sex'] = lb.fit_transform(df2['sex'])
df2['smoker'] = lb.fit_transform(df2['smoker'])
df2.head()
# Check point encoding
df3 = raw_data.copy()
df3.head()
## Using OneHotEncoding

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(drop='first')
x = pd.DataFrame(ohe.fit_transform(df3[['sex','children','smoker','region']]).toarray())
x.columns = ['Male','children_1','children_2','children_3','children_4','children_5','smoker','northwest','southeast','southwest']
df4 = pd.concat([df3.drop(['sex','children','smoker','region'],axis=1),x],axis=1)
df4.head()
plt.figure(figsize=(15,6))
sns.heatmap(df4.corr(),annot=True)
## Creating the check Point
new_data = df4
sns.scatterplot(x=raw_data['bmi'],y=raw_data['charges'],hue=raw_data['smoker'])
sns.lmplot(data=df4,x='bmi',y='charges',aspect=2,height=6,hue='smoker')
# Looking for continous variable

fig,axes = plt.subplots(nrows=1,ncols=2,figsize=(15,5))
sns.distplot(df4['charges'],ax=axes[0],color='purple')
sns.distplot(df4['bmi'],ax=axes[1],color='orange')
# 
sns.distplot(np.log(df4['charges']),color='purple')
df4.columns
# Checkin point applying linear regression

new_data = df4.copy()
new_data = new_data.reindex(['age', 'bmi', 'Male', 'children_1', 'children_2',
       'children_3', 'children_4', 'children_5', 'smoker', 'northwest',
       'southeast', 'southwest','charges'],axis=1)
new_data.head()
import statsmodels.api as sm
x1 = new_data.iloc[:,:-1] # independent variable
y = new_data.iloc[:,-1] #dependent variable
x = sm.add_constant(x1)
result = sm.OLS(y,x).fit()
result.summary()
