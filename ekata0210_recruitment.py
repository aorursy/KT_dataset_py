import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn import preprocessing
train = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')

print(train.columns.values)
train.shape
train.head()
train.tail()
plt.hist(train['status'])
train.info()
def find_missing_data(data):

    Total = data.isnull().sum().sort_values(ascending = False)

    Percentage = (data.isnull().sum()/data.isnull().count()).sort_values(ascending = False)

    

    return pd.concat([Total,Percentage] , axis = 1 , keys = ['Total' , 'Percent'])
find_missing_data(train)
corrMatrix = train.corr()

sns.heatmap(corrMatrix, annot=True)

plt.show()
number = preprocessing.LabelEncoder()

train['status'] = number.fit_transform(train['status'].astype('str'))
number = preprocessing.LabelEncoder()

train['gender'] = number.fit_transform(train['gender'].astype('str'))
number = preprocessing.LabelEncoder()

train['workex'] = number.fit_transform(train['workex'].astype('str'))

train.head()
corr_numeric = sns.heatmap(train[["status","mba_p","etest_p","hsc_p","degree_p","ssc_p", "gender", "workex"]].corr(),

                           annot=True, fmt = ".2f", cmap = "summer")
sns.barplot(x='gender', y='status', data=train)
sns.barplot(x='degree_t', y='status', data=train)
sns.barplot(x='degree_t', y='salary', data=train)
sns.barplot(x='specialisation', y='status', data=train)
# ‘hue’ is used to visualize the effect of an additional variable to the current distribution.  

sns.countplot(train.degree_t, hue=train['status'])  

plt.show() 
corr_numeric = sns.heatmap(train[["salary","mba_p","etest_p","hsc_p","degree_p","ssc_p", "gender", "workex"]].corr(),

                           annot=True, fmt = ".2f", cmap = "summer")
sns.barplot(x='workex', y='salary', data=train)
fig, ax = plt.subplots(figsize=(10, 8))  

sns.violinplot(x='gender', y='salary', data=train, ax=ax)  

ax.set_title('Violin plot')  

plt.show()  