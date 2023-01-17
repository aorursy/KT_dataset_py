# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/titanic/train.csv')
data.head(5)
data.info()
def categorize(col):
    numerical,category=[],[]
    for i in col:
        if data[i].dtype ==object:
            category.append(i)
        else:
            numerical.append(i)
    print("The numerical features {}:".format(numerical))
    print("The categorical features {}:".format(category))
    return category,numerical
def handle_missing_values(col):
    print('Enter your choice\n1.Mean\t2.Median\t3.Mode')
    i=int(input())
    if i ==1:
        val =data[col].mean()
    elif i ==2:
        val = data[col].median()
    else:
        val = data[col].mode()[0]
    data[col] = data[col].fillna(val)

def get_correlated(cor):
    correlated =set()
    for i in cor.columns:
        for j in cor.columns:
            if cor[i][j]>0.7 or cor[i][j]>-0.7 and i!=j:
                correlated.add(i)
                correlated.add(j)
    print("The Correlated columns: {}".format(list(correlated)))
    return correlated
cat, num = categorize(data)
data[num].info()
df = data[num].isnull().sum()
for i in df.index:
    if df[i]>0:
        handle_missing_values(i)
data[num].isnull().sum()
df = data[cat].isnull().sum()
df
for i in df.index:
    if df[i]>0:
        handle_missing_values(i)
data[cat].isnull().sum()
sns.distplot(data['Age'],kde =False)
g = sns.FacetGrid(data, row = 'Survived')
g = g.map(plt.hist,'Age')
num
sns.boxplot(x='Fare',data=data)
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
sns.boxplot(x='Age',y='Sex',hue='Pclass',data=data)
sns.pairplot(data[num])
sns.heatmap(data.corr(),annot = True)
correleated = get_correlated(data[num].corr())
sns.regplot(x = 'Age', y= 'Fare',data =data)
sns.jointplot(x = 'SibSp' , y = 'Age',data= data,kind='reg')
sns.catplot(x='Sex',y='Age',data=data)
sns.countplot(data['Survived'])
plt.figure(figsize = (10,10))
sns.swarmplot(y = 'Embarked',x='Age',data=data,hue='Sex')
plt.figure(figsize=(10,10))
sns.violinplot(y = 'Sex',x ='Age',hue ='Pclass',data =data )