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
df1 = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict.csv')
df2 = pd.read_csv('/kaggle/input/graduate-admissions/Admission_Predict_Ver1.1.csv')
df2.shape
df = df2
df.head()
df.drop('Serial No.', axis=1, inplace=True)
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
df.head()
import seaborn as sns
df['SOP'].skew()
df
sns.distplot(df['CGPA'])
df['CGPA'].skew()
sns.boxplot(df['CGPA'])
sns.countplot(df['Research'])
sns.scatterplot(df['CGPA'],df['Chance of Admit '])
df['University Rating'].sample(5)
import matplotlib.pyplot as plt
plt.scatter(df['GRE Score'],df['Chance of Admit '],c=df['University Rating'])
cdict = {1: 'red', 2: 'blue', 3: 'green',4:'yellow',5:'black'}
group = df['University Rating']
fig, ax = plt.subplots()
for g in np.unique(group):
    ix = np.where(group == g)
    ax.scatter(df['GRE Score'], df['Chance of Admit '], c = cdict[g], label = g, s = 100)
ax.legend()
plt.show()
np.unique(group)
sns.barplot(df['University Rating'],df['Chance of Admit '],hue=df['Research'])
sns.barplot(df['Research'],df['Chance of Admit '])
sns.pairplot(df)
sns.heatmap(df.corr())
df.corr()['Chance of Admit ']
# How to handle categorical data
df.shape
df = pd.get_dummies(df,columns=['University Rating'],drop_first=True)
df.head()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(df)
X[0]
m = df['GRE Score'].mean()
s = df['GRE Score'].std()
(df['GRE Score'][0] - m)/s
X
# Conclusions

# GRE score has a linear relationship with chance of getting addmitted
# Reseach wale jyada fodte hai(both in GRE and TOEFL, even if they are from the same univ)