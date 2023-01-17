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
import pandas as pd
import numpy as np
import scipy 
data = pd.read_csv("../input/StudentsPerformance.csv")
data.head()
data.isnull().sum().any
data.shape
data.describe()
df=data[['math score','reading score','writing score']]
df.head()
import matplotlib.pyplot as plt
dat = [df['math score'],df['reading score'],df['writing score']]
fig7, ax7 = plt.subplots()
ax7.boxplot(dat)
#plt.boxplot()
plt.show()
passing = 40
df['math_status'] = np.where(df['math score']<passing,"Fail","Pass")
print("Maths Status\n",df['math_status'].value_counts())
df['reading status'] = np.where(df['reading score']<passing,"Fail","Pass")
print("reading status\n",df['reading status'].value_counts())
df['writing status'] = np.where(df['writing score']<passing,"Fail","Pass")
print("writing status\n",df['writing status'].value_counts())
import seaborn as sns 
sns.countplot(x="math score", data = df)
plt.show()
sns.countplot(x="reading score", data = df)
plt.show()
sns.countplot(x="writing score", data = df)
plt.show()
df['Total Marks'] = df['math score']+df['reading score']+df['writing score']
df['average'] = (df['math score']+df['reading score']+df['writing score'])/3
sns.countplot(x='parental level of education', data = data, hue=df['math_status'])
plt.show()
sns.countplot(x='gender', data = data, hue=df['math_status'],palette='bright')
plt.show()
sns.countplot(x='parental level of education', data = data, hue=df['reading status'])
plt.show()
sns.countplot(x='gender', data = data, hue=df['reading status'],palette='bright')
plt.show()
sns.countplot(x='parental level of education', data = data, hue=df['writing status'])
plt.show()
sns.countplot(x='gender', data = data, hue=df['writing status'],palette='bright')
plt.show()
sns.countplot(x = 'average',data = df,palette='bright')
plt.show()
mapping = {np.nan:0,"Pass": 1,"Fail":0}
name = np.array(['math_status','reading status' ,'writing status'])
for i in name:
     df[i] = df[i].map(mapping)
df['overall_status'] = df.apply(lambda x : 0 if x['math_status'] == 0 or 
                                    x['reading status'] == 0 or x['writing status'] == 0 else 1, axis =1)
df.overall_status.value_counts()
sns.countplot(x='parental level of education', data = data, hue=df['overall_status'])
def GetGrade(Percentage, OverAll_PassStatus):
    if ( OverAll_PassStatus == 'F'):
        return 'F'    
    if ( Percentage >= 80 ):
        return 'A'
    if ( Percentage >= 70):
        return 'B'
    if ( Percentage >= 60):
        return 'C'
    if ( Percentage >= 50):
        return 'D'
    if ( Percentage >= 40):
        return 'E'
    else: 
        return 'F'

df['Grade'] = df.apply(lambda x : GetGrade(x['average'], x['overall_status']), axis=1)
df.Grade.value_counts()
sns.countplot(x="Grade", data = df, order=['A','B','C','D','E','F'])
plt.show()
corr = df.corr()
sns.heatmap(corr)
data['Total Marks'] = df['Total Marks']
data.groupby('parental level of education')['Total Marks'].std()
df.boxplot('Total Marks')
