import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
data = pd.read_csv('../input/StudentsPerformance.csv')
data.head()
data['gender'].unique()
data['lunch'].unique()
data['test preparation course'].unique()
passmark = 40
data.isnull().sum()
plt.figure(figsize=(20,10))
sns.set_style(style='whitegrid')
sns.countplot(x='math score',data=data,palette='muted')
data['Math_PassStatus']=np.where(data['math score']<passmark,'F','P')
data['Math_PassStatus'].value_counts()
plt.figure(figsize=(12,6))
sns.countplot(x='parental level of education',data=data,hue='Math_PassStatus')
plt.figure(figsize=(20,10))
sns.countplot(x='reading score',data=data,palette='muted')
data['Reading_PassStatus']=np.where(data['reading score']<passmark,'F','P')
data['Reading_PassStatus'].value_counts()
plt.figure(figsize=(12,6))
sns.countplot(x='parental level of education',data=data,hue='Reading_PassStatus')
data['Writing_PassStatus']=np.where(data['writing score']<passmark,'F','P')
data['Writing_PassStatus'].value_counts()
plt.figure(figsize=(12,6))
sns.countplot(x='parental level of education',data=data,hue='Writing_PassStatus')
data['Overall_PassStatus']=data.apply(lambda x: 'F' if x['Math_PassStatus']=='F' or 
                                     x['Reading_PassStatus']=='F' or x['Writing_PassStatus']=='F' else 'P', axis=1)
data['Overall_PassStatus'].value_counts()
plt.figure(figsize=(12,6))
sns.countplot(x='parental level of education',data=data,hue='Overall_PassStatus')
data['Total Marks']=data['math score']+data['writing score']+data['reading score']
data['Percentage']=(data['Total Marks']/3)
plt.figure(figsize=(20,10))
sns.countplot(x='Percentage',data=data)
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

data['Grade'] = data.apply(lambda x : GetGrade(x['Percentage'], x['Overall_PassStatus']), axis=1)

data['Grade'].value_counts()

sns.countplot(x='Grade',order=['A','B','C','D','E','F'],data=data)
plt.figure(figsize=(12,8))
sns.countplot(x='parental level of education', data=data, hue='Grade',hue_order=['A','B','C','D','E','F'])