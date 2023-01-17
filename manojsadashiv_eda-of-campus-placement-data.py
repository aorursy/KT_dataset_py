import numpy as np
import pandas as pd
import seaborn as sns
import warnings
import io
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
sns.set(style = "ticks", palette = "inferno_r", font_scale = 1.2)
campus_data=pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
campus_data.head()
campus_data.info()
campus_data.shape
campus_data.describe()
campus_data.isnull().sum()
plt.figure(figsize=(8,6))
sns.distplot(campus_data['salary'],kde=True,color='green')
plt.show()
plt.figure(figsize=(8,6))
sns.kdeplot(campus_data['salary'].loc[campus_data['gender']=='M'],color='blue')
sns.kdeplot(campus_data['salary'].loc[campus_data['gender']=='F'],color='red')
plt.legend(['Male','Female'])
plt.show()
campus_data.groupby('status').salary.count()
campus=campus_data.copy()
campus['salary'].loc[campus['salary'].isnull()]=0
campus['salary'].isnull().sum()
campus['salary'].describe()
cat=campus.select_dtypes(include='object').columns
cont=campus.select_dtypes(exclude='object').columns
print(cat);print(cont)
campus_data.groupby('gender').status.value_counts()
campus_data.groupby('ssc_b').status.value_counts()
campus_data.groupby('hsc_b').status.value_counts()
campus_data.groupby('hsc_s').status.value_counts()
campus_data.groupby('degree_t').status.value_counts()
fig,axes=plt.subplots(2,4,figsize=(20,10))
sns.countplot('gender',hue='status',data=campus,ax=axes[0,0])
sns.countplot('ssc_b',hue='status',data=campus,ax=axes[0,1])
sns.countplot('hsc_b',hue='status',data=campus,ax=axes[0,2])
sns.countplot('hsc_s',hue='status',data=campus,ax=axes[0,3])
sns.countplot('degree_t',hue='status',data=campus,ax=axes[1,0])
sns.countplot('workex',hue='status',data=campus,ax=axes[1,1])
sns.countplot('specialisation',hue='status',data=campus,ax=axes[1,2])
fig.delaxes(axes[1,3])
plt.show()
plt.figure(figsize=(8,6))
sns.countplot(x='gender',data=campus)
import matplotlib as mpt
mpt.rcParams['figure.figsize'] = (8.0, 6.0)
sns.countplot('gender',hue='workex',data=campus)
campus.groupby('gender').workex.value_counts()
sns.countplot('specialisation',hue='gender',data=campus)
campus.groupby('specialisation').gender.value_counts()
campus.groupby(['gender','specialisation']).status.value_counts()
sns.catplot('gender',hue='degree_t',data=campus,col='status',kind='count')
campus.groupby('hsc_b').specialisation.value_counts()
sns.catplot(x='hsc_b',hue='specialisation',data=campus,kind='count')
sns.catplot(x='degree_t',hue='specialisation',data=campus,kind='count',col='status')
sns.catplot(x='degree_t',hue='workex',data=campus,kind='count')
campus.groupby('specialisation').workex.value_counts()
sns.catplot(x='specialisation',hue='workex',data=campus,kind='count')
sns.relplot(x='ssc_p',y='hsc_p',data=campus,hue='status')
sns.relplot(x='degree_p',y='hsc_p',data=campus,hue='status')
campus_cor=campus.corr()
campus_cor.drop('sl_no',inplace=True)
campus_cor.drop('sl_no',axis=1,inplace=True)
campus_cor
campus_cor.columns
campus_pairplot=campus[['ssc_p', 'hsc_p', 'degree_p', 'etest_p', 'mba_p', 'salary','status']]
plt.figure(figsize=(25,35))
g=sns.pairplot(campus_pairplot,hue='status',palette='husl')
sns.catplot(x='gender',y='ssc_p',data=campus,kind='swarm',hue='status')
campus_DF_placed = campus[campus['status'] == 'Placed']
campus_DF_not_placed = campus[campus['status'] == 'Not Placed']
plt.figure(figsize=(10,5))
sns.kdeplot(campus_DF_placed['salary'], color = 'orange', shade = True)
plt.show()
plt.figure(figsize=(10, 5))
sns.boxenplot(x = 'salary', y = 'gender', data = campus_DF_placed, linewidth = 2.2)
plt.show()
campus_DF_placed.groupby('gender').describe()['salary']
plt.figure(figsize=(10,5))
sns.boxenplot(x = 'salary', y = 'degree_t', data = campus_DF_placed, linewidth = 2.2)
plt.show()
campus_DF_placed.groupby('degree_t').describe()['salary']
plt.figure(figsize=(10,5))
sns.boxenplot(x = 'salary', y = 'specialisation', data = campus_DF_placed, linewidth = 2.2)
plt.show()
campus_DF_placed.groupby('specialisation').describe()['salary']
