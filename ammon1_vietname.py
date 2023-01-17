import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
df=pd.read_csv('../input/VIETNAM NATIONAL HIGHSCHOOL EXAM SCORE 2018.csv')

df.info()
df=df.drop(['Unnamed: 11'],axis=1)
df.KhoiD=df.KhoiD.convert_objects(convert_numeric=True)
df.info()
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(20)
df1=df.dropna() 
df1.info()
import matplotlib.pyplot as plt
import seaborn as sns
f, ax = plt.subplots(figsize=(10, 8))
df2=df1[['Math','Physics','Chemistry','Biology','English','Viet','History','Geography','GDCD']]
corr = df2.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

x0 = df.Math.dropna().values
x1 = df.Viet.dropna().values
x2 = df.English.dropna().values
x3 = df.Physics.dropna().values
x4 = df.Chemistry.dropna().values
x5 = df.Biology.dropna().values
x6 = df.History.dropna().values
x7 = df.Geography.dropna().values
x8 = df.GDCD.dropna().values
fig = plt.figure(figsize=(20, 40))
ax0 = fig.add_subplot(911)
plt.xlabel("Math")
ax1 = fig.add_subplot(912)
plt.xlabel("Viet")
ax2 = fig.add_subplot(913)
plt.xlabel("English")
ax3 = fig.add_subplot(914)
plt.xlabel("Physics")
ax4 = fig.add_subplot(915)
plt.xlabel("Chemistry")
ax5 = fig.add_subplot(916)
plt.xlabel("Biology")
ax6 = fig.add_subplot(917)
plt.xlabel("History")
ax7 = fig.add_subplot(918)
plt.xlabel("Geoghaphy")
ax8 = fig.add_subplot(919)
plt.xlabel("GDCA")
sns.distplot(x0,ax=ax0)
sns.distplot(x1,ax=ax1)
sns.distplot(x2,ax=ax2)
sns.distplot(x3,ax=ax3)
sns.distplot(x4,ax=ax4)
sns.distplot(x5,ax=ax5)
sns.distplot(x6,ax=ax6)
sns.distplot(x7,ax=ax7)
sns.distplot(x8,ax=ax8)
df1=df[(df.Physics>5)&(df.Biology>5)&(df.History>5)]
df.describe()
df1.describe()
print(df.shape)
print(df1.shape)
print ('Math ',100*df[df.Math>5].shape[0]/df.shape[0],'%')
print ('Viet ',100*df[df.Viet>5].shape[0]/df.shape[0],'%')
print ('English ',100*df[df.English>5].shape[0]/df.shape[0],'%')
print ('Physics ',100*df[df.Physics>5].shape[0]/df.shape[0],'%')
print ('Chemistry ',100*df[df.Chemistry>5].shape[0]/df.shape[0],'%')
print ('Biology ',100*df[df.Biology>5].shape[0]/df.shape[0],'%')
print ('History ',100*df[df.History>5].shape[0]/df.shape[0],'%')
print ('Geography ',100*df[df.Geography>5].shape[0]/df.shape[0],'%')
print ('GDCD ',100*df[df.GDCD>5].shape[0]/df.shape[0],'%')