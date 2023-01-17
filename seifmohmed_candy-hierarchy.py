# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from mpl_toolkits.mplot3d import Axes3D

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt # plotting

import numpy as np # linear algebra

import os # accessing directory structure

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import pandas as pd

import numpy as np

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
df= pd.read_excel('/kaggle/input/candy-data/candyhierarchy2017.xlsx')
df.head()
df.shape
df.iloc[:,0:50].info()
df.iloc[:,50:].info()
Q6=df.iloc[:,6:109]
df.drop(df.iloc[:,7:109],axis=1,inplace=True)
df
df.drop('Internal ID',axis=1,inplace=True)

df.drop('Unnamed: 113',axis=1,inplace=True)
df.info()
df.drop(['Q12: MEDIA [Yahoo]','Q12: MEDIA [ESPN]','Q12: MEDIA [Daily Dish]','Q9: OTHER COMMENTS','Q7: JOY OTHER','Q8: DESPAIR OTHER'],axis=1,inplace=True)
df.shape
df['Q3: AGE'].unique()
s=pd.to_numeric(df['Q3: AGE'],downcast='float',errors='ignore')
s=pd.to_numeric(s,downcast='float',errors='coerce')
df.info()
df['Q3: AGE'].unique()
df.replace(df['Q3: AGE'],s,inplace=True)
df['Q3: AGE'].replace(['old enough','45-55','24-50','?','no','Many','hahahahaha','older than dirt','Enough','See question 2','old','ancient','old enough'],np.nan,inplace=True)
df['Q3: AGE'].unique()
df['Q3: AGE'].replace(['5u','46 Halloweens.','sixty-nine','Over 50','OLD','MY NAME JEFF','59 on the day after Halloween','your mom'

                      'I can remember when Java was a cool new language', '60+'],np.nan,inplace=True)
df['Q3: AGE'].unique()
df['Q3: AGE'].replace([312,1000,'Old enough','your mom','I can remember when Java was a cool new language'],np.nan,inplace=True)
pd.to_numeric(df['Q3: AGE']).head()
df.info()
#df[['Click Coordinate(x)', 'Click Coordinate(y)']]=df['Click Coordinates (x, y)'].str.split(',', expand = True)
new = df['Click Coordinates (x, y)'].str.split(',', expand = True)
newx=new[0].str.split('(', expand = True)
newx.head()
newy=new[1].str.split(')', expand = True)
newy.head()
df.head(2)
df['Click Coordinate X']=newx[1]
df['Click Coordinate Y']=newy[0]
df.drop(columns=['Click Coordinates (x, y)'],axis=1,inplace=True)
df.head()
df.info()
df.columns.tolist()
nam=df.columns.str.split(': ').str[1]
nam
col_name = ['GOING OUT?','GENDER','AGE','COUNTRY','ADMINISTRATIVE DEFINITION','100 Grand Bar','DRESS','DAY',

            'MEDIA [Science]','Click Coordinate X','Click Coordinate Y']
df.columns=col_name
df.head()
df.info()
df['COUNTRY']= df['COUNTRY'].str.upper()

df['ADMINISTRATIVE DEFINITION']= df['ADMINISTRATIVE DEFINITION'].str.upper()

df['DRESS']= df['DRESS'].str.upper()

df['DAY']= df['DAY'].str.upper()
df.head()
df['GENDER'].unique()
df['GENDER'].replace("I'd rather not say",'Other',inplace=True)
df['COUNTRY'].unique()
df['100 Grand Bar'].unique()
df['GOING OUT?'].unique()
df.drop_duplicates(inplace=True)
df.drop(index=[0],axis=0,inplace=True)
df.reset_index(drop=True,inplace=True)
df.head()
df.shape
df['Click Coordinate X']=df['Click Coordinate X'].astype(float)
df['Click Coordinate Y']=df['Click Coordinate X'].astype(float)
df.info()
df.isnull().sum()
df['GOING OUT?'].value_counts()
df['GOING OUT?'].fillna('No',inplace=True)
df['GENDER'].value_counts()
df['GENDER'].fillna('Male',inplace=True)
df['COUNTRY'].value_counts().head()
df['COUNTRY'].fillna('USA',inplace=True)
df['ADMINISTRATIVE DEFINITION'].value_counts().head(6)
df['ADMINISTRATIVE DEFINITION'].fillna('CALIFORNIA',inplace=True)
df['AGE'].mode()
df['AGE'].fillna(40.0,inplace=True)
df.isnull().sum()
df['100 Grand Bar'].value_counts()
df['100 Grand Bar'].fillna('JOY',inplace=True)
df['DRESS'].value_counts()
df['DRESS'].fillna('WHITE AND GOLD',inplace=True)
df['DAY'].mode()
df['DAY'].fillna('FRIDAY',inplace=True)
(df['Click Coordinate X'].mode())
df['Click Coordinate X'].fillna(76.0,inplace=True)
(df['Click Coordinate Y'].mode())
df['Click Coordinate Y'].fillna(76.0,inplace=True)
df.drop(columns='MEDIA [Science]',inplace=True)
df.head()
df.info()
num=Q6.columns.str.split('|').str[1]
num
Q6.columns=num
df.shape
Q6.head()
Q6.shape
Q6.drop_duplicates(inplace=True)
#Q6.fillna(Q6.mode,inplace=True)
Q6.isnull().sum().head()
df.drop_duplicates(inplace=True)
df.reset_index(drop=True,inplace=True)
df.head()
sns.pairplot(df,hue='GENDER')
sns.pairplot(df,hue='GOING OUT?')
sns.pairplot(df,hue='100 Grand Bar')
df.head(1)
sns.countplot('GENDER',data=df)
sns.countplot('GOING OUT?',data=df)
sns.pairplot(df,hue='DRESS')
sns.pairplot(df,hue='DAY')
df.head(2)
sns.distplot(df['AGE'])
sns.distplot(df['Click Coordinate X'])
sns.distplot(df['Click Coordinate Y'])
df.head()
plt.figure(figsize=(20,10))

sns.countplot(x='COUNTRY',data=df)
df['COUNTRY'].value_counts()
df['COUNTRY'].replace(['UNITED STATES','USA','UNITED STATES OF AMERICA','US','US OF A','U.S.A.','U S A'],'USA',inplace=True)
df['COUNTRY'].value_counts().head(1)
plt.figure(figsize=(20,10))

sns.countplot(x='COUNTRY',data=df)
df.head()
df['ADMINISTRATIVE DEFINITION'].value_counts().head(10)
plt.figure(figsize=(20,10))

sns.countplot(x='ADMINISTRATIVE DEFINITION',data=df)
sns.stripplot(x="DAY", y="AGE",hue='GENDER',split=True, data=df)
sns.stripplot(x="100 Grand Bar", y="AGE",hue='GENDER',split=True, data=df)
sns.boxenplot(x="100 Grand Bar", y="AGE",hue='GENDER', data=df)
sns.countplot(x="100 Grand Bar",data=df)
df.shape
sns.countplot('100 Grand Bar',hue='GENDER',data=df)
df.head()
sns.countplot('100 Grand Bar',hue='GOING OUT?',data=df)
sns.countplot('100 Grand Bar',hue='DRESS',data=df)
sns.countplot('100 Grand Bar',hue='DAY',data=df)
g = sns.PairGrid(df,hue='GENDER')

g.map_diag(plt.hist)

g.map_upper(plt.scatter)

g.map_lower(sns.kdeplot)
def plott(x,y,df):

    sns.boxplot(x,y,data=df)
df.head(2)
plt.figure(figsize=(15,15))

plt.subplot(2,2,1)

plott('GENDER','AGE',df)

plt.subplot(2,2,2)

plott('GENDER','Click Coordinate X',df)

plt.subplot(2,2,3)

plott('GENDER','Click Coordinate Y',df)
plt.figure(figsize=(15,15))

plt.subplot(2,2,1)

plott('100 Grand Bar','AGE',df)

plt.subplot(2,2,2)

plott('100 Grand Bar','Click Coordinate X',df)

plt.subplot(2,2,3)

plott('100 Grand Bar','Click Coordinate Y',df)
plt.figure(figsize=(10,10))



plt.subplot(2,2,1)

plott('GOING OUT?','AGE',df)

plt.subplot(2,2,2)

plott('GOING OUT?','Click Coordinate X',df)

plt.subplot(2,2,3)

plott('GOING OUT?','Click Coordinate Y',df)
df.head()
plt.figure(figsize=(10,10))



plt.subplot(2,2,1)

plott('DRESS','AGE',df)

plt.subplot(2,2,2)

plott('DRESS','Click Coordinate X',df)

plt.subplot(2,2,3)

plott('DRESS','Click Coordinate Y',df)
sns.heatmap(df.corr(),cmap='coolwarm',annot=True,linecolor='white',linewidths=1)
df.drop(columns='Click Coordinate Y',inplace=True)
df.head()
def plott(x,y,h,df):

    sns.violinplot(x,y,hue=h,data=df)
plt.figure(figsize=(10,10))



plt.subplot(2,2,1)

plott('GOING OUT?','AGE','GENDER',df)

plt.subplot(2,2,2)

plott('GOING OUT?','Click Coordinate X','GENDER',df)
plt.figure(figsize=(10,10))



plt.subplot(2,2,1)

plott('GOING OUT?','AGE','100 Grand Bar',df)

plt.subplot(2,2,2)

plott('GOING OUT?','Click Coordinate X','100 Grand Bar',df)
plt.figure(figsize=(10,10))



plt.subplot(2,2,1)

plott('GOING OUT?','AGE','GOING OUT?',df)

plt.subplot(2,2,2)

plott('GOING OUT?','Click Coordinate X','GOING OUT?',df)
plt.figure(figsize=(10,10))



plt.subplot(2,2,1)

plott('GOING OUT?','AGE','DRESS',df)

plt.subplot(2,2,2)

plott('GOING OUT?','Click Coordinate X','DRESS',df)
plt.figure(figsize=(10,10))



plt.subplot(2,2,1)

plott('GOING OUT?','AGE','DAY',df)

plt.subplot(2,2,2)

plott('GOING OUT?','Click Coordinate X','DAY',df)
df['ADMINISTRATIVE DEFINITION'].value_counts().head()