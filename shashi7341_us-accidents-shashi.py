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
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns
data=pd.read_csv("/kaggle/input/us-accidents/US_Accidents_May19.csv")
data.head()
print(data.columns)
data.shape
data.info()
data.describe()
data.isnull().values.any()
null=data.isnull().sum()

null
fig=plt.figure(figsize=(20,4))

null.plot(kind='bar',color='green')

plt.title('List of columns and there Missing Values Count')
percent_null_values=null/len(data)*100

percent_null_values
lis=percent_null_values

for i in range(len(lis)):

    if lis[i]>60:

        del data[lis.index[i]]
data.shape
fig1=plt.figure(figsize=(20,4))

data.isnull().sum().plot(kind='bar',color='red')

plt.title('List of Features after removing Missing Values above 60%')
print(data.iloc[:,0:10].info())

print(data.iloc[:,0:10].nunique())
#Source column

data.Source.value_counts(dropna=False)
sns.countplot(data['Source'])
#TMC(Traffic Message Channel)

data.TMC.value_counts(dropna=False)
plt.figure(figsize=(10,4))

data.TMC.value_counts(dropna=False).plot(kind='bar',color='orange')
#Severity 

data.Severity.value_counts(dropna=False)
plt.figure(figsize=(10,4))

sns.countplot(data['Severity'])
print(data.Start_Time.head())

print(data.End_Time.head())
data['Start_Time']=pd.to_datetime(data['Start_Time'],errors='coerce')

data['Start_Time'].head()
data['End_Time']=pd.to_datetime(data['End_Time'],errors='coerce')

data['End_Time'].head()
data['SYear']=data['Start_Time'].dt.year

data['SMonth']=data['Start_Time'].dt.strftime('%b')

data['SDay']=data['Start_Time'].dt.day

data['SHour']=data['Start_Time'].dt.hour

data['SWeekday']=data['Start_Time'].dt.strftime('%a')
data['EYear']=data['End_Time'].dt.year

data['EMonth']=data['End_Time'].dt.strftime('%b')

data['EDay']=data['End_Time'].dt.day

data['EHour']=data['End_Time'].dt.hour

data['EWeekday']=data['End_Time'].dt.strftime('%a')
td='Time_Duration(min)'

data[td]=round((data['End_Time']-data['Start_Time'])/np.timedelta64(1,'m'))

data[td].head(20)
data.info()
data[td][data[td]<=0]
neg_values=data[td]<=0



#set negative values with NAN

data[neg_values]=np.nan
data[neg_values]
data.dropna(subset=[td],axis=0,inplace=True)
data.shape
data['Time_Duration(min)'].nunique()
plt.figure(figsize=(16,8))

sns.boxplot(data['Time_Duration(min)'])
data.Start_Lat.nunique()
##Start_Lat

plt.figure(figsize=(10,4))

plt.hist(data['Start_Lat'])
##Start_Lng

plt.figure(figsize=(10,4))

plt.hist(data['Start_Lng'])
#Distance in miles

data['Distance(mi)'].nunique()
plt.figure(figsize=(10,4))

plt.hist(data['Distance(mi)'])
#Description

data['Description'].nunique()
data.isnull().sum()
#Street

data['Street'].nunique()
#Side

data['Side'].value_counts()
sns.countplot(data['Side'])
#City

data.City.nunique()
#Country

data['Country'].value_counts()
print(data['State'].nunique())

print(data['State'].value_counts())
plt.figure(figsize=(16,8))

sns.countplot(data['State'],palette='Set2')
data.isnull().sum()
##Missing Values Imputation with Most frequently occuring values 

def fillna(col):

    col.fillna(col.value_counts().index[0],axis=0,inplace=True)

    return col

data=data.apply(lambda col: fillna(col))
data.isnull().sum()
data.head()
plt.figure(figsize=(16,16))

sns.heatmap(data.corr(),annot=True,cmap='magma')