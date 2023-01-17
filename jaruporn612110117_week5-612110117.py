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
import pandas as pd
df = pd.read_csv('/kaggle/input/titanic/titanic.txt', sep='\t', header = 0)
df
#Data Fram
df.head()
df.tail()
df.columns
df['Name']
df.iloc[2,0:5]
df.loc[2,'Passenger Class':'Age']
df.dtypes
df['Age'].min()
df[df['Age']==df['Age'].min()]
df['Name'][df['Age']==df['Age'].max()]
df.shape
df['Sex'].tolist()

df.values
df['Label'] = df['Age'] >= 20
df
df.isnull()
df.isnull().any()
df['Age'].isnull().sum()
df['Passenger Fare'].isnull().sum()
df['Cabin'].isnull().sum()
df_clean = df
df_clean = df_clean.drop('Cabin',1)
df_clean = df_clean.drop('Ticket Number',1)
df_clean.columns
df_clean = df_clean.dropna(1,'any')
df_clean = df_clean.dropna(0,'any')
df_clean.isnull().any()

df_clean['Sex'].unique()
df_clean['Sex'] = df_clean['Sex'].replace('Female','F')
df_clean['Sex'] = df_clean['Sex'].replace('Male','M')
df_clean

df_clean['Passenger Class'].unique()
p_class, levels = pd.factorize(df_clean['Passenger Class'])
p_class
levels
df_clean['Passenger Class'] = p_class
df_clean

df_clean.dtypes

df_clean['Sex'] = pd.factorize(df_clean['Sex'])[0]
df_clean['Survived'] = pd.factorize(df_clean['Survived'])[0]
df_clean.dtypes
df_clean = df_clean.set_index('Name')
df_clean
pd.crosstab(df_clean['Passenger Class'],df_clean['Survived']).plot(kind='bar')
df = pd.read_csv('https://raw.githubusercontent.com/plenoi/Clinic/master/ultima_all_clean.csv', sep=',', header = 0)
df
df_clean=df
df_clean = df_clean.set_index('hn')
df_clean
df_clean.isnull().any()
df_clean = df_clean.drop('hiv',1)
df_clean = df_clean.drop('size',1)
df_clean = df_clean.drop('utmet',1)
df_clean = df_clean.drop('vgmet',1)
df_clean = df_clean.drop('surgery',1)
df_clean = df_clean.drop('pchemo',1)
df_clean
df_clean.isnull().any()
df_clean.isnull().sum()
df_clean = df_clean.drop('RHlvsi',1)
df_clean = df_clean.drop('depth',1)
df_clean
df_clean.isnull().sum(axis=1)
df_clean

df_clean.columns
df_clean['age'] = df_clean['age'] <= 20
df_clean['age'] = df_clean['age'].replace(True,'Young')
df_clean['age'] = df_clean['age'].replace(False,'Old')
df_clean


df_clean['pmmet'].unique()
hn_pmmet, levels = pd.factorize(df_clean['pmmet'])
hn_pmmet
levels
df_clean['pmmet'] = hn_pmmet
df_clean
pmmet0=df_clean[(df_clean['pmmet']==0.0)]
pmmet0
pmmet1=df_clean[(df_clean['pmmet']==1.0)]
pmmet1
pmmet0ex=df_clean[(df_clean['pmmet']==0.0)].sample(10)
pmmet0ex
pmmet1ex=df_clean[(df_clean['pmmet']==1.0)].sample(10)
pmmet1ex
import matplotlib.pyplot as plt
import numpy as np
paritysort = np.sort(df_clean['parity'].unique())
paritysort

fig = plt.figure(1, figsize=(15,6))
parityamount = [sum(df_clean['parity'] == paritysort[0]),
        sum(df_clean['parity'] == paritysort[1]),
        sum(df_clean['parity'] == paritysort[2]),
        sum(df_clean['parity'] == paritysort[3]),
        sum(df_clean['parity'] == paritysort[4]),
        sum(df_clean['parity'] == paritysort[5]),
        sum(df_clean['parity'] == paritysort[6]),
        sum(df_clean['parity'] == paritysort[7]),
        sum(df_clean['parity'] == paritysort[8]),
        sum(df_clean['parity'] == paritysort[9]),
        sum(df_clean['parity'] == paritysort[10]),        
        sum(df_clean['parity'] == paritysort[11])]
plt.title('Total of parity')
plt.bar(paritysort,parityamount)



menopaussort = np.sort(df_clean['menopaus'].unique())
menopaussort
hnamount = df_clean.shape
fig = plt.figure(1, figsize=(15,6))
menopausamount = [sum(df_clean['menopaus'] == menopaussort[0])/hnamount[0],
       sum(df_clean['menopaus'] == menopaussort[1])/hnamount[0],
       sum(df_clean['menopaus'] == menopaussort[2])/hnamount[0]]

explode = (0, 0, 0) 
plt.subplot(1,2,1)
plt.title('proportion of menopaus')
plt.pie(menopausamount, labels=menopaussort,autopct='%1.2f%%', startangle=90, explode=explode)

plt.show()
Wardsizesort = np.sort(df_clean['Wardsize'].unique())
Wardsizesort
fig = plt.figure(1, figsize=(15,6))
Wardsizeamount = [sum(df_clean['Wardsize'] == Wardsizesort[0]),
        sum(df_clean['Wardsize'] == Wardsizesort[1]),
        sum(df_clean['Wardsize'] == Wardsizesort[2]),
        sum(df_clean['Wardsize'] == Wardsizesort[3]),
        sum(df_clean['Wardsize'] == Wardsizesort[4]),
        sum(df_clean['Wardsize'] == Wardsizesort[5]),
        sum(df_clean['Wardsize'] == Wardsizesort[6]),
        sum(df_clean['Wardsize'] == Wardsizesort[7]),
        sum(df_clean['Wardsize'] == Wardsizesort[8]),
        sum(df_clean['Wardsize'] == Wardsizesort[9]),
        sum(df_clean['Wardsize'] == Wardsizesort[10]),
        sum(df_clean['Wardsize'] == Wardsizesort[11]),
        sum(df_clean['Wardsize'] == Wardsizesort[12]),
        sum(df_clean['Wardsize'] == Wardsizesort[13]),
        sum(df_clean['Wardsize'] == Wardsizesort[14]),
        sum(df_clean['Wardsize'] == Wardsizesort[15]),
        sum(df_clean['Wardsize'] == Wardsizesort[16]),
        sum(df_clean['Wardsize'] == Wardsizesort[17]),
        sum(df_clean['Wardsize'] == Wardsizesort[18])]
plt.title('Total of each Wardsize')
labels=Wardsizesort
plt.bar(labels,Wardsizeamount)

diseasesort = np.sort(df_clean['disease'].unique())
diseasesort
hnamount = df_clean.shape
fig = plt.figure(1, figsize=(15,6))
diseaseamount = [sum(df_clean['disease'] == diseasesort[0])/hnamount[0],
       sum(df_clean['disease'] == diseasesort[1])/hnamount[0]]

explode = (0, 0) 
labels = 'No', 'Yes'
plt.subplot(1,2,1)
plt.title('proportion of disease')
plt.pie(diseaseamount, labels=labels,autopct='%1.2f%%', startangle=90, explode=explode)

plt.show()
OPDsizesort = np.sort(df_clean['OPDsize'].unique())
OPDsizesort
hnamount = df_clean.shape
fig = plt.figure(1, figsize=(15,6))

OPDsizeamount = [sum(df_clean['OPDsize'] == OPDsizesort[0])/hnamount[0],
                sum(df_clean['OPDsize'] == OPDsizesort[1])/hnamount[0],
                sum(df_clean['OPDsize'] == OPDsizesort[2])/hnamount[0],
                sum(df_clean['OPDsize'] == OPDsizesort[3])/hnamount[0],
                sum(df_clean['OPDsize'] == OPDsizesort[4])/hnamount[0],
                sum(df_clean['OPDsize'] == OPDsizesort[5])/hnamount[0],
                sum(df_clean['OPDsize'] == OPDsizesort[6])/hnamount[0],
                sum(df_clean['OPDsize'] == OPDsizesort[7])/hnamount[0],
                sum(df_clean['OPDsize'] == OPDsizesort[8])/hnamount[0],
                sum(df_clean['OPDsize'] == OPDsizesort[9])/hnamount[0],
                sum(df_clean['OPDsize'] == OPDsizesort[10])/hnamount[0],
                sum(df_clean['OPDsize'] == OPDsizesort[11])/hnamount[0],
                sum(df_clean['OPDsize'] == OPDsizesort[12])/hnamount[0],
                sum(df_clean['OPDsize'] == OPDsizesort[13])/hnamount[0],
                sum(df_clean['OPDsize'] == OPDsizesort[14])/hnamount[0],
                sum(df_clean['OPDsize'] == OPDsizesort[15])/hnamount[0],
                sum(df_clean['OPDsize'] == OPDsizesort[16])/hnamount[0],
                sum(df_clean['OPDsize'] == OPDsizesort[17])/hnamount[0],
                sum(df_clean['OPDsize'] == OPDsizesort[18])/hnamount[0],
                sum(df_clean['OPDsize'] == OPDsizesort[19])/hnamount[0]]
plt.subplot(1,2,1)
explode = (0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0) 

plt.title('proportion of OPDsize')
plt.pie(OPDsizeamount, labels=OPDsizesort,autopct='%1.2f%%', startangle=90, explode=explode)

plt.show()
