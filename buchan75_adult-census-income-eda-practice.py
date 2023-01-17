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
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
%matplotlib inline
df = pd.read_csv("../input/adult.csv")
df.columns
df
#df.groupby('workclass').count()
data = df.groupby('age').count()
fig = data.plot(kind='bar',stacked=True, figsize=(12,6), sort_columns=False)
#fig.set_title('Usage (%) at certain age')
plt.show()
data = df.groupby('race').count()
fig = data.plot(kind='bar',stacked=True, figsize=(12,6), sort_columns=False)
#fig.set_title('Usage (%) at certain age')
plt.show()
data = df.groupby('sex').count()
fig = data.plot(kind='bar',stacked=True, figsize=(12,6), sort_columns=False)
#fig.set_title('Usage (%) at certain age')
plt.show()
# Six columns have some missing values '-' . Therefore, we need to replace '-' into NaN value using apply function
df['workclass'] = df['workclass'].apply(lambda x: np.nan if x == '?' else str(x))
df['occupation'] = df['occupation'].apply(lambda x: np.nan if x == '?' else str(x))
df['native.country'] = df['native.country'].apply(lambda x: np.nan if x == '?' else str(x))

df2 = df.dropna(how='all')

df2.corr()
df2
fig4 = sns.pairplot(df2)
# fig4.set_title('Pairplot')
plt.show()
fig1 = sns.distplot(df2['education.num'])
fig1.set_title('Distribution of Education number')
plt.show()
fig1 = sns.distplot(df2['age'])
fig1.set_title('Distribution of age')
plt.show()
fig1 = sns.distplot(df2['capital.gain'])
fig1.set_title('Distribution of capital gain')
plt.show()
fig1 = sns.distplot(df2['capital.loss'])
fig1.set_title('Distribution of capital gain')
plt.show()
fig1 = sns.distplot(df2['hours.per.week'])
fig1.set_title('Distribution of capital gain')
plt.show()
df.groupby('sex').count()
#data = df.groupby('education')
Fs = df2[(df2['sex'] =='Female') & (df2['income'] =='>50K')].count()
Ms = df2[(df2['sex'] =='Male') & (df2['income'] =='>50K')].count()
Fl = df2[(df2['sex'] =='Female') & (df2['income'] =='<=50K')].count()
Ml = df2[(df2['sex'] =='Male') & (df2['income'] =='<=50K')].count()

#fig = plt.plot(x, x2, kind='scatter',stacked=True, figsize=(12,6), sort_columns=False)
#movies_rank['MOVIE'].count().sort_values(ascending=False).head(10)
#plt.plot(x,x2)
Fs.income
y = [Fs.income, Fl.income, Ms.income, Ml.income]
N = len(y)
x = range(N)
width = 1/1.5

plt.xlabel(['Female small','Female large', 'Male small', 'Male large'])
plt.ylabel('Income')
plt.bar(x, y, width, color=["blue","blue","red","red"])
df2.groupby('race').count()
Ws = df2[(df2['race'] =='White') & (df2['income'] =='<=50K')].count()
Wl = df2[(df2['race'] =='White') & (df2['income'] =='>50K')].count()
Bs = df2[(df2['race'] =='Black') & (df2['income'] =='<=50K')].count()
Bl = df2[(df2['race'] =='Black') & (df2['income'] =='>50K')].count()

APs = df2[(df2['race'] =='Asian-Pac-Islander') & (df2['income'] =='<=50K')].count()
APl = df2[(df2['race'] =='Asian-Pac-Islander') & (df2['income'] =='>50K')].count()
AIs = df2[(df2['race'] =='Amer-Indian-Eskimo') & (df2['income'] =='<=50K')].count()
AIl = df2[(df2['race'] =='Amer-Indian-Eskimo') & (df2['income'] =='>50K')].count()

Os = df2[(df2['race'] =='Other') & (df2['income'] =='<=50K')].count()
Ol = df2[(df2['race'] =='Other') & (df2['income'] =='>50K')].count()

Ws_per = Ws/(Ws+Wl)*100
Bs_per = Bs/(Bs+Bl)*100
Wl_per = Wl/(Ws+Wl)*100
Bl_per = Bl/(Bs+Bl)*100

APs_per = APs/(APs+APl)*100
AIs_per = AIs/(AIs+AIl)*100
APl_per = APl/(APs+APl)*100
AIl_per = AIl/(AIs+AIl)*100

Os_per = Os/(Os+Ol)*100
Ol_per = Ol/(Os+Ol)*100

y = [Ws_per.income, Wl_per.income, Bs_per.income, Bl_per.income, APs_per.income, APl_per.income, AIs_per.income, AIl_per.income, Os_per.income, Ol_per.income]
N = len(y)
x = range(N)
width = 1/1.5
plt.figure(figsize=(16,10))
plt.xlabel(['White small','White large', 'Balck small', 'Black large', 'Asian-P small','Asian-P large','Asian-I small', 'Asian-I large','Other small','Other large'])
plt.ylabel('Income (%)')

plt.bar(x, y, width, color=["blue","blue","red","red","green","green","black","black","yellow","yellow"])
plt.show()
df2.groupby('marital.status').count()
# select more than 1000 samples case
Ws = df2[(df2['marital.status'] =='Divorced') & (df2['income'] =='<=50K')].count()
Wl = df2[(df2['marital.status'] =='Divorced') & (df2['income'] =='>50K')].count()
Bs = df2[(df2['marital.status'] =='Married-civ-spouse') & (df2['income'] =='<=50K')].count()
Bl = df2[(df2['marital.status'] =='Married-civ-spouse') & (df2['income'] =='>50K')].count()

APs = df2[(df2['marital.status'] =='Never-married') & (df2['income'] =='<=50K')].count()
APl = df2[(df2['marital.status'] =='Never-married') & (df2['income'] =='>50K')].count()
AIs = df2[(df2['marital.status'] =='Separated') & (df2['income'] =='<=50K')].count()
AIl = df2[(df2['marital.status'] =='Separated') & (df2['income'] =='>50K')].count()

#Os = data[(data['marital.status'] =='Widowed') & (data['income'] =='<=50K')].count()
#Ol = data[(data['marital.status'] =='Widowed') & (data['income'] =='>50K')].count()

Ws_per = Ws/(Ws+Wl)*100
Bs_per = Bs/(Bs+Bl)*100
Wl_per = Wl/(Ws+Wl)*100
Bl_per = Bl/(Bs+Bl)*100

APs_per = APs/(APs+APl)*100
AIs_per = AIs/(AIs+AIl)*100
APl_per = APl/(APs+APl)*100
AIl_per = AIl/(AIs+AIl)*100

y = [Ws_per.income, Wl_per.income, Bs_per.income, Bl_per.income, APs_per.income, APl_per.income, AIs_per.income, AIl_per.income]
N = len(y)
x = range(N)
width = 1/1.5
plt.figure(figsize=(16,10))
plt.xlabel(['Divorced small','Divorced large', 'Married-civ-spouse small', 'Married-civ-spouse large', 'Never-married small','Never-married large','Separated small', 'Separated large'])
plt.ylabel('Income (%)')

plt.bar(x, y, width, color=["blue","blue","red","red","green","green","black","black"])
plt.show()
df2.groupby('workclass').count()
# select more than 1000 samples case
Ws = df2[(df2['workclass'] =='nan') & (df2['income'] =='<=50K')].count()
Wl = df2[(df2['workclass'] =='nan') & (df2['income'] =='>50K')].count()
Bs = df2[(df2['workclass'] =='Local-gov') & (df2['income'] =='<=50K')].count()
Bl = df2[(df2['workclass'] =='Local-gov') & (df2['income'] =='>50K')].count()

APs = df2[(df2['workclass'] =='Private') & (df2['income'] =='<=50K')].count()
APl = df2[(df2['workclass'] =='Private') & (df2['income'] =='>50K')].count()
AIs = df2[(df2['workclass'] =='Self-emp-inc') & (df2['income'] =='<=50K')].count()
AIl = df2[(df2['workclass'] =='Self-emp-inc') & (df2['income'] =='>50K')].count()

Os = df2[(df2['workclass'] =='Self-emp-not-inc') & (df2['income'] =='<=50K')].count()
Ol = df2[(df2['workclass'] =='Self-emp-not-inc') & (df2['income'] =='>50K')].count()

Ss = df2[(df2['workclass'] =='State-gov') & (df2['income'] =='<=50K')].count()
Sl = df2[(df2['workclass'] =='State-gov') & (df2['income'] =='>50K')].count()

Ws_per = Ws/(Ws+Wl)*100
Wl_per = Wl/(Ws+Wl)*100
Bs_per = Bs/(Bs+Bl)*100
Bl_per = Bl/(Bs+Bl)*100

APs_per = APs/(APs+APl)*100
APl_per = APl/(APs+APl)*100
AIs_per = AIs/(AIs+AIl)*100
AIl_per = AIl/(AIs+AIl)*100

Os_per = Os/(Os+Ol)*100
Ol_per = Ol/(Os+Ol)*100

Ss_per = Ss/(Ss+Sl)*100
Sl_per = Sl/(Ss+Sl)*100

#fig = plt.plot(x, x2, kind='scatter',stacked=True, figsize=(12,6), sort_columns=False)
#movies_rank['MOVIE'].count().sort_values(ascending=False).head(10)
#plt.plot(x,x2)
# Fs.income
y = [Ws_per.income, Wl_per.income, Bs_per.income, Bl_per.income, APs_per.income, APl_per.income, AIs_per.income, AIl_per.income, Os_per.income, Ol_per.income, Ss_per.income, Sl_per.income]
N = len(y)
x = range(N)
width = 1/1.5
plt.figure(figsize=(20,10))
plt.xlabel(['NaN small','NaN-gov large', 'Local-gov small', 'Local-gov large', 'Private small','Private large','Self-emp-inc small', 'Self-emp-inc large','Self-emp-not-inc small','Self-emp-not-inc large', 'State-gov small', 'State-gov large'])
plt.ylabel('Income (%)')

plt.bar(x, y, width, color=["blue","blue","red","red","green","green","black","black","yellow","yellow","pink","pink"])
plt.show()
