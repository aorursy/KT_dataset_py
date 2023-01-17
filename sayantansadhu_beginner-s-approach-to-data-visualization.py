# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv('/kaggle/input/all-space-missions-from-1957/Space_Corrected.csv')
data.info()
data.describe()
data.head()
data.drop(['Unnamed: 0','Unnamed: 0.1'],axis=1,inplace=True)
data['Location']
data['Country'] = data.Location.apply(lambda x : x.split()[-1])
data.head()
# percentage calculation
success = data['Status Mission'].isin(['Success']).sum()
Failure = data['Status Mission'].isin(['Failure']).sum()
Partial_failure = data['Status Mission'].isin(['Partial_failure']).sum()
Prelaunch_failure = data['Status Mission'].isin(['Prelaunch Failure']).sum()
print('The percentage of success',success/len(data['Status Mission'])*100,'%')
print('The percentage of Failure',Failure/len(data['Status Mission'])*100,'%')
print('The percentage of Partial_failure',Partial_failure/len(data['Status Mission'])*100,'%')
print('The percentage of Prelaunch Failure',Prelaunch_failure/len(data['Status Mission'])*100,'%')
plt.figure(figsize=(10,10))
labels = ['success','failure','partial failure','prelaunch_failure']
percentages = [success,Failure,Partial_failure, Prelaunch_failure]
plt.pie(percentages, labels = labels,explode=[0,0,0.1,0.1], autopct='%1.1f%%',shadow= False)
sns.countplot(data['Status Mission'])
data['Company Name'].value_counts()
ax = sns.countplot(x = 'Company Name',data=data,order=['RVSN USSR','NASA','ISRO','SpaceX','CASC'])
for p in ax.patches:
        ax.annotate(format(p.get_height()), (p.get_x()+0.2, p.get_height()))
plt.figure(figsize=(10,10))
sns.set(style="darkgrid")
ax = sns.countplot(x = 'Company Name',data=data,order=['RVSN USSR','NASA','ISRO','SpaceX','CASC'],hue= "Status Mission",)
for p in ax.patches:
        ax.annotate(format(p.get_height()), (p.get_x()-0.05, p.get_height()+20))
plt.figure(figsize=(10,10))
sns.set(style="darkgrid")
sns.countplot(y = 'Country',data=data)
plt.figure(figsize=(10,10))
sns.set(style="darkgrid")
sns.countplot(y = 'Country',data=data,order=['Russia','USA','Kazakhstan','France','China','Japan','India'],hue='Status Mission')

plt.figure(figsize=(10,10))
country = data.groupby('Country')
per_usa = len(country.get_group('USA'))
per_russia = len(country.get_group('Russia'))
per_china = len(country.get_group('China'))
per_india= len(country.get_group('India'))
per_france = len(country.get_group('France'))
per = [per_usa,per_russia,per_china,per_india,per_france]
labels = ['USA','Russia','China','India','France']
plt.figsize = (20,20)
plt.pie(per,labels= labels,explode=[0,0,0,0.1,0], autopct='%1.1f%%')
usa = country.get_group('USA')
sns.countplot(usa['Status Mission'])
plt.figure(figsize=(10,10))
success = usa['Status Mission'].isin(['Success']).sum()
failure = usa['Status Mission'].isin(['Failure']).sum()
partial_failure = usa['Status Mission'].isin(['Partial Failure']).sum()
prelaunch_failure = usa['Status Mission'].isin(['Prelaunce Failure']).sum()
outcome = [success,failure,partial_failure,prelaunch_failure]
labels = ['Success','Failure','Partial Failure','Prelaunch Failure']
plt.pie(outcome,labels= labels, autopct='%1.1f%%')
plt.title('Percentage Suceessful and failed mission by USA ')

russia = country.get_group('Russia')
sns.countplot(russia['Status Mission'])
plt.figure(figsize=(10,10))
success = russia['Status Mission'].isin(['Success']).sum()
failure = russia['Status Mission'].isin(['Failure']).sum()
partial_failure = russia['Status Mission'].isin(['Partial Failure']).sum()
prelaunch_failure = russia['Status Mission'].isin(['Prelaunce Failure']).sum()
outcome = [success,failure,partial_failure,prelaunch_failure]
labels = ['Success','Failure','Partial Failure','Prelaunch Failure']
plt.pie(outcome,labels= labels, autopct='%1.1f%%')
plt.title('Percentage Suceessful and failed mission by Russia ')
plt.figure(figsize=(5,5))
india = country.get_group('India')
sns.countplot(india['Status Mission'])
plt.figure(figsize=(10,10))
success = india['Status Mission'].isin(['Success']).sum()
failure = india['Status Mission'].isin(['Failure']).sum()
partial_failure = india['Status Mission'].isin(['Partial Failure']).sum()
prelaunch_failure = india['Status Mission'].isin(['Prelaunce Failure']).sum()
outcome = [success,failure,partial_failure,prelaunch_failure]
labels = ['Success','Failure','Partial Failure','Prelaunch Failure']
plt.pie(outcome,labels= labels, autopct='%1.1f%%')
plt.title('Percentage Suceessful and failed mission by India ')
plt.figure(figsize=(10,10))
st = data['Status Rocket'].value_counts()
plt.pie(st,shadow=False,autopct='%1.1f%%',colors=('tab:red', 'tab:blue'),explode=(0,0.05),startangle=40)
plt.legend(['Stattus Retired','Status Acitve'])
plt.title('Status Rocket', fontsize=18)
plt.show()
data['Year'] = pd.to_datetime(data['Datum']).apply(lambda year: year.year)

plt.figure(figsize=(20,10))
sns.distplot(data['Year'],bins = 83)

plt.figure(figsize=(10,20))
sns.countplot(y = data['Year'])
