import pandas as pd

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sns
import os
train = pd.read_csv('/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv')
train.head()
plt.figure(figsize=(10,10)) 

sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
len(train)
train.isnull().sum()
train['SmokingStatus'].unique()
len(train['Patient'].unique())
labels = ['Ex-smoker', 'Never smoked', 'Currently smokes']

sizes = train['SmokingStatus'].value_counts()

colors = plt.cm.afmhot(np.linspace(0, 1, 5))

explode = [0.1, 0.1, 0.1,]



plt.rcParams['figure.figsize'] = (9, 9)

plt.pie(sizes, labels = labels, colors = colors,explode=explode, shadow = True)

plt.title('Distribution of Smoking Status', fontsize = 20)

plt.legend()

plt.show()
plt.style.use('seaborn-white')



sns.countplot(x='SmokingStatus',  data=train)
plt.figure(figsize=(10,10)) 



sns.countplot(x='SmokingStatus',data=train,hue='Sex')
plt.figure(figsize=(20,10)) 



sns.countplot(x='SmokingStatus',data=train,hue='Age')
sns.kdeplot(train.loc[train['SmokingStatus'] == 'Ex-smoker', 'Age'], label = 'Ex-smoker',shade=True)



sns.kdeplot(train.loc[train['SmokingStatus'] == 'Never smoked', 'Age'], label = 'Never smoked',shade=True)



# Labeling of plot

plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Smokers over Age');
plt.style.use('dark_background')



plt.rcParams['figure.figsize'] = (15, 7)

ax = sns.violinplot(x = train['SmokingStatus'], y = train['Age'], palette = 'Reds')

ax.set_xlabel(xlabel = 'Smoking habit', fontsize = 15)

ax.set_ylabel(ylabel = 'Age', fontsize = 15)

ax.set_title(label = 'Distribution of Smokers over Age', fontsize = 20)

plt.show()
plt.style.use('dark_background')



plt.rcParams['figure.figsize'] = (15, 7)

ax = sns.violinplot(x = train['SmokingStatus'], y = train['Percent'], palette = 'Reds')

ax.set_xlabel(xlabel = 'Smoking Habit', fontsize = 15)

ax.set_ylabel(ylabel = 'Percent', fontsize = 15)

ax.set_title(label = 'Distribution of Smoking Status Over Percentage', fontsize = 20)

plt.show()
plt.style.use('dark_background')

train['Age'].value_counts().head(80).plot.bar(color = 'orange', figsize = (20, 7))

plt.title('Different Ages in Data', fontsize = 30, fontweight = 20)

plt.xlabel('Different Age Groups')

plt.ylabel('count')

plt.show()
plt.style.use('seaborn-white')



sns.scatterplot(x='Age',y='Percent',data=train, color='Red')
sns.scatterplot(x='Weeks',y='Age',data=train, color = 'Black')
plt.figure(figsize=(10,7))

sns.distplot(train['Age'], color='Black',  bins = 30 )
plt.style.use('dark_background')



sns.kdeplot(train.loc[train['Sex'] == 'Male', 'Age'], label = 'Male',shade=True)



sns.kdeplot(train.loc[train['Sex'] == 'Female', 'Age'], label = 'Female',shade=True)



# Labeling of plot

plt.xlabel('Age (years)'); plt.ylabel('Density'); plt.title('Distribution of Ages over Gender');
plt.style.use('seaborn-white')

plt.rcParams['figure.figsize'] = (15, 8)

ax = sns.boxplot(x = train['Sex'], y = train['Age'], palette = 'viridis')

ax.set_xlabel(xlabel = 'Sex', fontsize = 9)

ax.set_ylabel(ylabel = 'Age', fontsize = 9)

ax.set_title(label = 'Distribution of Ages as per Sex', fontsize = 20)

plt.xticks(rotation = 90)

plt.show()
sns.lineplot(train['Age'], train['Percent'] , color = 'black')

plt.title('Age vs Percent', fontsize = 20)



plt.show()
sns.lineplot(train['Age'], train['FVC'] , color = 'black')

plt.title('Age vs FVC', fontsize = 20)



plt.show()
sns.lineplot(train['Age'], train['Weeks'] , color = 'black')

plt.title('Age vs FVC', fontsize = 20)



plt.show()
plt.figure(figsize=(15,30))

a = plt.subplot(10, 1, 1)

sns.pointplot(train.Age ,train.Percent)

plt.title("Patinets Perecent over Age" , fontsize = 25)

plt.ylabel('Percent', fontsize = 15)

plt.xlabel('Age', fontsize = 15)
plt.figure(figsize=(15,30))

a = plt.subplot(10, 1, 1)

sns.pointplot(train.Sex ,train.Age)

plt.title("Patinets Age by Sex" , fontsize = 25)

plt.ylabel('Age', fontsize = 15)

plt.xlabel('Sex', fontsize = 15)
plt.figure(figsize=(15,30))

a = plt.subplot(8, 1, 1)

sns.pointplot(train.Age ,train.FVC)

plt.title("Patinets FVC over Age" , fontsize = 25)

plt.ylabel('FVC', fontsize = 15)

plt.xlabel('Age', fontsize = 15)
sns.scatterplot(x='FVC',y='Percent',data=train, color='Black')
sns.scatterplot(x='FVC',y='Age',data=train, color ='Red')
sns.scatterplot(x='FVC',y='Weeks',data=train, color='magenta')
train.corr()['FVC'].sort_values()
plt.style.use('seaborn-white')

plt.figure(figsize=(10,7))

sns.distplot(train['FVC'], color='Blue')
sns.lineplot(train['Percent'], train['FVC'] , color = 'black')

plt.title('Percent vs FVC', fontsize = 20)



plt.show()
plt.style.use('seaborn-white')

train['Weeks'].value_counts().head(80).plot.bar(color = 'red', figsize = (25, 7))

plt.title('Number of Weeks in Data', fontsize = 50, fontweight = 20)

plt.xlabel('Weeks')

plt.ylabel('count')

plt.show()
sns.scatterplot(x='Weeks',y='Percent',data=train, color ='blue')
sns.scatterplot(x='Weeks',y='Age',data=train, color ='blue')
sns.scatterplot(x='Weeks',y='FVC',data=train, color ='blue')
plt.figure(figsize=(10,7))

sns.distplot(train['Weeks'], color='Blue')
sns.lineplot(train['Weeks'], train['Percent'] , color = 'black')

plt.title('Percent vs Week', fontsize = 20)



plt.show()
sns.countplot(x='Sex',  data=train)
plt.figure(figsize=(15,30))

a = plt.subplot(10, 1, 1)

sns.pointplot(train.Sex ,train.Percent)

plt.title("Patinets Perecent over Sex" , fontsize = 25)

plt.ylabel('Percent', fontsize = 15)

plt.xlabel('Sex', fontsize = 15)

plt.figure(figsize=(15,30))

a = plt.subplot(10, 1, 1)

sns.pointplot(train.Sex ,train.FVC)

plt.title("Patinet's FVC over Sex" , fontsize = 25)

plt.ylabel('Percent', fontsize = 15)

plt.xlabel('Sex', fontsize = 15)

plt.figure(figsize=(15,30))

a = plt.subplot(10, 1, 1)

sns.pointplot(train.Sex ,train.Weeks)

plt.title("Patinet's Weeks over Sex" , fontsize = 25)

plt.ylabel('Percent', fontsize = 15)

plt.xlabel('Sex', fontsize = 15)

plt.figure(figsize=(10,7))

sns.distplot(train['Percent'], color='Blue')
train_path = '../input/osic-pulmonary-fibrosis-progression/train'
import pydicom as dicom
fs = dicom.dcmread("../input/osic-pulmonary-fibrosis-progression/train/ID00007637202177411956430/1.dcm")
plt.imshow(fs.pixel_array) 
