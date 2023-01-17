import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



data = '../input/hcc.csv'

dataHCC = pd.read_csv(data)

dataHCC.head(5)
dataHCC = dataHCC.apply(lambda x: x.replace('?',np.nan)) # replacing ? for Nan.

dataHCC.head(5)
print('(number of instances, number of attributes) ')

print(dataHCC.shape)

print('##########################')

print(dataHCC.columns) 

print('##########################')

dataHCC.info()

print('##########################')

print(dataHCC['Class'].value_counts()) #instances per class

print('##########################')

print('NaN per attribute')

print(dataHCC.isnull().sum())

dataHCC.describe(include='all')
dataHCC.drop_duplicates() #removing duplicate instances

dataHCC=dataHCC.dropna(thresh=40) #maintaining only the instances with at least 80 non-NA values, that is, 80%.  

dataHCC=dataHCC.dropna(axis=1, thresh=125) # maintaining only attributes with at least 125 non-NA values, ie 80%.

print(dataHCC.shape)

print(dataHCC.columns)

dataHCC['Class'].value_counts() 

dataHCC.isnull().sum() 
dataHCC= dataHCC.convert_objects(convert_numeric=True) # converting the data type  

dataHCC.describe(include='all')
data1=dataHCC.iloc[:,0:19]  #categorical data 1/0

cat=dataHCC.iloc[:,39]

data1['Class'] = cat

data2=dataHCC.iloc[:,19:40]   #non categorical data
data2.hist(figsize=(22,20))
data2.iloc[:,0:20].plot(kind='box', subplots=True, layout=(4,5),figsize=(14,12))
corr= dataHCC.corr()

corr
ax = sns.heatmap(corr, annot = True, cmap="YlGnBu", cbar=False)

plt.setp(ax.axes.get_xticklabels(), rotation=90)

plt.rcParams['figure.figsize']=(24,20)
corr1= data1.corr()

ax1 = sns.heatmap(corr1, annot = True, cmap="YlGnBu", cbar=False)

plt.setp(ax1.axes.get_xticklabels(), rotation=90)

plt.rcParams['figure.figsize']=(24,20)
corr2= data2.corr()

ax2 = sns.heatmap(corr2, annot = True, cmap="YlGnBu", cbar=False)

plt.setp(ax2.axes.get_xticklabels(), rotation=90)

plt.rcParams['figure.figsize']=(24,20)
dataHCC.iloc[:,0:19].mode()
x = range(19)

for n in x:

 dataHCC=dataHCC.replace({dataHCC.columns[n]: np.nan}, dataHCC.loc[:,dataHCC.columns[n]].mode().loc[0]) 
dataHCC.iloc[:,19:39].mean()
x = range(19,40)

for n in x:

 dataHCC=dataHCC.replace({dataHCC.columns[n]: np.nan}, dataHCC.loc[:,dataHCC.columns[n]].mean()) 

dataHCC.isnull().sum() 
dataHCC.info()

print('##########################')

print(dataHCC['Class'].value_counts()) #verificando quantidade de instância em cada classe

print('##########################')

dataHCC.iloc[:,19:39].describe()
dataHCC.to_csv('EDAdataHCC.csv') #saving the new clean database