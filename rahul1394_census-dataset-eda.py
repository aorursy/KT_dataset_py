import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.preprocessing import LabelEncoder
censusData = pd.read_csv('../input/adult.csv')

censusData.head()
censusData.shape
censusData.replace({'?':np.nan},inplace=True)
censusData.info()
censusData.isnull().sum()
censusData.describe()   
censusData.describe(include=np.object)
censusData['workclass'].fillna(censusData['workclass'].mode().values[0],inplace=True)

censusData['occupation'].fillna(censusData['occupation'].mode().values[0],inplace=True)

censusData['native-country'].fillna(censusData['native-country'].mode().values[0],inplace=True)
censusData.info()
censusData.head()
plt.figure(figsize=(15,5))

sns.distplot(censusData['age'],kde=False)

plt.show()
(censusData.age>70).sum()       # right skewed but data > 70 is less person so not an issue 
plt.figure(figsize=(15,6))

sns.distplot(censusData['hours-per-week'],kde=False,color='r')

plt.show()



''' 

    3rd quantile is 45 i.e 75% spend less than 45 hr/week.

    majority population spend b/w 40-45 hr/wk

    '''
plt.figure(figsize=(15,6))

sns.distplot(censusData['capital-gain'],kde=False,color='b',bins=15)

plt.show()



''' it shows either no gain or high gain due to high deviation'''
print(censusData[['capital-gain','capital-loss']].corr())



plt.figure(figsize=(8,4))

sns.scatterplot(censusData['capital-gain'],censusData['capital-loss'])

plt.show()



'''

capital gain and loss have negative weak relationship.

If one is 0 ,another is high

'''
plt.figure(figsize=(15,6))



sns.countplot(censusData['workclass'])

plt.show()



'''

most people working in private sector.

Huge imbalance in data.

'''
plt.figure(figsize=(18,10))



ax = sns.countplot(censusData['education'])



for p in ax.patches:

    

    x = p.get_x()+p.get_width()/2 - 0.1

    y = p.get_height() + 3

    t = round((p.get_height()/censusData.shape[0])*100)

    

    ax.text(x,y,t)

    

plt.show()



'''

most people are HS-grad(32%) followed by some college and bachelors.



'''
sns.countplot(censusData['marital-status'])

plt.xticks(rotation = 90)

plt.show()



'''

majority is married with civilian spouse having highest followed by never married.

'''
sns.countplot(censusData['occupation'])

plt.xticks(rotation = 90)

plt.show()



'''

Prof-specialty has the maximum count(8981) but Craft-repair, Exec-managerial and Adm-clerical Sales has comparable number of observations.

Armed-Forces has minimum samples in the occupation attribute.

'''
sns.countplot(censusData['gender'])

plt.show()

'''

males are in majority

'''
sns.countplot(censusData['relationship'])

plt.xticks(rotation = 90)

plt.show()



'''

husband have highest frequency

'''
plt.figure(figsize=(12,5))

sns.countplot(censusData['native-country'])

plt.xticks(rotation = 90)

plt.show()



'''

US have highest frequency, that means data is of country US.

'''
censusData['income'] = LabelEncoder().fit_transform((censusData['income']))
censusData.head()
sns.countplot(censusData['income'])

plt.show()

'''

people with income less than 50k are higher.

'''
sns.boxplot(x='income',y='age',data=censusData)

plt.show()

'''

avg people's age is 44 approx. for income > 50k which is more compared to income <= 50k

'''
plt.figure(figsize=(12,5))

sns.boxplot(x='income',y='hours-per-week',data=censusData)

plt.show()



''' income >50k have longer avg working hrs than <50k

working hr range is also higher'''
plt.figure(figsize=(12,5))

sns.boxplot(x='income',y='fnlwgt',data=censusData)

plt.show()



''' weight has no significance on income'''
plt.figure(figsize=(12,5))

sns.countplot(x='workclass',hue='income',data=censusData)

plt.show()



''' private working class have highest no. of people >50k

however conversion ratio is minimal in private while in self-emp-inc have very high conversion ratio'''
plt.figure(figsize=(12,5))

sns.countplot(x='education',hue='income',data=censusData)

plt.xticks(rotation = 90)

plt.show()



''' higher education have better chance of earning >50k'''
plt.figure(figsize=(8,4))

sns.countplot(x='gender',hue='income',data=censusData)

plt.xticks(rotation = 90)

plt.show()



''' females are more likely to be earning <= 50k'''
dd = pd.crosstab(censusData['occupation'],censusData['income'])

dd.plot.bar(stacked=True,figsize=(12,5))

plt.xticks(rotation = 90)

plt.show()



''' In every occupation, people who earn less than 50k is greater than people who earn >50k.

'''
plt.figure(figsize=(12,5))

sns.countplot(x='native-country',hue='income',data=censusData)

plt.xticks(rotation = 90)

plt.show()



'''country has negligible effect on income'''
plt.figure(figsize=(12,5))

sns.boxplot(censusData['income'],censusData['hours-per-week'],hue=censusData['gender'])

plt.show()

'''average working hour of female is less compared to males.

However for income group >50k mens have more flexible

working hrs compared to females. '''
plt.figure(figsize=(12,5))

sns.boxplot(censusData['income'],censusData['age'],hue=censusData['gender'])

plt.show()

'''average age of female is less compared to males for income group >50k.

However range is less for men compared to females for group <=50k. '''
plt.figure(figsize=(12,5))

sns.boxplot(censusData['workclass'],censusData['age'],hue=censusData['gender'])

plt.show()

'''average age of private and gov dept is less for income group >50k compared to group <=50k. '''
censusData['capital_range'] = censusData['capital-gain'] - censusData['capital-loss']

censusData.head()
sns.distplot(censusData['capital_range'],kde=False)

plt.show()



'''capital range has same plot as capital gain/loss so we can replace capital gain and loss with one column.'''
plt.figure(figsize=(12,5))

sns.heatmap(censusData.corr(),annot=True)

plt.show()