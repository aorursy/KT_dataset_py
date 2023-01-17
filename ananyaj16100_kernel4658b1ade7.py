import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import missingno as msno

import seaborn as sns

df = pd.read_csv('../input/insurance/insurance.csv')

print('The rows and columns in the dataset', df.shape)

print('Stastical Analysis of numeric columns', df.describe())

print('Columns in the dataset', df.columns)

print('Information about the dataset',df.info())

msno.matrix(df)

#There are no missing values 


pd.plotting.scatter_matrix(df, figsize = (15,15))

nullity = df.isnull()

print(nullity.sum())

#Checking for duplicate values

dup = df.duplicated(subset = None, keep = 'last')

print(df[dup].index)

#There is 1 duplicate row in the dataset

print(df.index)



print(df.loc[195, :])

#print(df.loc[581,:])

print('Locations 195 and 581 have duplicate values')



#Dropping duplicate values

df.drop_duplicates(inplace = True)

#We have dropped the index 581

#Unique values in the dataset

for i in list(df.columns):

    print(i + ' : ',df[i].unique())
#Correlation matrix

df.corr()

#Converting sex to categorical datatype

df['sex'] = df['sex'].astype('category')

df.dtypes

#Constructing heat map for the data frame

age_count = df.groupby(['age']).count()

#df['age'].sns(kind = 'hist', kde = False)

sns.distplot(df['age'], kde = False)

plt.xlabel('Age Interval')

a = df.groupby('sex').mean().sort_values('children')

print(a.head())

sns.set(style="whitegrid")

sns.barplot(a.index,a['children'])

plt.xlabel('Gender')
#Box plot for the dataframe

sns.set(style="whitegrid")

sns.boxplot(data = df[['age','bmi']])

sns.despine(left=True)

plt.show()

sns.boxplot(data=df['children'])

plt.show()

sns.boxplot(data= df['charges'])

plt.show()
#Average charges incurred by each age group

import seaborn as sns

a = df.groupby('age').mean().sort_values('charges', ascending = False)



#sns.set_style("dark")

sns.set_style("darkgrid", {"axes.facecolor": ".9"})

sns.lineplot(x=a.index, y=a.charges,marker = 'o', size = 5, color = 'green', label = 'line', legend = False) 

plt.xlabel('Age')

plt.ylabel('Average Charge')
#Is bmi dependent on age group

b= df.groupby('age').mean().sort_values('bmi',ascending = False)





sns.set_style("darkgrid", {"axes.facecolor": ".9"})

sns.lineplot(x=b.index, y=b.bmi,marker = 'd', size = 5, color = 'blue', label = 'line', legend = False) 

plt.xlabel('Age')

plt.ylabel('Average bmi')
#Are charges incured different for smoker and non smoker

a = df.groupby('smoker').mean().sort_values('charges')

print(a.head())

sns.set(style="darkgrid")

sns.barplot(a.index,a['charges'])

plt.xlabel('Charges')
#In which region more charges are incurred

#Are charges incured different for smoker and non smoker

a = df.groupby('region').mean().sort_values('charges')

print(a.head())

sns.set(style="darkgrid")

sns.barplot(a.index,a['charges'])

plt.xlabel('Region')
#Which region had more number of people enrolling for medical insurance

a = df.groupby('region').count().sort_values('charges')

print(a.head())

sns.set(style="whitegrid")

sns.barplot(a.index,a['charges'])

plt.xlabel('Region')

plt.ylabel('Count')
#Were there more number of smokers who enrolled for insurance

a = df.groupby('smoker').count().sort_values('charges')

print(a.head())

#sns.set(style="whitegrid")

sns.barplot(a.index,a['charges'])

plt.xlabel('Smoker')

plt.ylabel('Count')
#What is the gender of the people who smoke

a = df.groupby(['smoker', 'sex']).count()

sns.barplot(a.index, a['charges'])