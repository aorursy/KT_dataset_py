#Import libraries

import numpy as np 

import pandas as pd 



import matplotlib.pyplot as plt

import seaborn as sns

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)
#reading data

train= pd.read_csv('/kaggle/input/hackerearth-ml-challenge-pet-adoption/train.csv')

test= pd.read_csv('/kaggle/input/hackerearth-ml-challenge-pet-adoption/test.csv')



print("Train Shape: ",train.shape)

print("Test Shape: ", test.shape)
# Check for columns

print(train.columns)

print(test.columns)
# Checking the data

train.head()
test.head()
#check for datatypes

print(train.dtypes)

print('*'*30)

print(test.dtypes)
print('Var1: Breed Category')

print(train['breed_category'].value_counts())

print()

print('Var2: Pet Category')

print(train['pet_category'].value_counts())
# train

total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data
# test

total_test = test.isnull().sum().sort_values(ascending=False)

percent_test = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)

missing_data_test = pd.concat([total_test, percent_test], axis=1, keys=['Total', 'Percent'])

missing_data_test
# Col1: pet_id

print(train.shape)

print(train.pet_id.nunique())



print()



print(test.shape)

print(test.pet_id.nunique())
train.sort_values(by=['pet_id']).head()
test.sort_values(by=['pet_id']).head()
train.sort_values(by=['issue_date']).head()
# feature engg

# getting substring from pet_id for new feature

train['nf1_pet_id'] = train['pet_id'].str[:6]

train['nf2_pet_id'] = train['pet_id'].str[:7]
# check for new feature-1

print(train.nf1_pet_id.nunique())

print(train.nf1_pet_id.value_counts())
# check for new feature-2

print(train.nf2_pet_id.nunique())

print(train.nf2_pet_id.value_counts())
train.groupby(['nf1_pet_id', 'pet_category']).size()
test['pet_id'].str[:6].value_counts()
# Col2-3: issue_data and listing_date 



#anomoly detection datetime- train

train['issue_date']= pd.to_datetime(train['issue_date'])

train['listing_date']= pd.to_datetime(train['listing_date'])



train['duration_days'] = (train['listing_date'] - train['issue_date']).dt.days

train.loc[train['listing_date'] < train['issue_date']]
#anomoly detection datetime- test

test['issue_date']= pd.to_datetime(test['issue_date'])

test['listing_date']= pd.to_datetime(test['listing_date'])

test.loc[test['listing_date'] < test['issue_date']]
# Col4: condition

train = train.fillna(-99)

test = test.fillna(-99)

print(train['condition'].value_counts())

print()

print(test['condition'].value_counts())
train.groupby(['condition','pet_category']).size()
train.columns
train.groupby(['condition','X1','X2','breed_category']).size()
# Col5: color_type

print(train['color_type'].value_counts())

print('*'*40)

print(test['color_type'].value_counts())
train.groupby(['color_type', 'pet_category']).size()
train.groupby(['color_type','breed_category']).size()
print(train['color_type'].nunique())

print(test['color_type'].nunique())
#to find which two color types not present in test

set(train.color_type) - set(test.color_type)
set(test.color_type) - set(train.color_type)
# Col6-7: length(m) and height(cm)

sns.distplot(train['length(m)'])
df=train[['length(m)','height(cm)']]

df['length(cm)'] = df['length(m)']*100

df[['length(cm)','height(cm)']].boxplot()
train.describe()
print(len(train[train['length(m)'] == 0]))

print(len(test[test['length(m)']==0]))
#convert length(m) to length(cm)

train['length(cm)'] = train['length(m)'].apply(lambda x: x*100)

test['length(cm)'] = test['length(m)'].apply(lambda x: x*100)
train.drop('length(m)', axis=1, inplace=True)

test.drop('length(m)', axis=1, inplace=True)
train[train['length(cm)']==0].groupby(['length(cm)','pet_category']).size()
test['length(cm)'].mean()
# replace all 0 length with mean of lengths

val = train['length(cm)'].mean()

train['length(cm)'] = train['length(cm)'].replace(to_replace=0, value=val)

test['length(cm)'] = test['length(cm)'].replace(to_replace=0, value=val)
# check again for 0 length

print(len(train[train['length(cm)'] == 0]))

print(len(test[test['length(cm)']==0]))
train[['length(cm)','height(cm)']].describe()
#new feature

train['ratio_len_height'] = train['length(cm)']/train['height(cm)']
#relation between ratio and pet_category

sns.catplot(x='pet_category',y='ratio_len_height',data=train)
sns.catplot(x='breed_category',y='ratio_len_height',data=train)
sns.catplot(x='pet_category',y='duration_days',data=train)
sns.boxplot(x='breed_category',y='height(cm)',data=train)
# Col8-9: X1, X2 

#X1

print(train['X1'].value_counts())

print('*'*30)

print(test['X1'].value_counts())
#X2

print(train['X2'].value_counts())

print('*'*30)

print(test['X2'].value_counts())
#correlation matrix

plt.subplots(figsize=(10,8))

sns.heatmap(train.corr(), annot= True)