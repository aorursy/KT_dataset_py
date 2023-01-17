# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.DataFrame()

df['Name'] = ['sahil','richard']

df['Score'] = [1,2]

df
# define new row

newRow = pd.Series(['Rafi',3],index = ['Name','Score'])

# add new row

df = df.append(newRow,ignore_index = True)

df
url = 'https://tinyurl.com/titanic.csv'

df = pd.read_csv("../input/titanic/train.csv")

df.describe()

df.shape
# iloc takes only numerical index

df.iloc[1:4,:]
# loc takes both numerical and text index

df.loc[:,'PassengerId']
df[df['Sex'] == 'female'].head(2)
df[(df['Sex'] == 'female') & (df['Age'] >= 50)].head(2)
df['Sex'].replace('female','woman').head(2)
# replace female with woman and male with man

df['Sex'].replace(['female','male'],['woman','man'])
df.rename(columns = {'Pclass':'Passenger Class'}).head(2)
print('minimum : ',df['Age'].min())

print('maximum : ', df['Age'].max())

print('mean : ',df['Age'].mean())

print('sum : ',df['Age'].sum())

print('count : ',df['Age'].count())

print('skew : ',df['Age'].skew())

print('variance : ',df['Age'].var())

print('mode : ',df['Age'].mode())

print('median : ',df['Age'].median())
df['Sex'].unique()
df['Sex'].value_counts()
df['Age'].isnull().sum()
df[df['Age'].isnull()].head()
df.drop(['Age'],axis=1)
df.drop([0],axis=0)
df.drop_duplicates()
df.drop_duplicates(subset=['Sex'])
df.groupby('Sex').mean()
df.groupby('Survived')['Name'].count()
df.groupby(['Sex','Survived'])['Age'].mean()
# create date range

time_index = pd.date_range('06/06/2017',periods=100000, freq='30s')

# create a dataframe

dataframe = pd.DataFrame(index=time_index)

# create column of random values

dataframe['Sale_Amount'] = np.random.randint(1,10,100000)

dataframe
# group rows by week and calculate sum per week

dataframe.resample('W').sum()
# group rows by two week and calculate sum

dataframe.resample('2W').sum()
# group rows by month and calculate sum

dataframe.resample('M').count()
df.head()

for name in df['Name'][0:2]:

    print(name.upper())
def uppercase(x):

    return x.upper()

df['Name'].apply(uppercase)[0:2]
df.groupby('Sex').apply(lambda x:x.count())
dataA = {

    'id':[1,2,3],

    'Name':['sahil','richard','rafi']

}

dfA = pd.DataFrame(dataA)

dataB = {

    'id':[4,5,6],

    'Name':['shukul','kartik','joe']

}

dfB = pd.DataFrame(dataB)



pd.concat([dfA,dfB],axis=0)
pd.concat([dfA,dfB],axis=1)
empData = {

    'emp_id':['1','2','3'],

    'name':['sahil','richard','rafi']

}

empDf = pd.DataFrame(empData)

salData = {

    'emp_id':['2','3','4'],

    'score':[3,4,5]

}

salDf = pd.DataFrame(salData)



# merging

pd.merge(empDf,salDf,on='emp_id')
pd.merge(empDf,salDf,on='emp_id',how='outer')
pd.merge(empDf,salDf,on='emp_id',how='left')
pd.merge(empDf,salDf,on='emp_id',how='right')