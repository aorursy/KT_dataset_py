import pandas as pd

import pylab as plt

import numpy as np

plt.style.use('ggplot')
train = pd.read_csv('../input/train.csv')

train.head()
train.describe()
train.describe(include = 'all')
Survived_by_class =train[train.Survived==1].groupby('Pclass').agg({'Survived': [np.size]})

Survived_by_class.rename(columns={'size': 'Survived'}, inplace=True)

Survived_by_class
class_sizes =train.groupby('Pclass').agg({'Survived': [np.size]})

class_sizes.rename(columns={'size': 'Total'}, inplace=True)

class_sizes
percentage_survived_by_class = pd.concat([Survived_by_class, class_sizes], axis=1)



def percentage(row):

    result=(row['Survived'] /row['Total']) *100

    return result.round(decimals=1)



percentage_survived_by_class['Percentage']= percentage_survived_by_class.Survived.apply(lambda row: percentage(row), axis=1)



percentage_survived_by_class
train.Age.describe()
train2=train.copy()



train['Age'].fillna(30, inplace=True)



train.Age.isnull().sum().sum()
labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89']

train['AgeGroup'] = pd.cut(train.Age, range(0, 92, 10), right=False, labels=labels)



train.AgeGroup.describe()
age_group_all=train.groupby('AgeGroup').agg({'Survived': [np.size]})

age_group_all.rename(columns={'size': 'Total'}, inplace=True)



age_group_all
Survived=train[train['Survived'] == 1]

Survived.describe(include = 'all')
Survived=Survived.groupby('AgeGroup').agg({'Survived': [np.size]})



Survived.rename(columns={'size': 'Survived'}, inplace=True)

Survived
percentage_survived_by_age = pd.concat([Survived, age_group_all], axis=1)



percentage_survived_by_age['Percentage']= percentage_survived_by_age.Survived.apply(lambda row: percentage(row), axis=1)



percentage_survived_by_age
Survived_by_gender =train[train.Survived==1]



Survived_Gender = Survived_by_gender.groupby('Sex').size()



Survived_Gender
Survived_by_gender.Sex = Survived_by_gender.Sex.apply(lambda x: 1 if x == 'female' else 0)



Survived_by_gender.Sex.hist()
Distribution_by_sex = train.groupby('Sex').size()

Distribution_by_sex.rename(columns={'size': 'Total'}, inplace=True)

Distribution_by_sex
Survived_by_gender =train[train.Survived==1]

Survived_sex=Survived_by_gender.groupby('Sex').agg({'Survived': [np.size]})

Survived_sex.rename(columns={'size': 'Survived'}, inplace=True)

Total_Gender=train.groupby('Sex').agg({'Survived': [np.size]})

Total_Gender.rename(columns={'size': 'Total'}, inplace=True)





percentage_survived_by_gender = pd.concat([Survived_sex, Total_Gender], axis=1)

percentage_survived_by_gender['Percentage']= percentage_survived_by_gender.Survived.apply(lambda row: percentage(row), axis=1)

percentage_survived_by_gender
train.Name.head(n=100)
x=train['Name'].str.split(',', expand= True)



x
y=x[1].str.split('.', expand=True).get(0)



y.rename("Title", inplace=True)

y
df=train2.join(y)
df.groupby('Title').size()
No_NA=df.dropna(subset=['Age'])



No_NA.describe(include = 'all')
No_NA.groupby('Title').agg({'Age': [np.mean, np.median, np.min, np.max, np.size]})

Avg_Ages=No_NA.groupby('Title').agg({'Age': [np.size]})

Avg_Ages.rename(columns={'size': 'AgeAvail'}, inplace=True)

Avg_Ages
Ages= df.groupby('Title').agg({'Title': [np.size]})

Ages.rename(columns={'size': 'Total'}, inplace=True)

Ages
Age_NA = pd.concat([Avg_Ages, Ages], axis=1)

Age_NA
df.Age.fillna(df.groupby("Title")["Age"].transform("mean"), inplace=True)

df.describe(include = 'all')
z=x[0]



z.rename("Surname", inplace=True)

z
df=df.join(z)



df
df.groupby('Surname').size()