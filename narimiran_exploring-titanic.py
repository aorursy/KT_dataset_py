import numpy as np 

import pandas as pd 

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



plt.rcParams['figure.figsize'] = (12.0, 6.0)

plt.rcParams['axes.titlesize'] = 16

plt.rcParams['axes.titleweight'] = 'bold'

plt.rcParams["axes.labelsize"] = 13

plt.rcParams["axes.labelweight"] = 'bold'

plt.rcParams["xtick.labelsize"] = 12

plt.rcParams["ytick.labelsize"] = 12

sns.set_style('whitegrid')
df = pd.read_csv('../input/train.csv', index_col='PassengerId')

df.head()
df.info()
df.dropna().describe()
df.select_dtypes(include=['object']).describe()
df['Survived'].value_counts()
df['Pclass'].value_counts(sort=False)
Title_Dictionary = {

                    "Capt":       "Officer", 

                    "Col":        "Officer",

                    "Major":      "Officer", 

                    "Jonkheer":   "Royalty", 

                    "Don":        "Royalty", 

                    "Sir" :       "Royalty", 

                    "Dr":         "Officer",

                    "Rev":        "Officer", 

                    "the Countess":"Royalty", 

                    "Dona":       "Royalty", 

                    "Mme":        "Mrs",

                    "Mlle":       "Miss",

                    "Ms":         "Mrs",

                    "Mr" :        "Mr",

                    "Mrs" :       "Mrs",

                    "Miss" :      "Miss",

                    "Master" :    "Master",

                    "Lady" :      "Royalty" 

                    } 



df['Title'] = df['Name'].apply(lambda x: Title_Dictionary[x.split(',')[1].split('.')[0].strip()])
df['Age'].value_counts(dropna=False)[:20]
df['Age'].hist(bins=20)
df['SibSp'].value_counts(sort=False)
df['Parch'].value_counts(sort=False)
df['Fare'].value_counts().head(20)
df['Sex'].value_counts()
df['Ticket'].value_counts()[:20]
df['Cabin'].value_counts(dropna=False)[:20]
df['Embarked'].value_counts(dropna=False)
df['Embarked'].fillna('S', inplace=True)
sns.factorplot(x='Sex', y='Survived', data=df, kind='bar', size=5, ci=None)

plt.title('Survival Rate by Gender')
sns.factorplot(x='Pclass', y='Survived', data=df, kind='bar', size=5, ci=None)

plt.title('Survival Rate by Class')
sns.factorplot(x='Pclass', y='Survived', hue='Sex', data=df, kind='bar', size=5, aspect=1.5, ci=None)

plt.title('Survival Rate by Class and Gender')
ages_mean = df.pivot_table('Age', index=['Title'], columns=['Sex', 'Pclass'], aggfunc='mean')

ages_mean
ages_std = df.pivot_table('Age', index=['Title'], columns=['Sex', 'Pclass'], aggfunc='std')

ages_std
def age_guesser(person):

    gender = person['Sex']

    mean_age = ages_mean[gender].loc[person['Title'], person['Pclass']]

    std = ages_std[gender].loc[person['Title'], person['Pclass']]

    persons_age = np.random.randint(mean_age - std, mean_age + std)

#     persons_age = median_ages[gender].loc[person['Title'], person['Pclass']]

    return persons_age



unknown_age = df['Age'].isnull()

people_w_unknown_age = df.loc[unknown_age, ["Age", "Title", "Sex", "Pclass"]]



people_w_unknown_age['Age'] = people_w_unknown_age.apply(age_guesser, axis=1)

people_w_unknown_age.head(10)
known_age = df['Age'].notnull()

people_w_known_age = df.loc[known_age, ["Age", "Title", "Sex", "Pclass"]]



df['new_age'] = pd.concat([people_w_known_age['Age'], people_w_unknown_age['Age']])

df.head(7)
for pclass in [1, 2, 3]:

    plt.subplot(211)

    df[df['Pclass'] == pclass]['Age'].plot.kde(figsize=(12,10))

    plt.subplot(212)

    df[df['Pclass'] == pclass]['new_age'].plot.kde()

plt.suptitle('Age Density by Passenger Class', size=12)



plt.subplot(211)

plt.xlabel('Age - before filling missing values')

plt.legend(('1st Class', '2nd Class', '3rd Class'))

plt.xlim(-10,90)

plt.ylim(0, 0.05)



plt.subplot(212)

plt.xlabel('Age - values filled')

plt.legend(('1st Class', '2nd Class', '3rd Class'))

plt.xlim(-10,90)

plt.ylim(0, 0.05)
sns.regplot(x='new_age', y='Survived', data=df, x_bins=50, x_ci=None)

plt.xlim(0, None)

plt.title('Survival Rate by Age Group')
df['parent'] = 0

df.loc[(df.Parch > 0) & (df.new_age >= 18), 'parent'] = 1



df['child'] = 0

df.loc[(df.Parch > 0) & (df.new_age < 18), 'child'] = 1



df.tail(5)
df['family'] = df['SibSp'] + df['Parch']

df['family'].value_counts()
sns.factorplot(x='family', y='Survived', data=df, kind='bar', size=5, ci=None)

plt.title('Survival Rate by Family Size')
sns.factorplot(x='Sex', y='Survived', data=df, kind='bar', size=5, ci=None, hue='family')

plt.title('Survival Rate by Gender and Family Size')
sns.factorplot(x='Pclass', y='Survived', data=df, kind='bar', size=5, aspect=1.5, ci=None, hue='family')

plt.title('Survival Rate by Class and Family Size')
df.pivot_table('Survived', index=['Sex', 'Pclass'], columns=['family'], margins=True)
sns.factorplot(x='Embarked', y='Survived', data=df, kind='bar', size=5, ci=None)

plt.title('Survival Rate by Embarkment Location')
sns.countplot(x='Embarked', hue='Sex', data=df)

plt.title('Number of Passengers by Embarkment Location and Class')
sns.countplot(x='Embarked', hue='Pclass', data=df)

plt.title('Number of Passengers by Embarkment Location and Class')
sns.factorplot(x='Title', y='Survived', data=df, kind='bar', size=5, aspect=1.5, ci=None)

plt.title('Survival Rate by Title')
ax = sns.regplot(x='Fare', y='Survived', data=df, x_bins=100, x_ci=None)

ax.set(xscale="log", xlim=(1e0, 1e3))