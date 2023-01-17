import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



plt.style.use('seaborn')

sns.set(font_scale=2.5)



import missingno as msno



# ignore warnings

import warnings

warnings.filterwarnings('ignore')



%matplotlib inline
df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')
df_train.head()
df_train.describe()
df_train.shape
for col in df_train.columns:

    msg = "column: {:>10}\t Percent of NaN value: {:.2f}%".format(col, 100*(df_train[col].isnull().sum() / df_train[col].shape[0]))

    print(msg)
df_test.describe()
for col in df_test.columns:

    msg = "column: {:>10}\t Percent of NaN value: {:.2f}%".format(col, 100*(df_test[col].isnull().sum() / df_test[col].shape[0]))

    print(msg)
msno.matrix(df=df_train.iloc[:,:], figsize=(8,8), color=(0.8,0.5,0.2))
msno.bar(df=df_train.iloc[:,:], figsize=(8,8), color=(0.8,0.5,0.2))
f, ax = plt.subplots(1, 2, figsize=(18,8))

df_train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)

ax[0].set_title('Pie plot - Survived')

ax[0].set_ylabel('')

sns.countplot('Survived', data=df_train, ax=ax[1])

ax[1].set_title('Count plot - Survived')

plt.show()
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).count()
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).sum()
pd.crosstab(df_train["Pclass"], df_train["Survived"], margins=True).style.background_gradient(cmap='cool')
df_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean().sort_values(by='Survived',ascending=False).plot.bar()
y_position = 1.02

f, ax = plt.subplots(1, 2, figsize=(18,8))

df_train['Pclass'].value_counts().plot.bar(color=['#CD7F32', '#FFDF00', '#030303'], ax=ax[0])

ax[0].set_title("Number of passengers by PClass", y=y_position)

ax[0].set_ylabel('Count')

sns.countplot('Pclass', hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title('Pclass: Survived vs Dead', y=y_position)

plt.show()
f, ax = plt.subplots(1, 2, figsize=(18,8))

df_train[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean().plot.bar(ax=ax[0])

ax[0].set_title('Survived vs Sex')

sns.countplot('Sex', hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title('Sex: Survived vs Dead')

plt.show()
pd.crosstab(df_train['Sex'], df_train['Survived'], margins=True).style.background_gradient(cmap='cool')
sns.factorplot('Pclass', 'Survived', hue='Sex', data=df_train, size=6, aspect=1.5)
sns.factorplot(x="Sex", y="Survived", col="Pclass", data=df_train, saturation=5, size=9, aspect=1)
f, ax = plt.subplots(1, 1, figsize=(9,5))

sns.kdeplot(df_train[df_train['Survived'] == 1]['Age'], ax=ax)

sns.kdeplot(df_train[df_train['Survived'] == 0]['Age'], ax=ax)

plt.legend(['Survived == 1', 'Survived == 0'])

plt.show()
f, ax = plt.subplots(1, 1, figsize=(9,5))

df_train['Age'][df_train['Pclass'] == 1].plot(kind='kde')

df_train['Age'][df_train['Pclass'] == 2].plot(kind='kde')

df_train['Age'][df_train['Pclass'] == 3].plot(kind='kde')

plt.xlabel("Age")

plt.title("Age distribution within Pclass")

plt.legend(['1st class', '2nd class', '3rd class'])
f, ax = plt.subplots(1, 1, figsize=(9,5))

sns.kdeplot(df_train[(df_train['Survived'] == 1) & (df_train['Pclass'] == 1)]['Age'], ax=ax)

sns.kdeplot(df_train[(df_train['Survived'] == 0) & (df_train['Pclass'] == 1)]['Age'], ax=ax)

plt.legend(['Survived == 1', 'Survived == 0'])

plt.title("Age distribution in 1st class")

plt.show()
f, ax = plt.subplots(1, 1, figsize=(9,5))

sns.kdeplot(df_train[(df_train['Survived'] == 1) & (df_train['Pclass'] == 2)]['Age'], ax=ax)

sns.kdeplot(df_train[(df_train['Survived'] == 0) & (df_train['Pclass'] == 2)]['Age'], ax=ax)

plt.legend(['Survived == 1', 'Survived == 0'])

plt.title("Age distribution in 2nd class")

plt.show()
f, ax = plt.subplots(1, 1, figsize=(9,5))

sns.kdeplot(df_train[(df_train['Survived'] == 1) & (df_train['Pclass'] == 3)]['Age'], ax=ax)

sns.kdeplot(df_train[(df_train['Survived'] == 0) & (df_train['Pclass'] == 3)]['Age'], ax=ax)

plt.legend(['Survived == 1', 'Survived == 0'])

plt.title("Age distribution in 3rd class")

plt.show()
cumulative_survival_ratio = []

for i in range(1,80):

    cumulative_survival_ratio.append(df_train[df_train['Age'] < i]['Survived'].sum() / len(df_train[df_train['Age'] < i]['Survived']))

plt.figure(figsize=(7,7))

plt.plot(cumulative_survival_ratio)

plt.title('Survival rate change depending on range of Age', y=1.02)

plt.xlabel('Range of Age')

plt.show()
f, ax = plt.subplots(1, 2, figsize=(18,8))

sns.violinplot("Pclass", "Age", hue="Survived", data=df_train, scale='count', split=True, ax=ax[0])

ax[0].set_title("Pclass and Age vs Survived")

ax[0].set_yticks(range(0, 110, 10))



sns.violinplot("Sex", "Age", hue="Survived", data=df_train, scalse='count', split=True, ax=ax[1])

ax[1].set_title("Sex and Age vs Survied")

ax[1].set_yticks(range(0, 110, 10))

plt.show()
f, ax = plt.subplots(1, 1, figsize=(7,7))

df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax)
df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_values(by='Survived')
df_train[['Embarked', 'Survived']].groupby(['Embarked'], as_index=True).mean().sort_index()
f, ax = plt.subplots(2, 2, figsize=(20,15))



sns.countplot('Embarked', data=df_train, ax=ax[0,0])

ax[0, 0].set_title("(1) Number of passengers boarded")



sns.countplot("Embarked", hue='Sex', data=df_train, ax=ax[0,1])

ax[0, 1].set_title("(2) Male-Female split for embarked")



sns.countplot("Embarked", hue='Survived', data=df_train, ax=ax[1,0])

ax[1, 0].set_title("(3) Embarked vs Survived")



sns.countplot("Embarked", hue='Pclass', data=df_train, ax=ax[1,1])



ax[1,1].set_title("(4) Embarked vs Pcalss")

plt.subplots_adjust(wspace=0.5, hspace=0.5)

plt.show
df_train["FamilySize"] = df_train['SibSp'] + df_train['Parch'] + 1
f, ax = plt.subplots(1, 3, figsize=(40,10))



sns.countplot("FamilySize", data=df_train, ax=ax[0])

ax[0].set_title("(1) Number of passenger boarded", y=1.02)



sns.countplot("FamilySize", hue='Survived', data=df_train, ax=ax[1])

ax[1].set_title("(2) Survived countplot depending on FamilySize", y=1.02)



df_train[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=True).mean().sort_values(by='Survived', ascending=False).plot.bar(ax=ax[2])

ax[2].set_title("(3) Survived rate depending on FamilySize", y=1.02)



plt.subplots_adjust(wspace=0.2, hspace=0.5)

plt.show()
f, ax = plt.subplots(1, 1, figsize=(8,8))

g = sns.distplot(df_train['Fare'], color='b', label='Skewness: {:.2f}'.format(df_train['Fare'].skew()), ax=ax)

g = g.legend(loc='best')
df_train['Fare'] = df_train['Fare'].map(lambda i: np.log(i) if i>0 else 0)
f, ax = plt.subplots(1, 1, figsize=(8,8))

g = sns.distplot(df_train['Fare'], color='b', label='Skewness: {:.2f}'.format(df_train['Fare'].skew()), ax=ax)

g = g.legend(loc='best')