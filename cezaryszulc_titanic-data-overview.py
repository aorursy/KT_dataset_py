import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import warnings





warnings.filterwarnings("ignore")
!ls ../input/titanic
df_train = pd.read_csv('../input/titanic/train.csv', index_col='PassengerId')

df_test = pd.read_csv('../input/titanic/test.csv', index_col='PassengerId')
df_train.head(2)
df_train.shape
df_test.head(2)
df_test.shape
df_train['Pclass'].unique()
df_train['Pclass'].value_counts().sort_index().plot.bar(grid=True)

plt.title('Number of people in each class')

plt.show()
df_train.groupby(['Pclass'])['Survived'].mean()
df_train.groupby(['Pclass', 'Survived'])['Survived'].count().unstack(1).plot.bar(grid=True)

plt.title('The number of people in each class divided into survival')

plt.show()
df_train['Sex'].unique()
df_train['Sex'].value_counts().sort_index().plot.bar(grid=True)

plt.title('The number of people by gender')

plt.show()
df_train.groupby(['Sex', 'Survived'])['Survived'].count().unstack(1).plot.bar(grid=True)

plt.title('The number of people by gender divided into survival')

plt.show()
df_train.groupby(['Sex', 'Pclass'])['Survived'].mean()
df_train.groupby(['Sex', 'Survived', 'Pclass'])['Survived'].count().unstack(1).plot.bar(grid=True)

plt.title('The number of people by gender divided into survival and Pclass')

plt.show()
titles = [

    'Mr.', 'Mrs.', 'Miss.', 'Master.', 'Don.', 

    'Rev.', 'Dr.', 'Mme.', 'Ms.', 'Major.', 

    'Lady.', 'Sir.', 'Mlle.', 'Col.', 'Capt.',

    'Countess.', 'Jonkheer.'

]
df_train['Name'][~df_train['Name'].apply(lambda x: any(title in x for title in titles))]
for title in titles:

    df_train[title] = df_train['Name'].apply(lambda x: title in x)
df_train[titles].sum()
df_train.drop(titles, axis='columns', inplace=True)



updated_titles = ['Mr.', 'Mrs.', 'Miss.', 'Master.']



for title in updated_titles:

    df_train[title] = df_train['Name'].apply(lambda x: title in x)

    

df_train['others'] = ~df_train[updated_titles].any(axis='columns')
df_train.head(2)
df_train.groupby(['others']+updated_titles)['Survived'].mean()
df_train.groupby(['others']+updated_titles)['Survived'].count()
ax = df_train.groupby(['others', 'Survived']+updated_titles)['Survived'].count().unstack(1).plot.bar(grid=True)

ax.set_xticklabels((['others']+updated_titles)[::-1])

# plt.yscale('log')

plt.title('The number of people by title divided into survival')

plt.show()
df_train['Age'].describe()
df_train['Age'].plot.hist(grid=True)

plt.title('The number of people by age')

plt.show()
plt.hist(

    [

        df_train['Age'][df_train['Pclass']==1], 

        df_train['Age'][df_train['Pclass']==2], 

        df_train['Age'][df_train['Pclass']==3]

    ],

    bins=15, stacked=True)

plt.legend(['Pclass 1', 'Pclass 2', 'Pclass 3'])

plt.title('The number of people by age divided into Pclass')

plt.grid()

plt.show()
df_train['Age'][df_train['Pclass']==1].describe()
df_train['Age'][df_train['Pclass']==2].describe()
df_train['Age'][df_train['Pclass']==3].describe()
plt.hist([df_train['Age'][df_train['Survived']==0], df_train['Age'][df_train['Survived']==1]], 

         bins=15, stacked=True, color = ['#1f77b4', '#ff7f0e'])

plt.legend(['Survived 0', 'Survived 1'])

plt.title('The number of people by age divided into survival')

plt.grid()

plt.show()
df_train['SibSp'].unique()
df_train.groupby(['SibSp', 'Survived'])['Survived'].count().unstack(1).plot.bar(grid=True)

plt.yscale('log')

plt.title('The number of people by SibSp divided into Survived')

plt.show()
df_train.groupby(['SibSp', 'Survived', 'Pclass'])['Survived'].count().unstack(1).plot.bar(grid=True)

plt.yscale('log')

plt.title('The number of people by SibSp divided into Survived and Pclass')

plt.show()
df_train.groupby(['SibSp', 'Survived', 'Sex'])['Survived'].count().unstack(1).plot.bar(grid=True)

plt.yscale('log')

plt.title('The number of people by SibSp divided into Survived and Sex')

plt.show()
df_train['Parch'].unique()
df_train.groupby(['Parch', 'Survived'])['Survived'].count().unstack(1).plot.bar(grid=True)

plt.title('The number of people by Parch divided into Survived')

plt.yscale('log')

plt.show()
df_train[['SibSp', 'Parch']].corr(method='spearman')

# pearson : standard correlation coefficient

# kendall : Kendall Tau correlation coefficient

# spearman : Spearman rank correlation
df_train['Ticket'].head()
df_train['Ticket'].apply(lambda x: x.split(' ')[0]).nunique()
df_train['Ticket'].apply(lambda x: len(x.split(' '))).unique()
df_train['Ticket_segment'] = df_train['Ticket'].apply(lambda x: len(x.split(' ')))
df_train.groupby(['Ticket_segment', 'Survived'])['Survived'].count().unstack(1).plot.bar(grid=True)

plt.title('The number of people by Ticket_segment divided into Survived')

plt.show()
df_train['Fare'].describe()
df_train['Fare'].plot.hist(grid=True)

plt.title('The number of people by Fare')

plt.yscale('log')

plt.show()
plt.hist(

    [

        df_train['Fare'][df_train['Pclass']==1], 

        df_train['Fare'][df_train['Pclass']==2], 

        df_train['Fare'][df_train['Pclass']==3]

    ],

    bins=15, stacked=True)

plt.legend(['Pclass 1', 'Pclass 2', 'Pclass 3'])

plt.title('The number of people by Fare divided into Pclass')

plt.grid()

plt.show()
df_train['Fare'][df_train['Pclass']==1].describe()
df_train['Fare'][df_train['Pclass']==2].describe()
df_train['Fare'][df_train['Pclass']==3].describe()
plt.hist([df_train['Fare'][df_train['Survived']==0], df_train['Fare'][df_train['Survived']==1]], 

         bins=15, stacked=True, color = ['#1f77b4', '#ff7f0e'])

plt.legend(['Survived 0', 'Survived 1'])

plt.title('The number of people by Fare divided into survival')

plt.grid()

plt.show()
df_train['Cabin'].head()
df_train['Cabin'].isna().sum()
df_train['Cabin'][~df_train['Cabin'].isna()].apply(lambda x: x[0]).unique()
unique_cabin_char = df_train['Cabin'][~df_train['Cabin'].isna()].apply(lambda x: x[0]).unique().tolist()



for cabin_char in unique_cabin_char:

    df_train[cabin_char] = df_train['Cabin'].apply(lambda x: True if (isinstance(x, str) and x[0] == cabin_char) else False)
df_train[unique_cabin_char].sum()
df_train.groupby(['Survived']+unique_cabin_char)['Survived'].count().unstack(0)
ax = df_train.groupby(['Survived']+unique_cabin_char)['Survived'].count().unstack(0).plot.bar(grid=True)

ax.set_xticklabels((['NaN']+unique_cabin_char[::-1]))

plt.title('The number of people by Cabin first character divided into survival')

plt.legend(['Survived 0', 'Survived 1'])

plt.yscale('log')

plt.show()
update_unique_cabin_char = {'AB':['A', 'B'], 'CD': ['C', 'D'], 'EFG': ['E', 'F', 'G']}

df_train.drop(unique_cabin_char, axis='columns', inplace=True)



for keys, update_cabin_char in update_unique_cabin_char.items():

    df_train[f'Cabin_{keys}'] = df_train['Cabin'].apply(lambda x: True if (isinstance(x, str) and (x[0] in update_cabin_char)) else False)
update_unique_cabin_char = [f'Cabin_{x}' for x in list(update_unique_cabin_char.keys())]
ax = df_train.groupby(['Survived']+update_unique_cabin_char)['Survived'].count().unstack(0).plot.bar(grid=True)

ax.set_xticklabels((['NaN']+update_unique_cabin_char[::-1]))

plt.title('The number of people by Cabin first character divided into survival')

plt.yscale('log')

plt.show()
df_train['Embarked'].unique()
df_train['Embarked'].isna().sum()
df_train.groupby(['Embarked'])['Survived'].mean()
df_train.groupby(['Embarked', 'Survived'])['Survived'].count().unstack(1).plot.bar(grid=True)

plt.title('The number of people by Embarked divided into survival')

plt.show()
df_train.groupby(['Embarked', 'Survived', 'Pclass'])['Survived'].count().unstack(1).plot.bar(grid=True)

plt.title('The number of people by Embarked divided into survival and Pclass')

plt.yscale('log')

plt.show()
df_train.groupby(['Embarked', 'Survived', 'Pclass', 'Sex'])['Survived'].count().unstack(1).plot.bar(grid=True)

plt.title('The number of people by Embarked divided into survival, Pclass and Sex')

# plt.yscale('log')

plt.show()