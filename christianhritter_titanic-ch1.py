# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

import seaborn as sns

import matplotlib.pyplot as plt



from scipy.stats import chi2_contingency

from sklearn.impute import SimpleImputer



# Any results you write to the current directory are saved as output.
data_train_raw = pd.read_csv("/kaggle/input/titanic/train.csv")

data_test = pd.read_csv("/kaggle/input/titanic/test.csv")
data_train = data_train_raw.copy(deep = True)



#however passing by reference is convenient, because we can clean both datasets at once

data_cleaner = [data_train, data_test]
data_train.info()

data_test.info()

data_train.sample(10,random_state=42)
num_cols = ['Age','SibSp','Parch','Fare']

cat_cols = ['PassengerId','Survived','Name','Sex','Pclass','Ticket','Cabin','Embarked']
data_train.describe()
# fractions of values appearing

for col in ['Survived','Pclass','SibSp','Parch']:

    #print(col,data_train[col].unique())

    print(col,data_train[col].value_counts(normalize=True).sort_values())
# number of unique values

for col in ['PassengerId','Age','Fare']:

    print(col,data_train[col].nunique())
data_train[['Age','Fare','SibSp']].describe(percentiles=[.1, .2, .3, .4, .5, .6, .7, .8, .9, .99]).T
data_train['Fare'].value_counts().head(1)
data_train.describe(include=['O'])
# number of unique values

for col in ['Sex','Embarked']:

    print(col,data_train[col].value_counts('normalize=True'))
for col in ['Ticket','Cabin','Embarked']:

    #print(col,data_train[col].unique())

    print(col,data_train[col].value_counts().head())
data_train['Cabin'].value_counts().describe(percentiles=[.5, .6, .7, .8, .9, .99])
data_train['Embarked'].value_counts(normalize=True)
data_train[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
data_train[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
data_train[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
data_train[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(data_train, col='Survived')

g.map(plt.hist, 'Age', bins=20)
# grid = sns.FacetGrid(train_df, col='Pclass', hue='Survived')

grid = sns.FacetGrid(data_train, col='Survived', row='Pclass', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend();
grid = sns.FacetGrid(data_train, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
grid = sns.FacetGrid(data_train, row='Embarked', col='Survived', size=2.2, aspect=1.6)

grid.map(sns.barplot, 'Sex', 'Fare', alpha=.5, ci=None)

grid.add_legend()
data_train.sample(10,random_state=42)
data_train['Cabin'].value_counts(normalize=True,dropna=False).head(1)
data_train_pr = data_train.drop(['Ticket', 'PassengerId'], axis=1)

data_test_pr = data_test.drop(['Ticket', 'PassengerId'], axis=1)

combine_pr = [data_train_pr, data_test_pr]

"After", data_train_pr.shape, data_train_pr.shape
# boxplot indicating outliers in Fare and Age which could be removed.

fig, ax = plt.subplots(1, 2, figsize=(18, 4))

for var, subplot in zip(['Fare','Age'], ax.flatten()):

    sns.boxplot(y=data_test_pr[var],ax=subplot)
# extract titles

for dataset in combine_pr:

    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)



pd.crosstab(data_train_pr['Title'], data_train_pr['Sex'])
for dataset in combine_pr:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

    

data_train_pr[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
#title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "rare": 5} fillna(0)

# ordinal features do not make sense and instead use indicator variable for each title

for dataset in combine_pr:

    tmp = pd.get_dummies(dataset,prefix='Title',columns=['Title']) #, dummy_na=True)

    for col in tmp.columns:

           dataset[col] = tmp[col]

    dataset.drop(['Title','Name'],inplace=True,axis=1)

data_train_pr.head(1)
for dataset in combine_pr:

    dataset['FamFlag'] = ((data_train_pr['SibSp']>0) | (data_train_pr['Parch']>0)).astype(int)
data_train_pr[['FamFlag', 'Survived']].groupby(['FamFlag'], as_index=False).mean()
# encoding of category Sex as numerically variables

for dataset in combine_pr:

    dataset['Sex_female'] = pd.get_dummies(dataset['Sex'])['female']

    dataset.drop(['Sex'],inplace=True,axis=1)
data_train_pr.head(1)
def cab_count(val):

    try:

        return len(val.split())

    except:

        return np.nan

data_train_pr['Cabin_count'] =  data_train_pr['Cabin'].apply(cab_count)
data_train_pr['Cabin_count'].value_counts()
data_train_pr[['Cabin_count', 'Survived']].groupby(['Cabin_count'], as_index=False).mean()
# drop feature again, as there is not enough correlation or corresponding passenger numbers are too low.

data_train_pr.drop(['Cabin_count'],axis=1, inplace=True)
def extract_cabin_letter(val):

    # Extracts first letter of first cabin

    try:

        cabins = val.split()

        return cabins[0][0]

    except:

        return np.nan



for dataset in combine_pr:

    dataset['Cabin_letter'] =  dataset['Cabin'].apply(extract_cabin_letter)
data_train_pr[['Cabin_letter', 'Survived']].groupby(['Cabin_letter']).mean().join(

    data_train_pr['Cabin_letter'].value_counts().sort_index())
data_train_pr.isna().sum()
# investigating missingness of cabins

data_train_pr['Cabin_missing'] = data_train_pr['Cabin'].apply(lambda x: 1 if pd.isnull(x) else 0)
data_train_pr.corr()['Cabin_missing']
data_train_pr[['Pclass', 'Cabin_missing']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Cabin_missing', ascending=False)
g = sns.FacetGrid(data_train_pr, col='Cabin_missing')

g.map(plt.hist, 'Fare', bins=20)
data_train_pr[['Pclass', 'Cabin_letter']].groupby(['Cabin_letter']).mean().sort_values(by='Pclass', ascending=False).join(

    data_train_pr['Cabin_letter'].value_counts().sort_index())
contingency_table = pd.crosstab(data_train_pr.Pclass, data_train_pr.Cabin_missing)

contingency_table
stat, p, dof, expected_frequency_table = chi2_contingency(contingency_table)

p

# drop features not needed anymore

data_train_pr.drop(['Cabin_missing'],axis=1, inplace=True)

for dataset in combine_pr:

    dataset.drop(['Cabin'],axis=1, inplace=True)

    dataset.drop(['Cabin_letter'],axis=1, inplace=True)
data_train_pr['Age_missing'] = data_train_pr['Age'].apply(lambda x: 1 if pd.isnull(x) else 0)
data_train_pr.corr()['Age_missing']
grid = sns.FacetGrid(data_train_pr, row='Pclass', col='Sex_female', size=2.2, aspect=1.6)

grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid.add_legend()
contingency_table = pd.crosstab(data_train_pr.Pclass, data_train_pr.Age_missing)

contingency_table
stat, p, dof, expected_frequency_table = chi2_contingency(contingency_table)

p
guess_ages = np.zeros((2,3))

guess_ages
for dataset in combine_pr:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = dataset[(dataset['Sex_female'] == i) & \

                                  (dataset['Pclass'] == j+1)]['Age'].dropna()



            # age_mean = guess_df.mean()

            # age_std = guess_df.std()

            # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            dataset.loc[ (dataset.Age.isnull()) & (dataset.Sex_female == i) & (dataset.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    dataset['Age'] = dataset['Age'].astype(int)



data_train_pr.head()
guess_ages
data_train_pr.drop(['Age_missing'],axis=1, inplace=True)
data_train_pr['AgeBand'] = pd.cut(data_train_pr['Age'], 5)

data_train_pr[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
for dataset in combine_pr:    

    dataset.loc[ dataset['Age'] <= 16, 'Age_intervall'] = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age_intervall'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age_intervall'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age_intervall'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age_intervall'] = 4

data_train_pr.head()
data_train_pr.drop(['AgeBand'],axis=1, inplace=True)
imp_mode = SimpleImputer(strategy='most_frequent')
imp_mode.fit(data_train_pr[['Embarked']])
for dataset in combine_pr:    

    dataset['Embarked'] = imp_mode.transform(dataset['Embarked'].values.reshape(-1,1))
data_train_pr['Embarked'].value_counts(dropna=False)
grid = sns.FacetGrid(data_train_pr, row='Pclass', col='Sex_female', size=2.2, aspect=1.6)#,sharex=False, sharey=False)

grid.map(sns.distplot, 'Fare',kde=False, rug=False,bins=20,)#, alpha=.5, bins=20)

grid.add_legend()
data_test_pr[data_test_pr['Fare'].isna()]
one_fare_impute_median = data_train_pr[data_train_pr['Pclass']==3][data_train_pr['Sex_female']==0]['Fare'].median()

one_fare_impute_median
data_test_pr.loc[152,'Fare'] = one_fare_impute_median
data_test_pr[data_test_pr['Fare'].isna()]
data_train_pr.dtypes
for dataset in combine_pr:

    tmp = pd.get_dummies(dataset,prefix='Embarked',columns=['Embarked'])

    for col in tmp.columns:

           dataset[col] = tmp[col]

    dataset.drop(['Embarked'],inplace=True,axis=1)

data_train_pr.head(1)
data_train_pr.dtypes
data_train_pr.head(1)
#correlation heatmap of dataset

def correlation_heatmap(df):

    _ , ax = plt.subplots(figsize =(14, 12))

    colormap = sns.diverging_palette(220, 10, as_cmap = True)

    

    _ = sns.heatmap(

        df.corr(), 

        cmap = colormap,

        square=True, 

        cbar_kws={'shrink':.9 }, 

        ax=ax,

        annot=True, 

        linewidths=0.1,vmax=1.0, linecolor='white',

        annot_kws={'fontsize':12 }

    )

    plt.title('Pearson Correlation of Features', y=1.05, size=15)
#plt.figure(figsize=(15,15))

#sns.heatmap(data_train_pr.corr(),annot=True,linewidth=.5,fmt='.2f')

correlation_heatmap(data_train_pr)
pp = sns.pairplot(data_train_pr, hue = 'Survived', palette = 'deep', size=1.2, diag_kind = 'kde', diag_kws=dict(shade=True), plot_kws=dict(s=10) )

pp.set(xticklabels=[])
for dataset in combine_pr:

    dataset.info()
# guarantee same features in train and test set

all(combine_pr[0].columns.drop('Survived') == combine_pr[1].columns)
for prefix, dataset in zip(['train','test'],combine_pr):

    dataset.to_csv(prefix+'_chI.csv',index=False)