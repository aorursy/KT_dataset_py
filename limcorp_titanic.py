!pip install https://github.com/pandas-profiling/pandas-profiling/archive/master.zip
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_profiling import ProfileReport


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')
train_y = train[['Survived']]
train_x = train.drop('Survived', axis=1)
train.head(10)
train.tail(10)
train.info()
train.columns
train['PassengerId']
for column in train.columns:
    print(column)
    print(train[column].nunique())
    print("======================")
train.describe(include='O')
train.describe()
print(train.corr())
sns.heatmap(train.corr())
profile = ProfileReport(train, title='Pandas Profiling Report', explorative=True)
profile
# Let's see if we can check whether the attribute as a women
# influenced the survival rate.

print("Ratio of survival within each group (Female, Male)")
print(train[['Sex', 'Survived']].groupby(['Sex']).mean())
print("\nRatio of Female and Male")
print(train.groupby('Sex')['Survived'].value_counts())


sns.barplot(x="Sex", y="Survived", data=train)
print(train['Embarked'].value_counts())
sns.barplot(x="Embarked", y="Survived", data=train)
# Create Bins and convert numerical data into a categorical data.
bins = [0, 2, 18, 35, 65, np.inf]
names = ['<2', '2-18', '18-35', '35-65', '65+']

train['Age_band'] = pd.cut(train['Age'], bins, labels=names)

print("Ratio of survival within each group")
print(train[['Age_band', 'Survived']].groupby('Age_band').mean())


print("Break down of survival within each group")
print(train.groupby('Age_band')['Survived'].value_counts())
train.drop('Age_band', axis=1, inplace=True)
# Let's not group people into arbitrary divisions,
# instead, let me look at the distributions of
# those who survived, and those who didn't.

g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=30)
# Make equal-ranged age bins and compare the survival
train['Age_band_v2'] = pd.cut(train['Age'], bins=30)
survival_by_age = train.groupby('Age_band_v2', observed=True)['Survived'].value_counts()


survival_by_age_df = pd.DataFrame(survival_by_age)
survival_by_age_df.rename(columns={'Survived':'Count'}, inplace=True)
survival_by_age_df = survival_by_age_df.sort_values(by=['Age_band_v2', 'Survived'],
                                                    ascending=[True, True])
print(survival_by_age_df)
train.drop('Age_band_v2', axis=1, inplace=True)
# Let's look at the distribution of Pclass among train samples.

sns.barplot(x="Pclass", y="Survived", data=train)
train.head()
# Relationship between Fare and Pclass (Some might say this relationship is too obvious, but I think we should check every single relationship, so as not to miss one.)

sns.barplot(x="Pclass", y="Fare", data=train)
grid = sns.FacetGrid(train, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
train['#Family'] = train['SibSp'] + train['Parch']
test['#Family'] = test['SibSp'] + test['Parch']

train['#Family'].value_counts()
sns.barplot(x="#Family", y="Survived", data=train)
grid = sns.FacetGrid(train, size=2.2, aspect=1.6)
grid.map(sns.pointplot, '#Family', 'Survived', 'Pclass', palette='deep')
grid.add_legend()
train['CabinBool'] = train['Cabin'].notnull()
train['CabinBool'].corr(train['Pclass'])
sns.barplot(x="CabinBool", y="Survived", data=train)
corr = train.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(7, 5))
    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True)
train.head()
train['Name'].head()
train['LastName'] = train['Name'].apply(lambda x: x.split(', ')[0])
train.head(5)
train.LastName
# Can we match the LastName and find siblings so that we can have a hope that we would know the age with "SibSp" and "Parch"?

train[train['LastName'].isin(['Braund'])]
# Let's do some excercise about family members.
# Can you group two families from those who have the same last name, "Andersson"?
# They were all class 3, only two survived.

for lastname in train['LastName'].value_counts()[:20].index:
    print(train[train['LastName'] == lastname])
# Take out titles from the name

train['title'] = train['Name'].apply(lambda x: x.split(', ')[1].split('. ')[0])
test['title'] = test['Name'].apply(lambda x: x.split(', ')[1].split('. ')[0])
train.title
unique_title_list = list(set(train['title'].unique())|set(test['title'].unique()))
print(unique_title_list)
train.title.value_counts()
def tidy_title(dataset):
    dataset['title'].replace(['Lady', 'Countess', 'the Countess', 'Capt', 'Col','Don', 'Dr',\
                              'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare', inplace=True)
    dataset['title'].replace('Mlle', 'Miss', inplace=True)
    dataset['title'].replace('Ms', 'Miss', inplace=True)
    dataset['title'].replace('Mme', 'Mrs', inplace=True)
    return dataset

train = tidy_title(train)
test = tidy_title(test)
train['title'].unique()
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Fare', bins=10)
train['Fare'].corr(train['CabinBool'])
train['Fare'].corr(train['Pclass'])
variation = []

for name in train['LastName'].unique():
    subgroup = train[train['LastName'] == name]
    num_subgroup_members = len(subgroup)
    if num_subgroup_members == 1:
        continue
    num_ticket_type = subgroup['Ticket'].nunique()
    variation_by_lastname = num_ticket_type/num_subgroup_members*100 
    variation.append(variation_by_lastname)
    
    print("-"*10)
    print(subgroup['Ticket'].unique())
    print(num_ticket_type)
    print(num_subgroup_members)
    print(f'{name}: {variation_by_lastname:.2f}%')
plt.hist(variation)
# Masters' age distribution

bool_indexing = (train['title'] == 'Master') & (train['Age'].notnull())
master_median_age = train.loc[bool_indexing, 'Age'].median()
sns.distplot(train.loc[bool_indexing, 'Age'])
print(f'master median age: {master_median_age}')
# Miss' age distribution

bool_indexing = (train['title'] == 'Miss') & (train['Age'].notnull())
master_median_age = train.loc[bool_indexing, 'Age'].median()
sns.distplot(train.loc[bool_indexing, 'Age'])
print(f'master median age: {master_median_age}')
sns.distplot(train['#Family'], kde=False)
def distribution_by_pclass(dataset):
    # Only include data with age value
    data = dataset[dataset['Age'].notnull()]
    
    # Group number of family over 3 to be '>3'
    data['new_#fam'] = np.where(data['#Family']>=3, '>3', data['#Family'])

    # Draw Facet grid according to the Family size for the given title's population with age value.
    # I set hue to be Pclass to discover the possible differences among Passenger classes.
    g = sns.FacetGrid(data, col='Pclass', hue='new_#fam', height=3.5, col_wrap=3)
    g.map(plt.hist, 'Age', bins=10)


    # Add super title to the sns.FaceGrid
    axes = g.axes.flatten()
    g.add_legend()


    # Change title for each graph
    entire_count = data.shape[0]
    sorted_unique_list = np.sort(data['Pclass'].unique())
    for index, pclass in enumerate(sorted_unique_list):
        sub_group = data[data['Pclass'] == pclass]
        subgroup_ratio = sub_group.shape[0]/entire_count * 100
        median_age = sub_group.loc[:, 'Age'].median()
        axes[index].set_title(f'Pclass={pclass} ({subgroup_ratio:.2f}%) : {median_age}')
    
    # drop added column
    data.drop('new_#fam', axis=1, inplace=True)
distribution_by_pclass(train)
distribution_by_pclass(test)
def distribution_by_title(dataset, title):
    # Specific title's age distribution
    miss_pop_index = (dataset['title'] == title) & (dataset['Age'].notnull())
    miss_pop = dataset[miss_pop_index]

    # Draw Facet grid according to the Family size for the given title's population with age value.
    # I set hue to be Pclass to discover the possible differences among Passenger classes.
    g = sns.FacetGrid(miss_pop, col='#Family', hue='Pclass', height=3.5, col_wrap=3)  
    g.map(sns.distplot, 'Age', kde=False, axlabel=True)


    # Add super title to the sns.FaceGrid
    axes = g.axes.flatten()
    g.fig.suptitle(f'Median age of {title} by family size')
    g.fig.subplots_adjust(top=0.86)
    g.add_legend()


    # Change title for each graph
    entire_count = miss_pop.shape[0]
    sorted_unique_list = np.sort(miss_pop['#Family'].unique())
    for index, fam_num in enumerate(sorted_unique_list):    
        bool_indexing = (miss_pop['#Family'] == fam_num)
        sub_group = miss_pop[bool_indexing]
        subgroup_ratio = sub_group.shape[0]/entire_count * 100
        median_age = sub_group.loc[:, 'Age'].median()
        axes[index].set_title(f'#Family={fam_num} ({subgroup_ratio:.2f}%) : {median_age}')
distribution_by_title(train, 'Miss')
distribution_by_title(test, 'Miss')
distribution_by_title(train, 'Master')
distribution_by_title(train, 'Mr')
distribution_by_title(train, 'Mrs')
distribution_by_title(train, 'Rare')
def map_categ_values(dataset, col_name, dictionary, new_col_name=None, drop=True, categ=True):
    if new_col_name is not None:
        dataset[new_col_name] = dataset[col_name].map(dictionary)
        if drop:
            dataset.drop(col_name, axis=1, inplace=True)
        col_name = new_col_name
    else:
        dataset[col_name] = dataset[col_name].map(dictionary)
    
    if categ:
        dataset[col_name] = dataset[col_name].astype('category')
    
    return dataset

# map numbers to the title categories
title_dict = {'Mr': 1, 'Mrs': 2, 'Miss': 3, 'Master':4, 'Rare':5}
train = map_categ_values(train, 'title', title_dict)
test = map_categ_values(test, 'title', title_dict)

# map numbers to the Sex
sex_dict = {'male': 1, 'female': 2}
train = map_categ_values(train, 'Sex', sex_dict, categ=False)
test = map_categ_values(test, 'Sex', sex_dict, categ=False)
train.head()
from sklearn.cluster import KMeans

km = KMeans(n_clusters=7, 
            init='k-means++', 
            max_iter=10000, 
            tol=1e-04,
            random_state=0).fit(train[['title', 'Pclass', 'Sex']])
train['group'] = km.predict(train[['title', 'Pclass', 'Sex']])    # family number를 넣어서 또 해보자.
test['group'] = km.predict(test[['title', 'Pclass', 'Sex']])
group_summary = train.groupby('group').median()
train['group'].value_counts()
group_summary
group_summary['Age'].to_dict()
group_age_average = group_summary['Age'].to_dict()

# Impute Age column of the train data with average values for each group
mask = train['Age'].isnull()
train.loc[mask, 'Age_strategy1'] = train['group'].map(group_age_average)
train.loc[~mask, 'Age_strategy1'] = train['Age']

# Now impute Age column the same as train data's Age column.
mask = test['Age'].isnull()
test.loc[mask, 'Age_strategy1'] = test['group'].map(group_age_average)
test.loc[~mask, 'Age_strategy1'] = test['Age']
train.head()
train.drop('Age_strategy1', axis=1, inplace=True)
test.drop('Age_strategy1', axis=1, inplace=True)
train.head()
def age_strategy_with_kmeans(dataset, used_cols, num_clusters=None,
                             model=None, group_summary=None, pca=False, inplace=True):
    # deep copy a dataset
    data = dataset.copy()
    
    # get model for train dataset.
    if not model:
        model = KMeans(n_clusters=num_clusters, 
                       init='k-means++', 
                       max_iter=10000, 
                       tol=1e-04,
                       random_state=0).fit(data[used_cols])
        
    data['group'] = model.predict(data[used_cols])

    # get group summary for train dataset.
    if group_summary is None:
        group_summary = data.groupby('group').mean()
        
    # convert NaN values in the Age column
    group_age_average = group_summary['Age'].to_dict()
    mask = data['Age'].isnull()
    
    if inplace:
        col_name = 'Age'
    else:
        col_name = 'Age_strategy'
         
    data.loc[mask, col_name] = data['group'].map(group_age_average)
    data.loc[~mask, col_name] = data['Age']
    return data, model, group_summary


# stategy_1
dataset1_train, model1, group_summary1 = age_strategy_with_kmeans(train, ['title', 'Pclass', 'Sex'], 7)

# strategy_2
dataset2_train, model2, group_summary2 = age_strategy_with_kmeans(train, ['title', 'Pclass', '#Family'], 6)

# stategy_3
# dataset3_train, model3, group_summary3 = age_strategy_with_kmeans(train, ['title', 'Pclass', 'SibSp'], 6)
group_summary1
dataset1_train['group'].value_counts()
group_summary2
dataset2_train['group'].value_counts()
dataset2_train[dataset2_train['#Family'] == 10]
# It seems that setting an age for those who still don't have an age value
# with the value from similar group of people is reasonable.
# I thought value of SibSp over 1 and Parch value equal to 2 can be translated as
# the people with those value are traveling with siblings and parents.

cond = (dataset2_train['SibSp'] > 1) & (dataset2_train['Parch'] == 2)
dataset2_train.loc[cond, 'Age'].hist()
print(dataset2_train.Age.isnull().sum())

dataset2_train.loc[dataset2_train['Age'].isnull(), 'Age'] = dataset2_train.loc[cond, 'Age'].mean()
print(dataset2_train.Age.isnull().sum())

# transform test data with the same logic
dataset2_test, _, _ = age_strategy_with_kmeans(test, ['title', 'Pclass', '#Family'],
                                               model=model2, group_summary=group_summary2)
dataset2_test.loc[dataset2_test['Age'].isnull(), 'Age'] = dataset2_test.loc[cond, 'Age'].mean()
print(dataset2_test.Age.isnull().sum())
# It is clear that people with missing values in 'Embarked column' exists.
print("missing value counts in Embarked column", dataset2_train.Embarked.isnull().sum())
print("value counts in Embarked column\n", dataset2_train.Embarked.value_counts())
# Since most people were heading to South Hampton, filling missing values with S would be better than other values.
dataset2_train = dataset2_train.fillna({"Embarked": "S"})
dataset2_test = dataset2_test.fillna({"Embarked": "S"})

# map numbers to the Embarked categories
destination_dict = {'S': 1, 'C': 2, 'Q': 3}
dataset2_train = map_categ_values(dataset2_train, 'Embarked', destination_dict)
dataset2_test = map_categ_values(dataset2_test, 'Embarked', destination_dict)
dataset2_train.head()
train_y = dataset2_train['Survived']
drop_list_train = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'LastName']
drop_list_test = ['PassengerId', 'Name', 'Ticket', 'Cabin']
dataset2_train.drop(drop_list_train, axis=1, inplace=True)
dataset2_train.drop('Survived', axis=1, inplace=True)
dataset2_test.drop(drop_list_test, axis=1, inplace=True)

dataset2_train.info()
categ_list = ['Pclass', 'Sex', 'Embarked', 'title', 'group']

for categ in categ_list:
    dataset2_train[categ] = dataset2_train[categ].astype('category')
    dataset2_test[categ] = dataset2_test[categ].astype('category')

dataset2_train.info()
dataset2_train.head()
dataset2_test.head()
dataset2_train.shape
from sklearn.model_selection import train_test_split

x_train, x_val, y_train, y_val = train_test_split(
    dataset2_train, train_y, test_size=0.33,
    random_state=42,stratify=train_y
)
print(x_train.shape)
print(x_val.shape)
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gaussian)
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_logreg)
from sklearn.svm import SVC

svc = SVC()
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_svc)
# Linear SVC
from sklearn.svm import LinearSVC

linear_svc = LinearSVC()
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_linear_svc)
from sklearn.linear_model import Perceptron

perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_perceptron)
from sklearn.tree import DecisionTreeClassifier

decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_decisiontree)
from sklearn.ensemble import RandomForestClassifier

randomforest = RandomForestClassifier()
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_randomforest)
from sklearn.linear_model import SGDClassifier

sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_sgd)
from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()
gbk.fit(x_train, y_train)
y_pred = gbk.predict(x_val)
acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
print(acc_gbk)

corr = train.corr()
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"):
    f, ax = plt.subplots(figsize=(7, 5))
    ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True)
train.head()
train.info()