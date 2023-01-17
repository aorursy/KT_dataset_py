import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns

import scipy.stats

from scipy.stats import chi2_contingency

from scipy.stats import chi2

import re

from sklearn import feature_selection

from collections import Counter

pd.set_option("display.max_rows", 50, "display.max_columns", 50)
df_train = pd.read_csv('../input/titanic/train.csv')

df_test = pd.read_csv('../input/titanic/test.csv')

PassengerId = df_test['PassengerId']

sample_subm = pd.read_csv('../input/titanic/gender_submission.csv')
# Outlier detection 



def detect_outliers(df,n,features):

    """

    Takes a dataframe df of features and returns a list of the indices

    corresponding to the observations containing more than n outliers according

    to the Tukey method.

    """

    outlier_indices = []

    

    # iterate over features(columns)

    for col in features:

        # 1st quartile (25%)

        Q1 = np.percentile(df[col], 25)

        # 3rd quartile (75%)

        Q3 = np.percentile(df[col],75)

        # Interquartile range (IQR)

        IQR = Q3 - Q1

        

        # outlier step

        outlier_step = 1.5 * IQR

        

        # Determine a list of indices of outliers for feature col

        outlier_list_col = df[(df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step )].index

        

        # append the found outlier indices for col to the list of outlier indices 

        outlier_indices.extend(outlier_list_col)

        

    # select observations containing more than 2 outliers

    outlier_indices = Counter(outlier_indices)        

    multiple_outliers = list( k for k, v in outlier_indices.items() if v > n )

    

    return multiple_outliers   



# detect outliers from Age, SibSp , Parch and Fare

Outliers_to_drop = detect_outliers(df_train,2,["Age","SibSp","Parch","Fare"])

# Drop outliers

df_train = df_train.drop(Outliers_to_drop, axis = 0).reset_index(drop=True)
full_data = [df_train, df_test]



# Feature that tells whether a passenger had a cabin on the Titanic

df_train['Has_Cabin'] = df_train["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

df_test['Has_Cabin'] = df_test["Cabin"].apply(lambda x: 0 if type(x) == float else 1)



# Feature engineering steps taken from Sina

# Create new feature FamilySize as a combination of SibSp and Parch

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Create new feature IsAlone from FamilySize

for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Remove all NULLS in the Embarked column

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

    

    

# Remove all NULLS in the Fare column and create a new feature CategoricalFare

for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(df_train['Fare'].median())

df_train['CategoricalFare'] = pd.qcut(df_train['Fare'], 4)

# Create a New feature CategoricalAge

for dataset in full_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

df_train['CategoricalAge'] = pd.cut(df_train['Age'], 5)

# Define function to extract titles from passenger names

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""

# Create a new feature Title, containing the titles of passenger names

for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    

    # Mapping titles

    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    

    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] 						        = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare'] 							        = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age'] 					       = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age'] = 4 ;
# Feature selection

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

df_train = df_train.drop(drop_elements, axis = 1)

df_train = df_train.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

df_test  = df_test.drop(drop_elements, axis = 1)
#features = ['Pclass', 'Sex', 'Embarked']

features = df_train.columns[1:]

fig = plt.figure(figsize=(15, 13))

for i in range(len(features)):

    cont_table = np.array(pd.crosstab(df_train.Survived, df_train[features[i]], rownames=None, colnames=None))

    stat, p, dof, expected = chi2_contingency(cont_table)

    obs = np.sum(cont_table)

    mini = min(cont_table.shape)-1 

    cramers_v_stat = (stat/(obs*mini))

    fig.add_subplot(4, 3, i+1)

    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=0.5)

    ax = sns.countplot(x=features[i], hue="Survived", data=df_train)

    ax.set_xlabel(features[i] + ', ' + 'cramers V: ' + str(round(cramers_v_stat, 4)) + '\n' + 'p-value: ' + str(round(p,5)), fontsize=16)

plt.show()
cont_table = np.array(pd.crosstab(df_train.Title, df_train.Sex, rownames=None, colnames=None))

pd.crosstab(df_train.Title, df_train.Sex, rownames=None, colnames=None) # "Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5 
stat, p, dof, expected = chi2_contingency(cont_table)

obs = np.sum(cont_table)

mini = min(cont_table.shape)-1 

cramers_v_stat = (stat/(obs*mini))

ax = sns.countplot(x='Title', hue="Sex", data=df_train)

ax.set_xlabel('Title' + ', ' + 'cramers V: ' + str(round(cramers_v_stat, 4)) + '\n' + 'p-value: ' + str(round(p,5)), fontsize=16)

plt.show()