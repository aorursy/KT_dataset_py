import numpy as np

import pandas as pd



import os

print(os.listdir("../input"))



import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

import seaborn as sns

import scipy as sp



from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb

from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split

from sklearn import metrics

from sklearn.metrics import accuracy_score



%matplotlib inline
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df = df_train.append(df_test, sort=False)

df.head()
print("train dataset shape: ", df_train.shape)

print("test dataset shape: ", df_test.shape)

print("all dataset shape: ", df.shape)
df.dtypes
df.isnull().sum()
# plot age data

df_survived = df[(df['Survived']==1)]



fig, axes = plt.subplots(1, 2, figsize=(12,6))



axes[0].hist(x='Age', bins=8, data=df_train)

axes[0].set_xlabel('Age')

axes[0].set_ylabel('Passenger')

axes[0].set_title('All Passenger (train dataset)')



axes[1].hist(x='Age', bins=8, data=df_survived)

axes[1].set_xlabel('Age')

axes[1].set_ylabel('Survived Passenger')

axes[1].set_title('Survived Passenger (train dataset)')



print("Median of age (all dataset): ", df['Age'].median())



df_survived = df[(df['Age'].isnull()) & (df['Survived']==1)]

print("Survivor with non-data of age (train dataset): ", df_survived.shape[0])
# With a lot of missing data (~20%), if we can fill it with more precisely, it may help our prediction.



# Let's find the median of age by title name of passengers

# First, extract title name from name of passengers

df['Title'] = df['Name'].str.split(', ', expand=True)[1].str.split('.', expand=True)[0]



# Apply to train and test dataframe

df_train['Title'] = df['Title'][:891]

df_test['Title'] = df['Title'][891:]



df['Title'].unique()
# convert dataframe to list by unique title names

title_list = df['Title'].unique().tolist()



# median of age by title names

print("Median of age by title names.")

for title in title_list:

    median_age = df.groupby('Title')['Age'].median()[title_list.index(title)]

    print(title_list[title_list.index(title)], ":", median_age)

    # Fill missing age data by this median

    df.loc[(df['Age'].isnull()) & (df['Title']==title), 'Age'] = median_age

    

# Apply to train and test dataframe

df_train['Age'] = df['Age'][:891]

df_test['Age'] = df['Age'][891:]



df.isnull().sum()
# check fare unique data



np.sort(df['Fare'].dropna().unique())
# with only 1 missing fare data, fill it with the median.



# Fill non-data of fare with median

df['Fare'] = df['Fare'].fillna(df['Fare'].median()) 



# Apply to train and test dataframe

df_train['Fare'] = df['Fare'][:891]

df_test['Fare'] = df['Fare'][891:]



df.isnull().sum()
print("The value that appears most often in embarked: ", df['Embarked'].mode()[0])
# with only 2 missing embarked data, fill them with mode.



# Fill non-data of embarked with mode

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0]) 



# Apply to train and test dataframe

df_train['Embarked'] = df['Embarked'][:891]

df_test['Embarked'] = df['Embarked'][891:]



df.isnull().sum()
# define percentage compute function (train data)



def percentage(column):

    unique = np.sort(column.dropna().unique())

    pct = []

    for idx in unique:

        df_pct = df_train[(column == idx)]

        df_pct_s = df_train[(column == idx) & (df_train['Survived'] == 1)]

        pct_s_p = (df_pct_s.shape[0]/df_pct.shape[0])*100

        pct.append(pct_s_p)

    return unique, pct
# Embarked survivor percentage (train data)



column = df_train['Embarked']

unique, pct = percentage(column)

m = 0

for i in pct:

    print(column.name, "", unique[m], ":", round(i), "%")

    m += 1
# binning age data by bin cut

df['AgeBin'] = pd.cut(df['Age'], bins=8, labels=range(0,8))



# Apply to train and test dataframe

df_train['AgeBin'] = df['AgeBin'][:891]

df_test['AgeBin'] = df['AgeBin'][891:]
# binning fare data by quantile cut

df['FareBin'] = pd.qcut(df['Fare'], q=4, labels=range(0,4))



# Apply to train and test dataframe

df_train['FareBin'] = df['FareBin'][:891]

df_test['FareBin'] = df['FareBin'][891:]
# checking Pclass factor



# plot survival by Pclass

sns.set(style='darkgrid')

ax = sns.countplot(x='Pclass', hue='Survived', data=df_train)



column = df_train['Pclass']

unique, pct = percentage(column)

m = 0

for i in pct:

    print(column.name, "", unique[m], ":", round(i), "%")

    m += 1
# checking Sex factor



sns.set(style='darkgrid')

ax = sns.countplot(x='Sex', hue='Survived', data=df_train)



column = df_train['Sex']

unique, pct = percentage(column)

m = 0

for i in pct:

    print(column.name, "", unique[m], ":", round(i), "%")

    m += 1
# checking title factor



# use pd.crosstab to cross-check

pd.crosstab(df['Title'], [df['Sex'], df['AgeBin']])
# checking survival rate by title names



column = df_train['Title']

unique, pct = percentage(column)

m = 0

for i in pct:

    print(column.name, "", unique[m], ":", round(i), "%")

    m += 1
# df_title = df[(df['Title'] == 'Dr') & (df['Sex'] == 'male')]

# df_title.head()



# df['Title'] = df['Title'].replace(['Lady', 'Mlle', 'Mme', 'Ms', 'Sir', 'the Countess', 'Capt', 'Don', 'Jonkheer', 'Rev', 'Col', 'Dr', 'Master', 'Major'], 'Rare')

# df['Title'].value_counts()



df['Title'] = df['Title'].replace(['Lady', 'Mlle', 'Mme', 'Ms', 'Sir', 'the Countess', 'Dona'], 'VIP')

df['Title'] = df['Title'].replace(['Capt', 'Don', 'Jonkheer', 'Rev'], 'Sacrificed')

df['Title'] = df['Title'].replace(['Col', 'Dr', 'Master', 'Major'], 'Mid')



# Apply to train and test dataframe

df_train['Title'] = df['Title'][:891]

df_test['Title'] = df['Title'][891:]



# stat_min = 10

# title_names = df['Title'].value_counts() < stat_min

# df['Title'] = df['Title'].apply(lambda x: 'Rare' if title_names.loc[x] == True else x)



df['Title'].value_counts()
# checking survival rate by title names (grouped)



sns.set(style='darkgrid')

plt.figure(figsize=(12,6))

ax = sns.countplot(y='Title', hue='Survived', data=df_train)



column = df_train['Title']

unique, pct = percentage(column)

m = 0

for i in pct:

    print(column.name, "", unique[m], ":", round(i), "%")

    m += 1
# checking survival rate by SibSp



sns.set(style='darkgrid')

plt.figure(figsize=(12,6))

ax = sns.countplot(x='SibSp', hue='Survived', data=df_train)



column = df_train['SibSp']

unique, pct = percentage(column)

m = 0

for i in pct:

    print(column.name, "", unique[m], ":", round(i), "%")

    m += 1
# checking survival rate by Parch



sns.set(style='darkgrid')

plt.figure(figsize=(12,6))

ax = sns.countplot(x='Parch', hue='Survived', data=df_train)



column = df_train['Parch']

unique, pct = percentage(column)

m = 0

for i in pct:

    print(column.name, "", unique[m], ":", round(i), "%")

    m += 1
# Making new feature, Family and Alone



# Everyone start with 1

df['IsAlone'] = 1



# Create family size feature

df['FamilySize'] = df['SibSp'] + df['Parch'] + df['IsAlone']



# if travel with family, set alone feature to 0 

df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0



# Apply to train and test dataframe

df_train['IsAlone'] = df['IsAlone'][:891]

df_test['IsAlone'] = df['IsAlone'][891:]

df_train['FamilySize'] = df['FamilySize'][:891]

df_test['FamilySize'] = df['FamilySize'][891:]
# checking survival rate by IsAlone



sns.set(style='darkgrid')

plt.figure(figsize=(12,6))

ax = sns.countplot(x='IsAlone', hue='Survived', data=df_train)



column = df_train['IsAlone']

unique, pct = percentage(column)

m = 0

for i in pct:

    print(column.name, "", unique[m], ":", round(i), "%")

    m += 1
# checking survival rate by FamilySize



sns.set(style='darkgrid')

plt.figure(figsize=(12,6))

ax = sns.countplot(x='FamilySize', hue='Survived', data=df_train)



column = df_train['FamilySize']

unique, pct = percentage(column)

m = 0

for i in pct:

    print(column.name, "", unique[m], ":", round(i), "%")

    m += 1
# Credit to konstantinmasich's kernel

# https://www.kaggle.com/konstantinmasich/titanic-0-82-0-83



# df['Last_Name'] = df['Name'].apply(lambda x: str.split(x, ",")[0])



# DEFAULT_SURVIVAL_VALUE = 0.5

# df['Family_Survival'] = DEFAULT_SURVIVAL_VALUE



# for grp, grp_df in df[['Survived', 'Name', 'Last_Name', 'Fare', 'Ticket', 'PassengerId',

#                            'SibSp', 'Parch', 'Age', 'Cabin']].groupby(['Last_Name', 'Fare']):

    

#     if (len(grp_df) != 1):

#         # A Family group is found.

#         for ind, row in grp_df.iterrows():

#             smax = grp_df.drop(ind)['Survived'].max()

#             smin = grp_df.drop(ind)['Survived'].min()

#             passID = row['PassengerId']

#             if (smax == 1.0):

#                 df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 1

#             elif (smin==0.0):

#                 df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 0



# print("Number of passengers with family survival information:", 

#       df.loc[df['Family_Survival']!=0.5].shape[0])



# for _, grp_df in df.groupby('Ticket'):

#     if (len(grp_df) != 1):

#         for ind, row in grp_df.iterrows():

#             if (row['Family_Survival'] == 0) | (row['Family_Survival']== 0.5):

#                 smax = grp_df.drop(ind)['Survived'].max()

#                 smin = grp_df.drop(ind)['Survived'].min()

#                 passID = row['PassengerId']

#                 if (smax == 1.0):

#                     df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 1

#                 elif (smin==0.0):

#                     df.loc[df['PassengerId'] == passID, 'Family_Survival'] = 0

                        

# print("Number of passenger with family/group survival information: " 

#       +str(df[df['Family_Survival']!=0.5].shape[0]))



# # Family_Survival in TRAIN_DF and TEST_DF:

# df_train['Family_Survival'] = df['Family_Survival'][:891]

# df_test['Family_Survival'] = df['Family_Survival'][891:]
# Target Selection (Train Data)

df_tar = df_train[['Survived']]



# Feature Selection (Train Data)

df_sel = df_train[['Pclass', 'Sex', 'Title', 'Embarked', 'AgeBin', 'FareBin', 'IsAlone', 'FamilySize']]



# Feature Selection (Test Data)

df_test_sel = df_test[['Pclass', 'Sex', 'Title', 'Embarked', 'AgeBin', 'FareBin', 'IsAlone', 'FamilySize']]
# Onehot Encoding (Train Data)

df_sel_dummy = pd.get_dummies(df_sel)



# Remove some str from column name

# import re

# regex = re.compile(r"\[|\]|<", re.IGNORECASE)

# df_sel_dummy.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in df_sel_dummy.columns.values]



df_sel_dummy.head()
# Onehot Encoding (Test Data)

df_test_sel_dummy = pd.get_dummies(df_test_sel)



# Remove some str from column name

# import re

# regex = re.compile(r"\[|\]|<", re.IGNORECASE)

# df_test_sel_dummy.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in df_test_sel_dummy.columns.values]



df_test_sel_dummy.head()
# Label Encoder

# from sklearn import preprocessing

# le = preprocessing.LabelEncoder()



# df_sel_lab = df_sel

# df_sel_lab['Sex_Code'] = le.fit_transform(df_sel_lab['Sex'])

# df_sel_lab['Embarked_Code'] = le.fit_transform(df_sel_lab['Embarked'])

# df_sel_lab['Title_Code'] = le.fit_transform(df_sel_lab['Title'])

# df_sel_lab['AgeBin_Code'] = le.fit_transform(df_sel_lab['AgeBin'])

# df_sel_lab['FareBin_Code'] = le.fit_transform(df_sel_lab['FareBin'])



# df_sel_lab.drop(columns=['Sex', 'Embarked', 'Title', 'AgeBin', 'FareBin'], axis=1, inplace=True)

# df_sel_lab.head()
# Machine Learning Algorithm: LogisticRegression



X = df_sel_dummy

y = np.ravel(df_tar)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = LogisticRegression(max_iter=1000, solver='liblinear')

clf.fit(X_train, y_train)

print(clf)

#y_pred = clf.predict(df_test_sel_dummy)

print('Accuracy on test set: {:.2f}'.format(accuracy_score(y_test, clf.predict(X_test))))

print('Accuracy on Train set: {:.2f}'.format(accuracy_score(y_train, clf.predict(X_train))))
# Machine Learning Algorithm: RandomForestClassifier



X = df_sel_dummy

y = np.ravel(df_tar)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = RandomForestClassifier(n_estimators=100, max_depth=9, random_state=0)

clf.fit(X_train, y_train)

#y_pred = clf.predict(df_test_sel_dummy)

print(clf)

print('Accuracy on test set: {:.2f}'.format(accuracy_score(y_test, clf.predict(X_test))))

print('Accuracy on Train set: {:.2f}'.format(accuracy_score(y_train, clf.predict(X_train))))
# Machine Learning Algorithm: XGBClassifier



X = df_sel_dummy

y = np.ravel(df_tar)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

clf = xgb.XGBClassifier(max_depth=8, learning_rate=0.2, n_estimators=100)

clf.fit(X_train, y_train)

#y_pred = clf.predict(df_test_sel_dummy)

print(clf)

print('Accuracy on test set: {:.2f}'.format(accuracy_score(y_test, clf.predict(X_test))))

print('Accuracy on Train set: {:.2f}'.format(accuracy_score(y_train, clf.predict(X_train))))
# Submit

# LogisticRegression

# Public Score: 0.79425



X = df_sel_dummy

y = np.ravel(df_tar)

clf = LogisticRegression(max_iter=1000, solver='liblinear')

clf.fit(X, y)

y_pred = clf.predict(df_test_sel_dummy)

print(clf)



print(y_pred.shape)



submit = pd.DataFrame({

        "PassengerId": df_test["PassengerId"],

        "Survived": y_pred

    })

submit.to_csv('titanic_submit.csv', index=False)



print(submit.head())