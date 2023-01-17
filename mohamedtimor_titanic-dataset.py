# numpy, matplotlib, seaborn, pandas
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

# machine learning
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Ignore warnings thrown by Seaborn
import warnings
warnings.filterwarnings('ignore')
# get titanic & test csv files as a DataFrame
titanic_df = pd.read_csv("../input/train.csv")
test_df    = pd.read_csv("../input/test.csv")

# preview the data
titanic_df.head()
# Understand the datatypes
print(titanic_df.dtypes)
print('_'*40)
# Focus first on null values
print(titanic_df.isna().sum())
titanic_df.info()
print("----------------------------")
test_df.info()
print(test_df.dtypes)
print('_'*40)
print(test_df.isna().sum())
# Check the correlation for the current numeric feature set.
print(titanic_df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr())
sns.heatmap(titanic_df[['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']].corr(), annot=True, fmt = ".2f", cmap = "coolwarm")
# Lets see the relation between Pclass and Survived
print(titanic_df[['Pclass', 'Survived']].groupby(['Pclass']).mean())
sns.catplot(x='Pclass', y='Survived',  kind='bar', data=titanic_df)
print(titanic_df[['Sex', 'Survived']].groupby(['Sex']).mean())
sns.catplot(x='Sex', y='Survived',  kind='bar', data=titanic_df)
sns.catplot(x='Sex', y='Survived',  kind='bar', data=titanic_df, hue='Pclass')
g = sns.FacetGrid(titanic_df, col='Survived')
g = g.map(sns.distplot, "Fare")
group = pd.cut(titanic_df.Fare, [0,50,100,150,200,550])
piv_fare = titanic_df.pivot_table(index=group, columns='Survived', values = 'Fare', aggfunc='count')
piv_fare.plot(kind='bar')
g = sns.FacetGrid(titanic_df, col='Survived')
g = g.map(sns.distplot, "Age")
group = pd.cut(titanic_df.Age, [0,14,30,60,100])
piv_fare = titanic_df.pivot_table(index=group, columns='Survived', values = 'Age', aggfunc='count')
piv_fare.plot(kind='bar')
print(titanic_df[['Embarked', 'Survived']].groupby(['Embarked']).mean())
sns.catplot(x='Embarked', y='Survived',  kind='bar', data=titanic_df)
sns.catplot('Pclass', kind='count', col='Embarked', data=titanic_df)

print(titanic_df[['SibSp', 'Survived']].groupby(['SibSp']).mean())
sns.catplot(x='SibSp', y='Survived', data=titanic_df, kind='bar')
print(titanic_df[['Parch', 'Survived']].groupby(['Parch']).mean())
sns.catplot(x='Parch', y='Survived', data=titanic_df, kind='bar')
# Get the titles
for dataset in [titanic_df, test_df]:
    # Use split to get only the titles from the name
    dataset['Title'] = dataset['Name'].str.split(',').str[1].str.split('.').str[0].str.strip()
    # Check the initial list of titles.
    print(dataset['Title'].value_counts())
    print()
sns.catplot(x='Survived', y='Title', data=titanic_df, kind ='bar')

for df in [titanic_df, test_df]:
    print(df.shape)
    print()
    print(df.isna().sum())
# Drop rows with nulls for Embarked
for df in [titanic_df, test_df]:
    df.dropna(subset = ['Embarked'], inplace = True)
test_df['Fare'].fillna(test_df[test_df['Pclass'] == 3].Fare.median(), inplace = True)
# Returns titles from the passed in series.
def getTitle(series):
    return series.str.split(',').str[1].str.split('.').str[0].str.strip()
# Prints the count of titles with nulls for the train dataframe.
print(getTitle(titanic_df[titanic_df.Age.isnull()].Name).value_counts())
# Fill Age median based on Title
mr_mask = titanic_df['Title'] == 'Mr'
miss_mask = titanic_df['Title'] == 'Miss'
mrs_mask = titanic_df['Title'] == 'Mrs'
master_mask = titanic_df['Title'] == 'Master'
dr_mask = titanic_df['Title'] == 'Dr'
titanic_df.loc[mr_mask, 'Age'] = titanic_df.loc[mr_mask, 'Age'].fillna(titanic_df[titanic_df.Title == 'Mr'].Age.mean())
titanic_df.loc[miss_mask, 'Age'] = titanic_df.loc[miss_mask, 'Age'].fillna(titanic_df[titanic_df.Title == 'Miss'].Age.mean())
titanic_df.loc[mrs_mask, 'Age'] = titanic_df.loc[mrs_mask, 'Age'].fillna(titanic_df[titanic_df.Title == 'Mrs'].Age.mean())
titanic_df.loc[master_mask, 'Age'] = titanic_df.loc[master_mask, 'Age'].fillna(titanic_df[titanic_df.Title == 'Master'].Age.mean())
titanic_df.loc[dr_mask, 'Age'] = titanic_df.loc[dr_mask, 'Age'].fillna(titanic_df[titanic_df.Title == 'Dr'].Age.mean())
print(getTitle(titanic_df[titanic_df.Age.isnull()].Name).value_counts())
# Prints the count of titles with nulls for the validation dataframe.
print(getTitle(test_df[test_df.Age.isnull()].Name).value_counts())
# Fill Age median based on Title
mr_mask = test_df['Title'] == 'Mr'
miss_mask = test_df['Title'] == 'Miss'
mrs_mask = test_df['Title'] == 'Mrs'
master_mask = test_df['Title'] == 'Master'
ms_mask = test_df['Title'] == 'Ms'
test_df.loc[mr_mask, 'Age'] = test_df.loc[mr_mask, 'Age'].fillna(test_df[test_df.Title == 'Mr'].Age.mean())
test_df.loc[miss_mask, 'Age'] = test_df.loc[miss_mask, 'Age'].fillna(test_df[test_df.Title == 'Miss'].Age.mean())
test_df.loc[mrs_mask, 'Age'] = test_df.loc[mrs_mask, 'Age'].fillna(test_df[test_df.Title == 'Mrs'].Age.mean())
test_df.loc[master_mask, 'Age'] = test_df.loc[master_mask, 'Age'].fillna(test_df[test_df.Title == 'Master'].Age.mean())
test_df.loc[ms_mask, 'Age'] = test_df.loc[ms_mask, 'Age'].fillna(test_df[test_df.Title == 'Miss'].Age.mean())
print(getTitle(test_df[test_df.Age.isnull()].Name).value_counts())
print(titanic_df.isna().sum())
print(test_df.isna().sum())
titanic_df.drop(columns=['PassengerId'], inplace = True)
[df.drop(columns=['Ticket'], inplace = True) for df in [titanic_df, test_df]]
[titanic_df, test_df] = [pd.get_dummies(data = df, columns = ['Pclass', 'Sex', 'Embarked']) for df in [titanic_df, test_df]]
for df in [titanic_df, test_df]:
    df['HasCabin'] = df['Cabin'].notna().astype(int)
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] > 1).astype(int)
[df.drop(columns=['Cabin', 'SibSp', 'Parch'], inplace = True) for df in [titanic_df, test_df]]

titanic_df['Title'] = titanic_df['Title'].replace('Mlle', 'Miss').replace('Ms', 'Miss').replace('Mme', 'Mrs').replace(['Dr', 'Major', 'Col', 'Rev', 'Lady', 'Jonkheer', 'Don', 'Sir', 'Dona', 'Capt', 'the Countess'], 'Special')
test_df['Title'] = test_df['Title'].replace('Mlle', 'Miss').replace('Ms', 'Miss').replace('Mme', 'Mrs').replace(['Dr', 'Major', 'Col', 'Rev', 'Lady', 'Jonkheer', 'Don', 'Sir', 'Dona', 'Capt', 'the Countess'], 'Special')
[df.drop(columns=['Name'], inplace = True) for df in [titanic_df, test_df]]
[titanic_df, test_df] = [pd.get_dummies(data = df, columns = ['Title']) for df in [titanic_df, test_df]]
# Check the updated dataset
print(titanic_df.columns.values)
print(test_df.columns.values)
# Check the correlation with the updated datasets
titanic_df.corr()
# Use only the features with a coeefficient greater than 0.3
X = titanic_df[['Fare', 'Pclass_1', 'Pclass_3', 'Sex_female', 'Embarked_C', 'Embarked_S', 'HasCabin', 'IsAlone', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs']]
y = titanic_df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)
print(X_train.shape, X_test.shape)
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

random_forest.score(X_train, y_train)
# Now we will pass the validation set provided for creating our submission
# Pick the same columns as in X_test
X_validation = test_df[['Fare', 'Pclass_1', 'Pclass_3', 'Sex_female', 'Embarked_C', 'Embarked_S', 'HasCabin', 'IsAlone', 'Title_Master', 'Title_Miss', 'Title_Mr', 'Title_Mrs']]
# Call the predict from the created classifier
y_valid = random_forest.predict(X_validation)
# Creating final output file
validation_pId = test_df.loc[:, 'PassengerId']
my_submission = pd.DataFrame(data={'PassengerId':validation_pId, 'Survived':y_valid})
print(my_submission['Survived'].value_counts())
my_submission.to_csv('submission.csv', index = False)
