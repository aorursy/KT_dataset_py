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
df_train["Age"].mean()
df_train["Initial"] = df_train['Name'].str.extract("([A-Za-z]+)\.")

df_test["Initial"] = df_test['Name'].str.extract("([A-Za-z]+)\.")
pd.crosstab(df_train['Initial'], df_train['Sex']).T.style.background_gradient(cmap='cool')
df_train['Initial'].replace(['Mlle', "Mme", "Ms", "Dr", "Major", "Lady", "Countess", "Jonkheer", "Col", "Rev", "Capt", "Sir", "Don", "Dona"], 

                           ["Miss", "Miss", "Miss", "Mr", "Mr", "Mrs", "Mrs", "Other", "Other", "Other", "Mr", "Mr", "Mr", "Mr"], inplace=True)

df_test['Initial'].replace(['Mlle', "Mme", "Ms", "Dr", "Major", "Lady", "Countess", "Jonkheer", "Col", "Rev", "Capt", "Sir", "Don", "Dona"], 

                           ["Miss", "Miss", "Miss", "Mr", "Mr", "Mrs", "Mrs", "Other", "Other", "Other", "Mr", "Mr", "Mr", "Mr"], inplace=True)
df_train.groupby("Initial").mean()
df_train.groupby("Initial")['Survived'].mean().plot.bar()
df_train.loc[(df_train['Age'].isnull()) & (df_train["Initial"]=="Mr"), "Age"] = 33

df_train.loc[(df_train['Age'].isnull()) & (df_train["Initial"]=="Mrs"), "Age"] = 36

df_train.loc[(df_train['Age'].isnull()) & (df_train["Initial"]=="Master"), "Age"] = 5

df_train.loc[(df_train['Age'].isnull()) & (df_train["Initial"]=="Miss"), "Age"] = 22

df_train.loc[(df_train['Age'].isnull()) & (df_train["Initial"]=="Other"), "Age"] = 46
df_test.loc[(df_test['Age'].isnull()) & (df_test["Initial"]=="Mr"), "Age"] = 33

df_test.loc[(df_test['Age'].isnull()) & (df_test["Initial"]=="Mrs"), "Age"] = 36

df_test.loc[(df_test['Age'].isnull()) & (df_test["Initial"]=="Master"), "Age"] = 5

df_test.loc[(df_test['Age'].isnull()) & (df_test["Initial"]=="Miss"), "Age"] = 22

df_test.loc[(df_test['Age'].isnull()) & (df_test["Initial"]=="Other"), "Age"] = 46
df_train['Embarked'].isnull().sum()
df_train.shape
df_train['Embarked'].fillna('S', inplace=True)
df_train['Embarked'].isnull().sum()
df_train['Age_cat'] = 0

df_train.head()
df_train.loc[df_train["Age"]<10, 'Age_cat'] = 0

df_train.loc[(10 <= df_train['Age']) & (df_train['Age'] < 20), 'Age_cat'] = 1

df_train.loc[(20 <= df_train['Age']) & (df_train['Age'] < 30), 'Age_cat'] = 2

df_train.loc[(30 <= df_train['Age']) & (df_train['Age'] < 40), 'Age_cat'] = 3

df_train.loc[(40 <= df_train['Age']) & (df_train['Age'] < 50), 'Age_cat'] = 4

df_train.loc[(50 <= df_train['Age']) & (df_train['Age'] < 60), 'Age_cat'] = 5

df_train.loc[(60 <= df_train['Age']) & (df_train['Age'] < 70), 'Age_cat'] = 6

df_train.loc[(70 <= df_train['Age']), 'Age_cat'] = 7
df_test.loc[df_test["Age"]<10, 'Age_cat'] = 0

df_test.loc[(10 <= df_test['Age']) & (df_test['Age'] < 20), 'Age_cat'] = 1

df_test.loc[(20 <= df_test['Age']) & (df_test['Age'] < 30), 'Age_cat'] = 2

df_test.loc[(30 <= df_test['Age']) & (df_test['Age'] < 40), 'Age_cat'] = 3

df_test.loc[(40 <= df_test['Age']) & (df_test['Age'] < 50), 'Age_cat'] = 4

df_test.loc[(50 <= df_test['Age']) & (df_test['Age'] < 60), 'Age_cat'] = 5

df_test.loc[(60 <= df_test['Age']) & (df_test['Age'] < 70), 'Age_cat'] = 6

df_test.loc[(70 <= df_test['Age']), 'Age_cat'] = 7
df_train.head()
def category_age(x):

    if x < 10:

        return 0

    elif x < 20:

        return 1

    elif x < 30:

        return 2

    elif x < 40:

        return 3

    elif x < 50:

        return 4

    elif x < 60:

        return 5

    elif x < 70:

        return 6

    else:

        return 7
df_train['Age_cat_2'] = df_train['Age'].apply(category_age)
(df_train['Age_cat'] == df_train['Age_cat_2']).all()
df_train.drop(['Age', 'Age_cat_2'], axis=1, inplace=True)

df_test.drop(['Age'], axis=1, inplace=True)
df_train.Initial.unique()
df_train["Initial"] = df_train['Initial'].map({"Master": 0, "Miss": 1, "Mr": 2, "Mrs": 3, "Other": 4})

df_test["Initial"] = df_test['Initial'].map({"Master": 0, "Miss": 1, "Mr": 2, "Mrs": 3, "Other": 4})
df_train.Embarked.unique()
df_train["Embarked"] = df_train["Embarked"].map({"C": 0, "Q": 1, "S": 2})

df_test["Embarked"] = df_test["Embarked"].map({"C": 0, "Q": 1, "S": 2})
df_train.head()
df_train.Embarked.isnull().any()
df_train["Sex"].unique()
df_train["Sex"] = df_train["Sex"].map({'female': 0, 'male': 1})

df_test["Sex"] = df_test["Sex"].map({'female': 0, 'male': 1})
df_train["FamilySize"] = df_train['SibSp'] + df_train['Parch'] + 1

df_test["FamilySize"] = df_test['SibSp'] + df_test['Parch'] + 1
heatmap_data = df_train[['Survived', 'Pclass', 'Sex', 'Fare', 'Embarked', 'FamilySize', 'Initial', 'Age_cat']]
colormap = plt.cm.Blues

plt.figure(figsize=(12,10))

plt.title("Pearson Correlation of Features", y=1.05, size=15)

sns.heatmap(heatmap_data.astype(float).corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True, annot_kws={'size': 16}, fmt=".2f")
df_train = pd.get_dummies(df_train, columns=['Initial'], prefix="Initial")

df_test = pd.get_dummies(df_test, columns=['Initial'], prefix="Initial")
df_train.head()
df_test.head()
df_train = pd.get_dummies(df_train, columns=["Embarked"], prefix="Embarked")

df_test = pd.get_dummies(df_test, columns=["Embarked"], prefix="Embarked")
df_train.head()
df_test.head()
df_train.drop(['PassengerId', 'Name', "SibSp", 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)

df_test.drop(['PassengerId', 'Name', "SibSp", 'Parch', 'Ticket', 'Cabin'], axis=1, inplace=True)
from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.model_selection import train_test_split
df_test.isnull().sum()
df_test.fillna(0.0,inplace=True)
df_test.isnull().sum()
x_train = df_train.drop("Survived", axis=1).values

target_label = df_train['Survived'].values

x_test = df_test.values
x_tr, x_vld, y_tr, y_vld = train_test_split(x_train, target_label, test_size=0.3, random_state=2020)
model = RandomForestClassifier()

model.fit(x_tr, y_tr)
prediction = model.predict(x_vld)

print("Accuracy {:.2f}%".format(100*metrics.accuracy_score(prediction,y_vld)))
model.feature_importances_
from pandas import Series

feature_importance = model.feature_importances_

Series_feat_imp = Series(feature_importance, index=df_test.columns)
Series_feat_imp
plt.figure(figsize=(8, 8))

Series_feat_imp.sort_values(ascending=True).plot.barh()

plt.xlabel('Feature importance')

plt.ylabel('Feature')

plt.show()
submission = pd.read_csv('../input/titanic/gender_submission.csv')
submission.head()
prediction = model.predict(x_test)
submission['Survived'] = prediction
submission.to_csv('./submission.csv', index=False)