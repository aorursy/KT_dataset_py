import numpy as np 

import pandas as pd



from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder





# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_train = pd.read_csv("/kaggle/input/titanic/train.csv")

df_test = pd.read_csv("/kaggle/input/titanic/test.csv")

df_data = df_train.append(df_test)

df_data.reset_index(inplace=True, drop=True)

df_data.head()
df_test.info()
df_data.info()
df_data.head()
df_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df_data[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
g = sns.FacetGrid(df_data, col='Survived')

g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(df_data, row='Embarked', size=2.2, aspect=1.6)

grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')

grid.add_legend()
df_data['Sex'] = df_data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
df_data['Fare'] = df_data['Fare'].fillna(df_data['Fare'].median())
df_data['FareBand'] = pd.qcut(df_data['Fare'], 4)

df_data[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)
df_data.loc[df_data['Fare'] <= 7.896, 'Fare'] = 0

df_data.loc[(df_data['Fare'] > 7.896) & (df_data['Fare'] <= 14.454), 'Fare'] = 1

df_data.loc[(df_data['Fare'] > 14.454) & (df_data['Fare'] <=  31.275), 'Fare']   = 2

df_data.loc[ df_data['Fare'] >  31.275, 'Fare'] = 3

df_data['Fare'] = df_data['Fare'].astype(int)

df_data[['Fare', 'Survived']].groupby(['Fare'], as_index=False).mean().sort_values(by='Fare', ascending=True)
df_data['Embarked'] = df_data['Embarked'].fillna(df_data['Embarked'].dropna().mode()[0])

df_data['Embarked'] = df_data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)
df_data[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Embarked', ascending=True)
df_data['Title'] = df_data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(df_data['Title'], df_data['Sex']).T
df_data['Title'] = df_data['Title'].replace(['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

df_data['Title'] = df_data['Title'].replace('Mlle', 'Miss')

df_data['Title'] = df_data['Title'].replace('Ms', 'Miss')

df_data['Title'] = df_data['Title'].replace('Mme', 'Mrs')

df_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

df_data['Title'] = df_data['Title'].map(title_mapping)

df_data['Title'] = df_data['Title'].fillna(0)
df_data[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()
guess_ages = np.zeros((2,3))
for i in range(0, 2):

    for j in range(0, 3):

        guess_df = df_data[(df_data['Sex'] == i) & (df_data['Pclass'] == j+1)]['Age'].dropna()



        # age_mean = guess_df.mean()

        # age_std = guess_df.std()

        # age_guess = rnd.uniform(age_mean - age_std, age_mean + age_std)



        age_guess = guess_df.median()



        # Convert random age float to nearest .5 age

        guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5



for i in range(0, 2):

    for j in range(0, 3):

        df_data.loc[ (df_data.Age.isnull()) & (df_data.Sex == i) & (df_data.Pclass == j+1),\

                'Age'] = guess_ages[i,j]



df_data['Age'] = df_data['Age'].astype(int)
df_data['AgeBand'] = pd.cut(df_data['Age'], 5)

df_data[['AgeBand', 'Survived']].groupby(['AgeBand'], as_index=False).mean().sort_values(by='AgeBand', ascending=True)
df_data.loc[ df_data['Age'] <= 16, 'Age'] = 0

df_data.loc[(df_data['Age'] > 16) & (df_data['Age'] <= 32), 'Age'] = 1

df_data.loc[(df_data['Age'] > 32) & (df_data['Age'] <= 48), 'Age'] = 2

df_data.loc[(df_data['Age'] > 48) & (df_data['Age'] <= 64), 'Age'] = 3

df_data.loc[ df_data['Age'] > 64, 'Age'] = 4

df_data.head()
df_data[['Age', 'Survived']].groupby(['Age'], as_index=False).mean()
df_data['FamilySize'] = df_data['SibSp'] + df_data['Parch'] + 1
df_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
df_data['IsAlone'] = 0

df_data.loc[df_data['FamilySize'] == 1, 'IsAlone'] = 1
df_data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index=False).mean()
df_data.head()
df_data['Age*Class'] = df_data.Age * df_data.Pclass
df_train = df_data[:len(df_train)]

df_test = df_data[len(df_train):]
y = df_train["Survived"]

# features = ["Pclass", "Sex", "Age", "Fare", "Embarked", "Title", "IsAlone", "Age*Class"]

features = ["Pclass", "Sex", "Age", "Fare", "Embarked"]



X = df_train[features]

X_test = df_test[features]



model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=2)

# model = RandomForestClassifier(n_estimators=100)

model.fit(X, y)



predictions = model.predict(X_test)

predictions = predictions.astype(int)

# print('Base oob score :%.5f' %(model.oob_score_))

output = pd.DataFrame({'PassengerId': df_test.PassengerId, 'Survived': predictions})

output.to_csv('my_submission_1.csv', index=False)

print("Your submission was successfully saved!")