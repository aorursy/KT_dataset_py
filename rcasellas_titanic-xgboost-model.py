import numpy as np

import pandas as pd

import xgboost as xgb

from xgboost.sklearn import XGBClassifier

import matplotlib.pylab as plt

%matplotlib inline

from matplotlib.pylab import rcParams

rcParams['figure.figsize'] = 12, 4

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

combine = [train, test]
#Method for finding substrings

def substrings_in_string(big_string, substrings):

    for substring in substrings:

        if substring in big_string:

            return substring

    return np.nan
#Map titles

title_list=['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',

                    'Dr', 'Ms', 'Mlle','Col', 'Capt', 'Mme', 'Countess',

                    'Don', 'Jonkheer']

title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

cabin_list = ['A', 'B', 'C', 'D', 'E', 'F', 'T', 'G', 'Unknown']
for df in combine:

    #set child feature

    df['child'] = float('NaN')

    df["child"][df["Age"] < 18] = 1

    df["child"][df["Age"] >=18] = 0

    

    # Convert the male and female groups to integer form

    df["Sex"][df["Sex"] == "male"] = 0

    df["Sex"][df["Sex"] == "female"] = 1

    

    df['Title'] = df['Name'].astype(str).map(lambda x: substrings_in_string(x, title_list))

    

    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col',\

 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')



    df['Title'] = df['Title'].replace('Mlle', 'Miss')

    df['Title'] = df['Title'].replace('Ms', 'Miss')

    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    

    df['Title'] = df['Title'].map(title_mapping)

    df['Title'] = df['Title'].fillna(0)

    

    df['Deck'] = df['Cabin'].astype(str).map(lambda x: substrings_in_string(x, cabin_list))

    

    df["Deck"][df["Deck"] == "A"] = 1

    df["Deck"][df["Deck"] == "B"] = 2

    df["Deck"][df["Deck"] == "C"] = 3

    df["Deck"][df["Deck"] == "D"] = 4

    df["Deck"][df["Deck"] == "E"] = 5

    df["Deck"][df["Deck"] == "F"] = 6

    df["Deck"][df["Deck"] == "G"] = 7

    df["Deck"][df["Deck"] == "T"] = 8

    

    df["Deck"] = df["Deck"].fillna(0)

    

    df['Family_size'] = df['SibSp']+df['Parch']+1

    

    df['Fare_Per_Person']=df['Fare']/(df['Family_size']+1)

    

    df['isAlone']=0

    df.loc[df['Family_size']==1, 'isAlone'] = 1

    

    # Impute the Embarked variable

    df["Embarked"] = df["Embarked"].fillna("S")



    # Convert the Embarked classes to integer form

    df["Embarked"][df["Embarked"] == "S"] = 0

    df["Embarked"][df["Embarked"] == "C"] = 1

    df["Embarked"][df["Embarked"] == "Q"] = 2
guess_ages = np.zeros((2,3))

guess_ages
#Guess ages based off Sex and Pclass

for df in combine:

    for i in range(0, 2):

        for j in range(0, 3):

            guess_df = df[(df['Sex'] == i) & \

                                  (df['Pclass'] == j+1)]['Age'].dropna()

            age_guess = guess_df.median()



            # Convert random age float to nearest .5 age

            guess_ages[i,j] = int( age_guess/0.5 + 0.5 ) * 0.5

            

    for i in range(0, 2):

        for j in range(0, 3):

            df.loc[ (df.Age.isnull()) & (df.Sex == i) & (df.Pclass == j+1),\

                    'Age'] = guess_ages[i,j]



    df['Age'] = df['Age'].astype(int)
excl = ['PassengerId', 'Survived', 'Ticket', 'Cabin', 'Name']

cols = [c for c in train.columns if c not in excl]
target = train["Survived"].values

features = train[cols].values
xgb1 = XGBClassifier(learning_rate=0.1, n_estimators=100, max_depth=4, min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, objective='binary:logistic', nthread=4, scale_pos_weight=1, seed=27)
xgbmod = xgb1.fit(features, target)
xgbmod.score(features, target)
feat_imp = pd.Series(xgbmod.booster().get_fscore()).sort_values(ascending=False)

feat_imp.plot(kind='bar', title='Feature Importances')

plt.ylabel('Feature Importance Score')
test_features = test[cols].values
pred = xgb1.predict(test_features)
PassengerId = np.array(test["PassengerId"]).astype(int)

my_solution = pd.DataFrame(pred, PassengerId, columns = ["Survived"])
my_solution.to_csv("XGBoost.csv", index_label = ["PassengerId"])