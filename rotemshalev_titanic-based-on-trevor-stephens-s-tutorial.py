import numpy as np

import pandas as pd

import pandas_profiling as pp

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.metrics import zero_one_loss



import os

print(os.listdir("../input"))

train_set = pd.read_csv("../input/train.csv")

pp.ProfileReport(train_set)
# print(train_set.groupby('Sex').count()["PassengerId"])

# train_set[train_set["Survived"]==1].groupby('Sex').count()["Survived"]

print("Percentage of males that survived:", train_set[train_set["Sex"]=='male'][train_set["Survived"]==1]["Survived"].count()/train_set[train_set["Sex"]=='male']["Survived"].count())

print("Percentage of females that survived:", train_set[train_set["Sex"]=='female'][train_set["Survived"]==1]["Survived"].count()/train_set[train_set["Sex"]=='female']["Survived"].count())

# train_set[train_set["Survived"]==1]['Sex'].value_counts(normalize=True) * 100
# dt_model = DecisionTreeClassifier(random_state=0)

# print(train_set["Age"][train_set["Age"].notnull()])

# dt_model.fit(train_set.drop("Age",axis=1), train_set["Age"][train_set["Age"].notnull()])

# train_set["Age"][train_set["Age"].istnull()] = dt_model.predict(train_set["Age"][train_set["Age"].istnull()])
test_set = pd.read_csv("../input/test.csv")

merged_set = pd.concat([train_set, test_set])
# len(train_set["Age"][train_set["Age"].isnull().values==True])

# merged_set["Age"][merged_set["Age"].isnull().values==True] = merged_set["Age"].mean() # Fills mean to NA value



# merged_set["Age"].describe()
merged_set["Child"] = (merged_set["Age"] < 16).astype(int)

merged_set["Child"][merged_set["Child"].isnull()] = 0

chlidren_survived = merged_set[['Child', 'Survived']].groupby('Child').agg(['sum', 'count'])

print(chlidren_survived['Survived']['sum']/chlidren_survived['Survived']['count'])



# A table with subsets of sex and is/isn't a child, with sum (number of survivals) and count (total number) for each subset:

gender_age_subsets = merged_set[['Sex','Child', 'Survived']].groupby(['Sex', 'Child']).agg(['sum', 'count'], axis="columns")  

# The percentage of survivals of each subset from the above:

gender_age_subsets['Survived']['sum']/gender_age_subsets['Survived']['count']

merged_set['Fare_level'] = np.zeros_like(merged_set["Fare"])

merged_set['Fare_level'][merged_set["Fare"] < 10] = 0

merged_set['Fare_level'][(merged_set["Fare"] < 20) & (merged_set["Fare"] >= 10)] = 1

merged_set['Fare_level'][(merged_set["Fare"] < 30) & (merged_set["Fare"] >= 20)] = 2

merged_set['Fare_level'][merged_set["Fare"] >= 30] = 3



fare_level_survived = merged_set[['Fare_level', 'Survived']].groupby('Fare_level').agg(['sum', 'count'], axis="columns")

fare_level_survived['Survived']['sum']/fare_level_survived['Survived']['count']



fare_level_survived = merged_set[['Fare_level', 'Sex', 'Survived', 'Pclass']].groupby(['Sex', 'Fare_level', 'Pclass']).agg(['sum', 'count'], axis="columns")

print(fare_level_survived)

fare_level_survived['Survived']['sum']/fare_level_survived['Survived']['count']
splitted = [name.split('. ')[0] for name in merged_set["Name"]]

titles = set([name.split(', ')[-1] for name in splitted])

print(titles)



to_remove = set(['Capt', 'Don', 'Major', 'Sir','Dona', 'Lady', 'the Countess', 'Jonkheer', 'Mlle', 'Mlle'])

titles -= to_remove



titles_united = {'Sir': ['Capt', 'Don', 'Major', 'Sir'], 'Lady': ['Dona', 'Lady', 'the Countess', 'Jonkheer'], 'Mme': ['Mme', 'Mlle']}

for title in titles:

    merged_set[title] = list(map(lambda name: int(title in name), merged_set["Name"]))

for key in titles_united:

    merged_set[key] = list(map(lambda name: int(name.split('. ')[0].split(', ')[-1] in key), merged_set["Name"]))
merged_set['Family_size'] = merged_set['SibSp'] + merged_set['Parch'] + 1

merged_set['No_family'] = (merged_set['Family_size']==1).astype(int)
splitted = [name.split('. ')[0] for name in merged_set["Name"]]

surnames = [name.split(', ')[0] for name in splitted]



merged_set['FamilyID'] = surnames + merged_set['Family_size'].astype(str)

merged_set['FamilyID'][merged_set['Family_size']<=2] = 'small'



merged_set.groupby('FamilyID')['PassengerId'].count()
merged_set["CA"] = list(map(lambda ticket: int(("CA" in ticket) or ("C.A" in ticket)), merged_set["Ticket"]))

merged_set["PC"] = list(map(lambda ticket: int("PC" in ticket), merged_set["Ticket"]))

merged_set["SOC"] = list(map(lambda ticket: int("S.O.C" in ticket), merged_set["Ticket"]))



# merged_set.groupby("Ticket")["Name"].count().sort_values(ascending=False)
# merged_set["Cabin_type"] = np.zeros_like(merged_set["Cabin"])

# merged_set["A_cabin"] = list(map(lambda cabin: int("A" in cabin), merged_set["Cabin"].astype(str)))

# merged_set["B_cabin"] = list(map(lambda cabin: int("B" in cabin), merged_set["Cabin"].astype(str)))

# merged_set["C_cabin"] = list(map(lambda cabin: int("C" in cabin), merged_set["Cabin"].astype(str)))

# merged_set["D_cabin"] = list(map(lambda cabin: int("D" in cabin), merged_set["Cabin"].astype(str)))

# merged_set["E_cabin"] = list(map(lambda cabin: int("E" in cabin), merged_set["Cabin"].astype(str)))

# merged_set["F_cabin"] = list(map(lambda cabin: int("F" in cabin), merged_set["Cabin"].astype(str)))

# merged_set["Cabin_type"][merged_set["A_cabin"] == 1] = 1

# merged_set["Cabin_type"][merged_set["B_cabin"] == 1] = 2

# merged_set["Cabin_type"][merged_set["C_cabin"] == 1] = 3

# merged_set["Cabin_type"][merged_set["D_cabin"] == 1] = 4

# merged_set["Cabin_type"][merged_set["E_cabin"] == 1] = 5

# merged_set["Cabin_type"][merged_set["F_cabin"] == 1] = 6

# sns.stripplot(x="Survived", y="Cabin_type", data=merged_set)

ids = merged_set['PassengerId'][merged_set['Survived'].isnull()]

merged_set = merged_set.drop(['Cabin', 'PassengerId'], axis=1)

merged_set['Embarked'][merged_set['Embarked'].isnull()] = 'S'

merged_set['Fare'][merged_set['Fare'].isnull()] = merged_set['Fare'].mean()
merged_noncategorical = pd.get_dummies(merged_set, columns=['Embarked', 'Name', 'Sex', 'Ticket', 'FamilyID'])
dt_model = DecisionTreeRegressor(random_state=0)

age_train_set = merged_noncategorical[merged_noncategorical["Age"].notnull()].drop(["Age", "Survived"],axis=1)

y = merged_noncategorical["Age"][merged_noncategorical["Age"].notnull()]

dt_model.fit(age_train_set, y)

age_test_set = merged_noncategorical[merged_noncategorical["Age"].isnull()].drop(["Age", "Survived"],axis=1)

merged_noncategorical["Age"][merged_noncategorical["Age"].isnull()] = dt_model.predict(age_test_set)
test_set = merged_noncategorical[merged_noncategorical['Survived'].isnull()]

train_set = merged_noncategorical[merged_noncategorical['Survived'].notnull()]
y = train_set.loc[:, 'Survived']

X = train_set.drop(['Survived'], axis=1)

test_set = test_set.drop(['Survived'], axis=1)

X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.33)
rf_model = RandomForestClassifier(random_state=1)

rf_model.fit(X, y)



# predictions = rf_model.predict(X_val)

# print(predictions)

# loss = zero_one_loss(Y_val.values, predictions)

# print("Validation loss for rf model is: {}".format(loss))
predictions = (rf_model.predict(test_set)).astype(int)



submission = pd.concat([ids, pd.Series(predictions, name="Survived")], axis=1)

print(submission.head(15))

submission.to_csv("submission.csv", index=False)
print(os.listdir("../working"))