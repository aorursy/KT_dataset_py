import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
df = pd.read_csv("../input/train.csv")

df.index = df.PassengerId

df.drop('PassengerId', axis = 1, inplace = True)

df.head()
df.boxplot(by = 'Survived', figsize = (10, 20))

plt.show()
df[['Age', 'Survived']].boxplot(by = "Survived")

plt.show()
ax = df[['Age', 'Survived', 'Sex']].groupby('Sex').boxplot(by = "Survived")

plt.show()
df[['Fare', 'Survived']].boxplot(by = "Survived")

plt.show()
df[['Pclass', 'Survived']].groupby('Pclass').mean()
df[['SibSp', 'Survived']].boxplot(by = 'Survived')

plt.show()
tempdf1 = df[['SibSp', 'Survived']].groupby('SibSp').count().merge(df[['SibSp', 'Survived']].groupby('SibSp').mean(), right_index = True, left_index = True)

tempdf1.columns = ['Count', 'Prob. Survived']

tempdf1
tempdf2 = df[['SibSp', 'Survived']].groupby('SibSp').count().merge(df[['SibSp', 'Survived']].groupby('SibSp').sum(), right_index = True, left_index = True)

tempdf2.columns = ['Count', 'Survived']

tempdf2.plot.bar(figsize = (10, 8))

plt.show()
tempdf3 = df[['Parch', 'Survived']].groupby('Parch').count().merge(df[['Parch', 'Survived']].groupby('Parch').mean(), right_index = True, left_index = True)

tempdf3.columns = ['Count', 'Ratio. Survived']

tempdf3
df['Family_Size'] = df.Parch + df.SibSp + 1
tempdf4 = df[['Family_Size', 'Survived']].groupby('Family_Size').count().merge(df[['Family_Size', 'Survived']].groupby('Family_Size').mean(), right_index = True, left_index = True)

tempdf4.columns = ['Count', 'Ratio. Survived']

tempdf4
tempdf6 = df[['Family_Size', 'Survived']].groupby('Family_Size').count().merge(df[['Family_Size', 'Survived']].groupby('Family_Size').sum(), right_index = True, left_index = True)

tempdf6.columns = ['Count', 'Survived']

tempdf6.plot.bar(figsize = (8, 5))

plt.show()
df.drop(['SibSp', 'Parch'], axis = 1, inplace = True)
df['Title'] = df['Name'].apply(lambda x: x.split(",")[1].split(" ")[1])

df.head()
tempdf5 = df[['Title', 'Survived']].groupby('Title').count().merge(df[['Title', 'Survived']].groupby('Title').mean(), right_index = True, left_index = True)

tempdf5.columns = ['Count', 'Ratio. Survived']

tempdf5
df.drop('Name', inplace = True, axis = 1)

df.head()
df['Cabin'] = df['Cabin'].fillna('No')

# Since all 3rd class passengers didnt have cabins

df.head()
df['Cabin_deck'] = df['Cabin'].apply(lambda x: x.split(" ")[-1][0] if x != "No" else "No")

df['Cabin_number'] = df['Cabin'].apply(lambda x: 0 if len(x) == 1 else int(x.split(" ")[-1][1:]) if x != "No" else 0)

df.head()
tempdf7 = df[['Cabin_deck', 'Survived']].groupby('Cabin_deck').count().merge(df[['Cabin_deck', 'Survived']].groupby('Cabin_deck').mean(), right_index = True, left_index = True)

tempdf7.columns = ['Count', 'Ratio. Survived']

tempdf7
tempdf8 = df[['Cabin_number', 'Survived']].groupby('Cabin_number').count().merge(df[['Cabin_number', 'Survived']].groupby('Cabin_number').mean(), right_index = True, left_index = True)

tempdf8.columns = ['Count', 'Ratio. Survived']

tempdf8
df['Cabin_numeric_range'] = df['Cabin_number'].apply(lambda x: str(int(x/10)) + "0 to " + str(int(x/10 + 1)) + "0" if x != 0 else "No Cabin")

df.head()
tempdf9 = df[['Cabin_numeric_range', 'Survived']].groupby('Cabin_numeric_range').count().merge(df[['Cabin_numeric_range', 'Survived']].groupby('Cabin_numeric_range').mean(), right_index = True, left_index = True)

tempdf9.columns = ['Count', 'Ratio Survived']

tempdf9
df.drop(['Cabin', 'Cabin_number'], inplace = True, axis = 1)

df.head()
df.drop('Ticket', inplace = True, axis = 1)

df.head()
tempdf10 = df[['Embarked', 'Survived']].groupby('Embarked').count().merge(df[['Embarked', 'Survived']].groupby('Embarked').mean(), right_index = True, left_index = True)

tempdf10.columns = ['Count', 'Ratio. Survived']

tempdf10
df['Male'] = df['Sex'].apply(lambda x: 1 if x == "male" else 0)

df.drop('Sex', inplace = True, axis = 1)

df.head()
df['Age'].fillna(np.mean(df.Age), inplace = True)
ndf = pd.get_dummies(df, columns = ['Embarked', 'Title', 'Cabin_deck', 'Cabin_numeric_range'])

ndf.head()
ndf.drop(['Cabin_numeric_range_No Cabin', 'Cabin_deck_No'], inplace = True, axis = 1)
ndf.columns
survived = ndf.Survived

ndf.drop('Survived', inplace = True, axis = 1)
ndf.head()
from sklearn.cross_validation import train_test_split as ttspl

df_train, df_test, out_train, out_test = ttspl(ndf, survived, test_size = 0.25)
from sklearn.neighbors import KNeighborsClassifier as KNN

for i in range(1, 20):

    knn = KNN(n_neighbors = i)

    knn.fit(df_train, out_train)

    print("Neighbors = " + str(i) + "\t Score: ",)

    print(knn.score(df_test, out_test))
from sklearn.naive_bayes import GaussianNB as GNB

gnb = GNB()

gnb.fit(df_train, out_train)

gnb.score(df_test, out_test)
from sklearn.naive_bayes import MultinomialNB as MNB

mnb = MNB()

mnb.fit(df_train, out_train)

mnb.score(df_test, out_test)
from sklearn.naive_bayes import BernoulliNB as BNB

bnb = BNB()

bnb.fit(df_train, out_train)

bnb.score(df_test, out_test)
from sklearn.cross_validation import cross_val_score as cvs

from sklearn.tree import DecisionTreeClassifier as dtree

tr = dtree()

cvs(tr, df_train, out_train, cv = 10)
for i in range(2, 20):

    tr = dtree(max_depth= i)

    print("Max Depth = " + str(i) + "\t Score: ")

    print(np.mean(cvs(tr, df_train, out_train, cv = 10)))

    print("\n")
x = []

y = []

for i in range(2, 20):

    x.append(i)

    tr = dtree(max_depth= i)

    y.append(np.mean(cvs(tr, df_train, out_train, cv = 10)))

    

p = plt.plot(x, y)

plt.show()

    
for i in range(2, 40):

    tr = dtree(max_leaf_nodes = i)

    print("Max Leaf Nodes = " + str(i) + "\t Score: ")

    print(np.mean(cvs(tr, df_train, out_train, cv = 10)))
x = []

y = []

for i in range(2, 100, 2):

    x.append(i)

    tr = dtree(max_leaf_nodes = i)

    y.append(np.mean(cvs(tr, df_train, out_train, cv = 10)))

    

p = plt.plot(x, y)

plt.show()
tr = dtree(max_leaf_nodes = 40)

tr.fit(df_train, out_train)

tr.score(df_test, out_test)