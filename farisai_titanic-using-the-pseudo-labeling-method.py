import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
# This creates a pandas dataframe and assigns it to the titanic variable.

titanic = pd.read_csv("../input/train.csv")

# Print the first 5 rows of the dataframe.

titanic.head()
#the info method displays the data type and null not null and how many entries there

titanic.info()
#This can tell us how many missing values are there in the dataset

titanic.isnull().sum()

#Cabin seems to have the most of the missing values 

#Age has 177, we should know that replacing missing ages by the mean or median will result in

#a less accurate estimations
df = titanic.drop(['PassengerId','Name', 'Ticket', 'Cabin'], axis =1)

df.head(5)
df[df['Embarked'].isnull()]
sns.boxplot(x="Embarked", y="Fare", hue="Pclass", data=df)
# we replace the missing values in the embarked to "C"

df["Embarked"] = df["Embarked"].fillna('C')

df.head(5)
sns.factorplot(x="Pclass", y="Age", hue="Sex", data=df, size=6, kind="bar", palette="muted")
sns.factorplot(x="Embarked", y="Age", hue="Sex", data=df, size=6, kind="bar", palette="muted")
sns.boxplot(x="Embarked", y="Age", hue="Pclass", data=df)
#Getting Embarked S data into separte frame and work to calculate Median for each class

s = df.loc[df["Embarked"] == "S"]

v = [1, 2, 3]

for i in v:

    ss = s.where(s["Pclass"] == i)

    print ("Median age of class ",i, " = ",ss["Age"].median())
#Getting Embarked C data into separte frame and work to calculate Median for each class

c = df.loc[df["Embarked"] == "C"]

v = [1, 2, 3]

for i in v:

    sc = c.where(c["Pclass"] == i)

    print ("Median age of class ",i, " = ",sc["Age"].median())
#Getting Embarked Q data into separte frame and work to calculate Median for each class

q = df.loc[df["Embarked"] == "Q"]

v = [1, 2, 3]

for i in v:

    sq = q.where(q["Pclass"] == i)

    print ("Median age of class ",i, " = ",sq["Age"].median())
#Embark S

s1 = df[(df["Pclass"] == 1) & (df['Embarked'] == "S") & (df['Age'].isnull())].fillna(37)

s2 = df[(df["Pclass"] == 2) & (df['Embarked'] == "S") & (df['Age'].isnull())].fillna(30)

s3 = df[(df["Pclass"] == 3) & (df['Embarked'] == "S") & (df['Age'].isnull())].fillna(25)



#Embark C

c1 = df[(df["Pclass"] == 1) & (df['Embarked'] == "C") & (df['Age'].isnull())].fillna(37.5)

c2 = df[(df["Pclass"] == 2) & (df['Embarked'] == "C") & (df['Age'].isnull())].fillna(25)

c3 = df[(df["Pclass"] == 3) & (df['Embarked'] == "C") & (df['Age'].isnull())].fillna(20)



#Embark Q

q1 = df[(df["Pclass"] == 1) & (df['Embarked'] == "Q") & (df['Age'].isnull())].fillna(38.5)

q2 = df[(df["Pclass"] == 2) & (df['Embarked'] == "Q") & (df['Age'].isnull())].fillna(43.5)

q3 = df[(df["Pclass"] == 3) & (df['Embarked'] == "Q") & (df['Age'].isnull())].fillna(21.5)
#Concatinating all the sub-frames into one frame with replaced Age values

scq = pd.concat([s1,s2,s3,c1,c2,c3,q1,q2,q3])

len(scq) #177 rows which equals to the missing values in the age column
#We drop records of missing vlaues

df = df.dropna(axis = 0, how = 'any')

#we will be left with only records with non-null values
#Now we concatinate the scq (replaced values frame) with main frame

data = pd.concat([df, scq])



#Checking the info to make sure we have same number of records as the original one we started with

data.info()

from sklearn.preprocessing import LabelEncoder

number = LabelEncoder()



data['Sex'] = number.fit_transform(data['Sex'].astype('str'))

data['Embarked'] = number.fit_transform(data['Embarked'].astype('str'))
#Check our Dataframe

data.head()
titanictest = pd.read_csv("../input/test.csv")

titanictest.head()
titanictest.info()
#Drop Columns

Ttest = titanictest.drop(['Name', 'Ticket', 'Cabin'], axis =1)

Ttest.head()
#Fill Missing values

Ttest[Ttest['Fare'].isnull()]
#calculate median fare

m = Ttest["Fare"].median()



#Since its the only one, we can just replace it with the mean Fare

Ttest["Fare"] = Ttest["Fare"].fillna(m)



#Embarked S

s = Ttest.loc[Ttest["Embarked"] == "S"]

v = [1, 2, 3]

for i in v:

    ss = s.where(s["Pclass"] == i)

    print ("Median age of class ",i, " = ",ss["Age"].median())
#Getting Embarked C data into separte frame and work to calculate Median for each class

c = Ttest.loc[Ttest["Embarked"] == "C"]

v = [1, 2, 3]

for i in v:

    sc = c.where(c["Pclass"] == i)

    print ("Median age of class ",i, " = ",sc["Age"].median())
#Getting Embarked Q data into separte frame and work to calculate Median for each class

q = Ttest.loc[Ttest["Embarked"] == "Q"]

v = [1, 2, 3]

for i in v:

    sq = q.where(q["Pclass"] == i)

    print ("Median age of class ",i, " = ",sq["Age"].median())
#Embark S

s1 = Ttest[(Ttest["Pclass"] == 1) & (Ttest['Embarked'] == "S") & (Ttest['Age'].isnull())].fillna(42)

s2 = Ttest[(Ttest["Pclass"] == 2) & (Ttest['Embarked'] == "S") & (Ttest['Age'].isnull())].fillna(26)

s3 = Ttest[(Ttest["Pclass"] == 3) & (Ttest['Embarked'] == "S") & (Ttest['Age'].isnull())].fillna(24)



#Embark C

c1 = Ttest[(Ttest["Pclass"] == 1) & (Ttest['Embarked'] == "C") & (Ttest['Age'].isnull())].fillna(43)

c2 = Ttest[(Ttest["Pclass"] == 2) & (Ttest['Embarked'] == "C") & (Ttest['Age'].isnull())].fillna(27)

c3 = Ttest[(Ttest["Pclass"] == 3) & (Ttest['Embarked'] == "C") & (Ttest['Age'].isnull())].fillna(21)



#Embark Q

q1 = Ttest[(Ttest["Pclass"] == 1) & (Ttest['Embarked'] == "Q") & (Ttest['Age'].isnull())].fillna(37)

q2 = Ttest[(Ttest["Pclass"] == 2) & (Ttest['Embarked'] == "Q") & (Ttest['Age'].isnull())].fillna(61)

q3 = Ttest[(Ttest["Pclass"] == 3) & (Ttest['Embarked'] == "Q") & (Ttest['Age'].isnull())].fillna(24)



scq = pd.concat([s1,s2,s3,c1,c2,c3,q1,q2,q3])



Ttest = Ttest.dropna(axis = 0, how = 'any')



Testdata = pd.concat([Ttest, scq])

TestdataID = pd.DataFrame(Testdata['PassengerId'], columns = ['PassengerId'])



Testdata = Testdata.drop(['PassengerId'], axis =1)

Testdata.info()
#Converting our categorical data inot numeric

Testdata['Sex'] = number.fit_transform(Testdata['Sex'].astype('str'))

Testdata['Embarked'] = number.fit_transform(Testdata['Embarked'].astype('str'))
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']

Y_data = data['Survived'].values

X_data = data[list(features)].values

X_test = Testdata[list(features)].values
from sklearn import svm

lin_clf = svm.LinearSVC()

lin_clf.fit(X_data, Y_data) 
#we predict the survival of the testdata and save it

pseudoY_test = lin_clf.predict(Testdata)


X = np.vstack((X_data, X_test))

Y = np.concatenate((Y_data, pseudoY_test), axis=0)



pseudo_model = svm.LinearSVC()

pseudo_model.fit(X, Y)
Accuracyclf = lin_clf.score(X_data, Y_data)

print ("Accuracy of the lin_clf model: ", Accuracyclf*100, "%")



Accuracypseudo = lin_clf.score(X, Y)

print ("Accuracy of the lin_clf model: ", Accuracypseudo*100, "%")
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

from sklearn.svm import LinearSVC, SVC

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression, LinearRegression

from sklearn.tree import DecisionTreeClassifier



classifiers = [

    KNeighborsClassifier(),

    SVC(),

    DecisionTreeClassifier(),

    RandomForestClassifier(),

    AdaBoostClassifier(),

    GradientBoostingClassifier(),

    GaussianNB(),

    LogisticRegression(),

    LinearSVC()]



for clf in classifiers:

    name = clf.__class__.__name__

    clf.fit(X, Y)

    accurracy = clf.score(X, Y)

    print(accurracy, name)
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import cross_val_score

from sklearn.ensemble import AdaBoostClassifier



clf = AdaBoostClassifier(n_estimators=500)

scores = cross_val_score(clf, X, Y)

scores.mean()

clf.fit(X, Y)



Accuracy = clf.score(X, Y)

print ("Accuracy in the training data: ", Accuracy*100, "%")



prediction = clf.predict(X_test)

dfPrediction = pd.DataFrame(data=prediction,columns=['Survived'])

dfsubmit = pd.concat([TestdataID['PassengerId'], dfPrediction['Survived']], axis = 1, join_axes=[TestdataID['PassengerId'].index])

dfsubmit = dfsubmit.reset_index(drop=True)

TestPredict = dfsubmit.to_csv('TestPredictADABOOST.csv')