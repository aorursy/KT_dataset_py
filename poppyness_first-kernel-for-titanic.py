import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
#Utility methods to do some fixes
def fixDS(dataset):
    gender = {'male': 1,'female': 2}
    city = {'S':0, 'C':1, 'Q':2}
    dataset['Embarked'].fillna(method='ffill', inplace=True)
    dataset.Sex = [gender[item] for item in dataset.Sex]
    dataset.Embarked = [city[item] for item in dataset.Embarked]
    return dataset
#Age has NaN. Rather than imputing it generically, let's replace it with median values for the type of passenger based on the title in their name which may indicate their age
def fixAge(dataset):
    #Find the median ages of boys (master), unmarried girls and women (miss), married women (Mrs) and men (Mr)
    boy = dataset['Name'].str.contains("Master", na=False)
    boyData = dataset[boy]
    medianBoyAge = boyData['Age'].median(axis=0)
    print ("The median boy age is: ", medianBoyAge)

    girl = dataset['Name'].str.contains("Miss", na=False)
    girlData = dataset[girl]
    medianGirlAge = girlData['Age'].median(axis=0)
    girlData['Age'].fillna(medianGirlAge, inplace=True, axis=0)

    print ("The median girl age is: ", medianGirlAge)

    woman = dataset['Name'].str.contains("Mrs", na=False)
    womanData = dataset[woman]
    medianWomanAge = womanData['Age'].median(axis=0)
    womanData['Age'].fillna(medianWomanAge, inplace=True, axis=0)

    print ("The median woman age is: ", medianWomanAge)

    man = dataset['Name'].str.contains("Mr", na=False)
    manData = dataset[man]
    medianManAge = manData['Age'].median(axis=0)
    manData['Age'].fillna(medianManAge, inplace=True, axis=0)

    print ("The median man age is: ", medianManAge)

    #Are there other labels?
    other = ~dataset['Name'].str.contains("Master|Miss|Mrs|Mr", na=False)
    otherData = dataset[other]
    medianOtherAge = otherData['Age'].median(axis=0)
    otherData['Age'].fillna(medianOtherAge, inplace=True, axis=0)

    print ("The median other age is: ", medianOtherAge)
    
    conditions = [dataset['Name'].str.contains("Mr"), dataset['Name'].str.contains("Mrs"), dataset['Name'].str.contains("Miss"),dataset['Name'].str.contains("Master"),~dataset['Name'].str.contains("Master|Miss|Mrs|Mr")]
    values = [medianManAge, medianWomanAge, medianGirlAge, medianBoyAge, medianOtherAge]

    # apply logic where company_type is null
    dataset['Age'] = np.where(dataset['Age'].isnull(),
                                  np.select(conditions, values),
                                  dataset['Age'])

    return dataset
#Now replace the values of each NaN depending on whether the Age belongs to a Boy, Girl, Woman, Man or other
trainData = pd.read_csv('../input/train.csv')
trainData = fixDS(trainData)
trainData = fixAge(trainData)
trainData.head()
#Let's drop useless columns, doesn't look like Ticket or Name have any useful information
trainData['Label'] = trainData['Survived']
trainData.drop(['Name', 'Ticket', 'Cabin','Survived'], axis=1, inplace=True)

y = trainData['Label']          # Split off classifications
X = trainData.iloc[:,0:8]
print (X.shape)
print (y.shape)
print(X.head())
from sklearn.metrics import accuracy_score
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
print (X_train.shape, y_train.shape)
print (X_test.shape, y_test.shape)
lm = linear_model.LogisticRegression()
model = lm.fit(X_train, y_train)
predictions = lm.predict(X_test)
print ("Score of Logistic Regression:", model.score(X_test, y_test))

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
score = accuracy_score(y_test, y_pred)
print("Score of Naive Gaussian Bayes:", score)

from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
y_pred = mnb.fit(X_train, y_train).predict(X_test)
score = accuracy_score(y_test, y_pred)
print("Score of Multinomial Bayes:", score)

from sklearn import svm
clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
score = accuracy_score(y_test, y_pred)
print("Score of SVM:", score)

from sklearn import tree
treeD = tree.DecisionTreeClassifier()
treeD = treeD.fit(X_train, y_train)
y_pred = treeD.predict(X_test)
score = accuracy_score(y_test, y_pred)
print("Score of Decision Tree:", score)


testData = pd.read_csv("../input/test.csv")
testData = fixDS(testData)
testData = fixAge(testData)
testData.drop(['Name', 'Ticket','Cabin'], axis=1, inplace=True)
testData.fillna(method='ffill', inplace = True)

print(testData.head())
print(testData.shape)
finalPred = gnb.predict(testData)
print(finalPred)

submission = pd.DataFrame({
        "PassengerId": testData["PassengerId"],
        "Survived": finalPred
    })

submission.to_csv("titanic.csv", index=False)
print(submission.shape)