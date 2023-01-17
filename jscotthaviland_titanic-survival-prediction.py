#import the packages we'll be using and read in our datasets

import pandas as pd

import seaborn as sb

import matplotlib.pyplot as pp

from sklearn.metrics import classification_report



% matplotlib inline



df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_train.head()
#Lets take a look at survival rates by age and sex.



df_women_survived = df_train[(df_train['Sex'] == 'female') & (df_train['Survived'] == 1)]

df_women_survived.dropna(inplace = True)



df_women_perished = df_train[(df_train['Sex'] == 'female') & (df_train['Survived'] == 0)]

df_women_perished.dropna(inplace = True)



df_men_survived = df_train[(df_train['Sex'] == 'male') & (df_train['Survived'] == 1)]

df_men_survived.dropna(inplace = True)



df_men_perished = df_train[(df_train['Sex'] == 'male') & (df_train['Survived'] == 0)]

df_men_perished.dropna(inplace = True)



pp.figure(1)

pp.subplot(121)

pp.hist(df_women_survived['Age'], bins = 10, alpha = 0.5)

pp.hist(df_women_perished['Age'], bins = 10, alpha = 0.5)

pp.xlabel('Age')

pp.ylabel('Count')

pp.legend()

pp.title('Female Survival')



pp.subplot(122)

pp.hist(df_men_survived['Age'], bins = 10, alpha = 0.5)

pp.hist(df_men_perished['Age'], bins = 10, alpha = 0.5)

pp.xlabel('Age')

pp.title('Male Survival')



#sb.distplot(df_women_survived['Age'], bins = 20)

#sb.distplot(df_women_perished['Age'], bins = 20)

pp.show()
#...and then use seaborn to look at survival by passenger class and sex.



sb.factorplot("Pclass", "Survived", "Sex", data=df_train, kind="bar", palette="deep", legend=True)

pp.show()
df_train.describe(include = 'all')
df_train_label = df_train['Survived']

df_train.drop(['Survived','PassengerId','Name','Ticket','Cabin','Embarked'], axis = 1, inplace=True)





df_results = pd.DataFrame(df_test['PassengerId'])

df_test.drop(['PassengerId','Name','Ticket','Cabin','Embarked'], axis = 1, inplace=True)





df_train.describe(include='all')

df_test.info()
df_train['Age'].fillna(df_train['Age'].mean(), inplace=True)

df_test['Age'].fillna(df_test['Age'].mean(), inplace=True)

df_test['Fare'].fillna(df_test['Fare'].mean(), inplace=True)
df_train.info()
df_test.info()
df_train['female'] = pd.get_dummies(df_train['Sex'])['female']

df_test['female'] = pd.get_dummies(df_test['Sex'])['female']



df_train.drop(['Sex'], inplace=True, axis=1)

df_test.drop(['Sex'], inplace=True, axis=1)
from sklearn.naive_bayes import GaussianNB



clf = GaussianNB()

clf.fit(df_train,df_train_label)

pred = clf.predict(df_test)

pred_train = clf.predict(df_train)



print(classification_report(df_train_label, pred_train, digits = 4))
df_results['Survived'] = pd.DataFrame(pred)

df_results.to_csv('submissionGNB.csv', header=True,index=False)
from sklearn.linear_model import LogisticRegressionCV as lr



clf = lr(cv=5)

clf.fit(df_train,df_train_label)



pred = clf.predict(df_test)

pred_train = clf.predict(df_train)



print(classification_report(df_train_label, pred_train, digits = 4))
df_results['Survived'] = pd.DataFrame(pred)

df_results.to_csv('submissionLOG.csv', header=True,index=False)
from sklearn import tree



clf = tree.DecisionTreeClassifier()

clf.fit(df_train,df_train_label)

pred = clf.predict(df_test)

pred_train = clf.predict(df_train)



print(classification_report(df_train_label, pred_train, digits = 4))
df_results['Survived'] = pd.DataFrame(pred)

df_results.to_csv('submissionDTC.csv', header=True,index=False)
from sklearn.svm import SVC



clf = SVC(kernel="linear")

clf.fit(df_train,df_train_label)

pred = clf.predict(df_test)

pred_train = clf.predict(df_train)



print(classification_report(df_train_label, pred_train, digits = 4))
df_results['Survived'] = pd.DataFrame(pred)

df_results.to_csv('submissionSVMlin.csv', header=True,index=False)
from sklearn.svm import SVC



clf = SVC(kernel="rbf")

clf.fit(df_train,df_train_label)

pred = clf.predict(df_test)

pred_train = clf.predict(df_train)



print(classification_report(df_train_label, pred_train, digits = 4))
df_results['Survived'] = pd.DataFrame(pred)

df_results.to_csv('submissionSVMrad.csv', header=True,index=False)
from sklearn.neighbors import KNeighborsClassifier as knn



clf = knn(n_neighbors = 5, weights = 'distance')

clf.fit(df_train,df_train_label)

pred = clf.predict(df_test)

pred_train = clf.predict(df_train)



print(classification_report(df_train_label, pred_train, digits = 4))
df_results['Survived'] = pd.DataFrame(pred)

df_results.to_csv('submissionKNN.csv', header=True,index=False)