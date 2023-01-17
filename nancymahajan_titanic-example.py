# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
from sklearn import preprocessing
from sklearn import tree
import subprocess
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
train = pd.read_csv("../input/train.csv", dtype={"Age": np.float64}, )
test = pd.read_csv("../input/test.csv", dtype={"Age": np.float64}, )

X_train, X_test  = train_test_split(train, test_size=0.02)
print(X_train.describe())

#print(test.head())
X = X_train.iloc[:,[2,4,5,6,7,9,11]].copy()
Y = X_train.iloc[:, 1]

accuracy = X_test.iloc[:, [0,1]].copy()
xText = X_test.iloc[:,[2,4,5,6,7,9,11]].copy()

finalTest = test.iloc[:,[1,3,4,5,6,8,10]].copy()

print(X.head(3))
print(Y.head(3))
print(finalTest.head(3))

def mapAgeToCategory(x):
    if x[0] < 10:
        return 0
    elif (x[0] >= 10 and x[0] < 20 ) :
        return 1
    elif (x[0] >= 20 and x[0] < 30 ):
        return 2
    elif (x[0] >= 30 and x[0] < 40):
        return 3
    elif (x[0] >= 40 and x[0] < 50) :
        return 4
    else:
        return 5
    

def mapFareToCategory(x):
    if x < 5:
       return 0
    elif x >= 5 and x < 10:
        return 1
    elif x >= 10 and x < 20:
        return 1
    elif x >= 20 and x < 30:
        return 2
    elif x >= 30 and x < 50:
        return 3
    elif x >=50 and x < 100:
        return 4
    else:
        return 5
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt
 
print(pd.DataFrame({'count' : X_train.groupby(['Embarked' ,'Survived']).size()}).reset_index())
embarked = pd.DataFrame({'count' : X_train.groupby(['Embarked' ,'Survived']).size()}).reset_index()
objects = pd.unique(embarked['Embarked'])

y_pos = np.arange(len(objects))
embarked_survived = embarked.loc[embarked['Survived'] == 1]
embarked_non_survived = embarked.loc[embarked['Survived'] == 0]

survived_non = embarked_non_survived['count']
survived = embarked_survived['count']

p1 = plt.bar(y_pos, survived_non, align='center', alpha=0.5)
p2 = plt.bar(y_pos, survived, align='center', alpha=0.5)

plt.xticks(y_pos, objects)
plt.ylabel('Non Survived')
plt.title('Port')

plt.legend((p1[0], p2[0]), ('non survived', 'survived'))

plt.show()
plt.bar(X_train.PassengerId,X_train.Fare)
plt.show()
from sklearn.metrics import classification_report
from sklearn.impute import SimpleImputer

#labelencoder
ohe = preprocessing.LabelEncoder()
ohe.fit(pd.unique(X.Embarked).astype(str))
X.Embarked = ohe.transform(X.Embarked.astype(str))
xText.Embarked  = ohe.transform(xText.Embarked.astype(str))
finalTest.Embarked  = ohe.transform(finalTest.Embarked.astype(str))

print(ohe.classes_)

#imputer = SimpleImputer(missing_values='NaN', strategy='most_frequent', axis=0)
#X.Embarked = imputer.fit_transform(X[['Embarked']])
#xText.Embarked = imputer.fit_transform(xText[['Embarked']])
#finalTest.Embarked = imputer.fit_transform(finalTest[['Embarked']])
'''
X.Embarked.fillna('S', inplace=True)
xText.Embarked.fillna('S', inplace=True)
finalTest.Embarked.fillna('S', inplace=True)
'''

X['dependent'] = np.add(X.SibSp, X.Parch)
xText['dependent'] = np.add(xText.SibSp, xText.Parch)
finalTest['dependent'] = np.add(finalTest.SibSp, finalTest.Parch)
X.Age.fillna(X.Age.mean(), inplace=True)
xText.Age.fillna(xText.Age.mean(), inplace=True)
finalTest.Age.fillna(finalTest.Age.mean(), inplace=True)
X.Fare.fillna(X.Fare.mean(), inplace=True)
xText.Fare.fillna(xText.Fare.mean(), inplace=True)
finalTest.Fare.fillna(finalTest.Fare.mean(), inplace=True)

#converting Sex into 0,1 format
le = preprocessing.LabelEncoder()
le.fit(pd.unique(X.Sex))
X.Sex = le.transform(X.Sex)
xText.Sex  = le.transform(xText.Sex)
finalTest.Sex  = le.transform(finalTest.Sex)


X.Age = X[['Age','Sex']].apply(mapAgeToCategory, axis=1)
xText.Age = xText[['Age', 'Sex']].apply(mapAgeToCategory, axis=1)
finalTest.Age = finalTest[['Age','Sex']].apply(mapAgeToCategory,  axis=1)

#X.Fare = X[['Fare']].applymap(mapFareToCategory)
#xText.Fare = xText[['Fare']].applymap(mapFareToCategory)
#finalTest.Fare = finalTest[['Fare']].applymap(mapFareToCategory)

#embarked data to label encoded
le1 = preprocessing.LabelEncoder()
le1.fit(pd.unique(X.Embarked))
X.Embarked = le1.transform(X.Embarked)
xText.Embarked  = le1.transform(xText.Embarked)
finalTest.Embarked  = le1.transform(finalTest.Embarked)

print(X.head(3))

X =  X.iloc[:,[0,1,2,5]]
xText = xText.iloc[:,[0,1,2,5]]
finalTest =  finalTest.iloc[:,[0,1,2,5]]

print(X.head(10))
'''
plt.plot( X.Age,Y, 'bo')
plt.ylabel('Age')
plt.show()
'''

#print(X.head(3))

#check corelation
#corelationCoeff = pearsonr(Y, X.Age)
#print(pearsonr(X.Pclass, X.Fare))
#print(pearsonr(X.Pclass, X.Embarked))
#print(pearsonr(X.Age, X.dependent))
#scaling the data
#sc_X = StandardScaler()
#X = sc_X.fit_transform(X)
#xText = sc_X.transform(xText)
#finalTest = sc_X.transform(finalTest)
from sklearn.tree import DecisionTreeRegressor
#desicionTreeclf = tree.DecisionTreeClassifier( criterion='entropy',random_state = 100, max_depth=6, min_samples_leaf=5)
#desicionTreeclf = desicionTreeclf.fit( X, Y)
#gave80%

#desicionTreeclf = RandomForestClassifier( criterion='entropy', min_samples_split=10)
#desicionTreeclf = desicionTreeclf.fit(X, Y)
#gave 78%
desicionTreeclf = DecisionTreeRegressor( criterion='mse',max_leaf_nodes=8, random_state=0,max_depth=6, min_samples_leaf=5)
desicionTreeclf = desicionTreeclf.fit(X, Y)

#plot the data(importances)
print(desicionTreeclf.feature_importances_)
print(X.columns.values)
f1 = plt.bar(X.columns.values, desicionTreeclf.feature_importances_)
plt.ylabel('importance')
plt.show()


isSurvived  = pd.DataFrame({'Survived' : (desicionTreeclf.predict(xText)),'PassengerId' : ( X_test['PassengerId'])})
finalIsSurvived = pd.DataFrame({'Survived' : (desicionTreeclf.predict(finalTest)),'PassengerId' : ( test['PassengerId'])})
isSurvived.Survived =isSurvived.Survived.round(0)
#df.a = df.a.astype(float)
isSurvived.Survived =isSurvived.Survived.astype(int)

finalIsSurvived.Survived =finalIsSurvived.Survived.round(0)
#df.a = df.a.astype(float)
finalIsSurvived.Survived =finalIsSurvived.Survived.astype(int)
print(isSurvived.head(1))
print(accuracy.head(1))
#print(finalIsSurvived.head(2))

#final.Survived = final.Survived.astype(int)
finalIsSurvived.to_csv('submit.csv', encoding='utf-8', index=False)
#print(isSurvived.count)
print(mean_absolute_error(accuracy.iloc[:,1], isSurvived.iloc[:,1]))
print(accuracy_score(accuracy.iloc[:,1], isSurvived.iloc[:,0]))
import graphviz 
#dot_data = tree.export_graphviz(desicionTreeclf, out_file=None) 
#graph = graphviz.Source(dot_data) 
#graph
#tree.export_graphviz(desicionTreeclf, out_file='tree.dot')
#subprocess.call(['dot', '-Tpdf', 'tree.dot', '-o' 'tree.pdf'])