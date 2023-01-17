import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
train_results = pd.read_csv('../input/train.csv',nrows=9999) 
train_results
survivors = train_results.loc[train_results['Survived'] == 1]
lost_souls = train_results.loc[train_results['Survived'] == 0]
survivors.describe()
lost_souls.describe()
females = train_results.loc[train_results['Sex'] == 'female']
females.describe()
female_survivors = females.loc[females['Survived'] == 1]
female_survivors.mean()
males = train_results.loc[train_results['Sex'] == 'male']
male_survivors = males.loc[males['Survived'] == 1]
males.mean()
male_survivors.mean()

from sklearn.linear_model import LinearRegression

train = pd.read_csv('../input/train.csv')
#Lets convert sex to dummies

#df = pd.DataFrame(data=my_data, columns=['y', 'dummy', 'x'])
just_dummies = pd.get_dummies(train['Sex'])

train = pd.concat([train, just_dummies], axis=1)      

#Now that we have our dummies, let's nix the sex column and names
del train['Sex']
del train['Name']
del train['Ticket']
del train['Cabin']
#del train['Parch']
del train['Embarked']
del train['SibSp']

#Let's get rid of those without an age assigned.
train = train.dropna(subset=['Age'])


train['isChild'] = np.where(train['Age']<=12, '1', '0')


dataY = train.iloc[:,1]
dataX = train.iloc[:,2:9]
from sklearn.preprocessing import StandardScaler
train_scale = StandardScaler()
dataX = train_scale.fit_transform(dataX)
from sklearn.svm import LinearSVC, SVC

C=0.5

#(C=1.0, tol=0.0001, max_iter=1000, penalty='l2', loss='squared_hinge', dual=True, multi_class='ovr', fit_intercept=True, intercept_scaling=1
clf_1 = LinearSVC(C=C, tol=0.0001, max_iter=1000, penalty='l2', loss='squared_hinge', dual=True, multi_class='ovr', fit_intercept=True, intercept_scaling=1).fit(dataX, dataY)  # possible to state loss='hinge'
clf_2 = SVC(kernel='linear').fit(dataX, dataY)
clf_3 = SVC(kernel='rbf', gamma=0.5, C=C).fit(dataX, dataY)
clf_4 = SVC(kernel='poly', degree=3, C=C).fit(dataX, dataY)

fit_1 = clf_1.fit(dataX, dataY)
fit_2 = clf_2.fit(dataX, dataY)

score_1 = clf_1.score(dataX, dataY)
score_2 = clf_2.score(dataX, dataY)

print()
print('Test score %s' % score_1)
print('Test score %s' % score_2)

test = pd.read_csv('../input/test.csv',nrows=9999) #, names=['source','text''faves','id'])

#Lets convert sex to dummies

#df = pd.DataFrame(data=my_data, columns=['y', 'dummy', 'x'])
just_dummies = pd.get_dummies(test['Sex'])

test = pd.concat([test, just_dummies], axis=1)      

#Now that we have our dummies, let's nix the sex column and names
del test['Sex']
del test['Name']
del test['Ticket']
del test['Cabin']
#del train['Parch']
del test['Embarked']
del test['SibSp']



test['isChild'] = np.where(test['Age']<=8, '1', '0')

#print(train)

testX = test.iloc[:,1:]
testX['Age'].fillna(testX['Age'].mean(),inplace=True)
testX['Fare'].fillna(testX['Fare'].mean(),inplace=True)
print(testX)
predY = clf_2.predict(testX)
predictions = predY
#Create a  DataFrame with the passengers ids and our prediction regarding whether they survived or not
submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':predictions})

#Visualize the first 5 rows
submission.head()
#Convert DataFrame to a csv file that can be uploaded
#This is saved in the same directory as your notebook
filename = 'TeamBallard.csv'

submission.to_csv(filename,index=False)

print('Saved file: ' + filename)
