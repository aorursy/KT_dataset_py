#laod the data
import pandas as pd
data=pd.read_csv('../input/train.csv', encoding="utf-8")
data.head()
#dealing with NaNs
data.Age = data.Age.astype(float).fillna(data['Age'].median())
data.Fare = data.Fare.astype(float).fillna(data['Fare'].median())
data.SibSp = data.SibSp.astype(float).fillna(data['SibSp'].median())
data.head()
# Data exploration
#it looks like females survived more than males
import matplotlib.pyplot as plt
import seaborn as sns

ax = sns.barplot(y=data['Survived'],x=data['Sex'])
plt.show()

#People who join on Cherbourg had more chance to survive
ax = sns.barplot(y=data['Survived'],x=data['Embarked'])
plt.show()
#People who bought first class tickets has more chance to survive
ax = sns.barplot(y=data['Survived'],x=data['Pclass'])
plt.show()
# The more relatives a passenger has, the less was his changes to survive (excluding passenger with no relatives aboard)
ax = sns.barplot(y=data['Survived'],x=data['SibSp'])
plt.show()
# No connection was found between the chances to survive and the number of parents/children of a passenger
ax = sns.barplot(y=data['Survived'],x=data['Parch'])
plt.show()
#Most of the passengers paid a relatively small amount for an ticket (around 32$)

print('Average fare cost: '+str(data['Fare'].mean()))
print('Maximum fare cost: '+str(data['Fare'].max()))
data.Fare.plot(kind='kde')
plt.show()

#As the fare cost rises, the chance to survive increased 
import numpy as np
print('Correlation: '+ str(np.corrcoef(data['Survived'], data['Fare'])[0, 1]))
ax = sns.regplot(y=data['Fare'],x=data['Survived'])
plt.show()
# Most of the passengers where arount thier 30
print('Average age: '+str(data['Age'].mean()))
print('Oldest passenger: '+str(data['Age'].max()))
data.Age.plot(kind='kde')
plt.show()

# As the age rises, the chance to survive decrease, but not dramatically. 

print('Correlation: '+ str(np.corrcoef(data['Survived'], data['Age'])[0, 1]))
ax = sns.regplot(y=data['Age'],x=data['Survived'])
# ax = sns.regplot(y=data['Age'],x=data['Survived'])
plt.show()
# The age distribution does not change dramatically when you examine females and males separately
data.groupby('Sex').Age.plot(kind='kde')
plt.show()
#Convert category variables to factors  
data['Embarked_factorized'] = data['Embarked'].factorize()[0]
data['Sex_factorized'] = data['Sex'].factorize()[0]
data.head()
# Let's build prediction models 
#Logistic regresion:
X=data.drop(['PassengerId','Survived', 'Sex','Name','Ticket','Cabin','Embarked'], axis=1)
y=data['Survived']
X.head()
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()

print(result.summary2())

# splitting to test and train
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# 80%, not bad...
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
# let's try other classifier
# Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train)
# 78%, not bad...
y_pred = gnb.predict(X_test)
print('Accuracy of Naive Bayes classifier on test set: {:.2f}'.format(gnb.score(X_test, y_test)))
# let's try other classifier 
# Random forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(X_train, y_train)
# 44%, not so good...
y_pred = rf.predict(X_test)
print('Accuracy of Random forest classifier on test set: {:.2f}'.format(rf.score(X_test, y_test)))
# let's try other classifier 
# KNN
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=1, n_jobs=-1) 
neigh.fit(X_train, y_train)
# 71%, not bad...
y_pred = neigh.predict(X_test)
print('Accuracy of Random forest classifier on test set: {:.2f}'.format(neigh.score(X_test, y_test)))
# let's try other classifier 
# SVC
from sklearn.svm import SVC
svc = SVC() 
svc.fit(X_train, y_train)
# 74%, not bad...
y_pred = svc.predict(X_test)
print('Accuracy of Random forest classifier on test set: {:.2f}'.format(svc.score(X_test, y_test)))
# let's try other classifier 
# SGD
from sklearn.linear_model import SGDClassifier
sgd=SGDClassifier()
sgd.fit(X_train, y_train)
# 72%, not bad...
y_pred = sgd.predict(X_test)
print('Accuracy of SGD classifier on test set: {:.2f}'.format(sgd.score(X_test, y_test)))
# Logistic regression made the best prediction
# Let's make the prediction on the test file
test=pd.read_csv('../input/test.csv', encoding="utf-8")
test.head()
#dealing with NaNs
test.Age = test.Age.astype(float).fillna(test['Age'].median())
test.Fare = test.Fare.astype(float).fillna(test['Fare'].median())
test.SibSp = test.SibSp.astype(float).fillna(test['SibSp'].median())
#Convert category variables to factors  
test['Embarked_factorized'] = test['Embarked'].factorize()[0]
test['Sex_factorized'] = test['Sex'].factorize()[0]
test.head()
test=test.drop(['PassengerId', 'Sex','Name','Ticket','Cabin','Embarked'], axis=1)
test.head()
# Prediction
y_pred = logreg.predict(test)
y_pred
# create a file for submission
submision=pd.read_csv('../input/test.csv', encoding="utf-8")
submision=submision.drop(['Pclass','Age','Sex','Name','Ticket','Cabin','Embarked','Fare','Parch','SibSp','Pclass'], axis=1)
submision['Survived']=y_pred
submision.head()
# Export the CSV file
submision.to_csv('Titanic submision.csv')
