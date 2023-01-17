import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
train_df = pd.read_csv('../input/train.csv', index_col = 0)
test_df = pd.read_csv('../input/test.csv', index_col = 0)
train_df.columns
print(train_df.isnull().sum())
print('-'*20)
print(test_df.isnull().sum())
y = train_df.Survived.value_counts()
print(y)
sns.barplot(y = y, x = y.index);
sex = train_df.groupby(['Sex','Survived'])['Survived'].count()
print (sex)
sns.barplot(x = sex['female'].index, y = sex['female']);
sns.barplot(x = sex['male'].index, y = sex['male']);
sns.countplot('Pclass',hue='Survived',data=train_df);
sns.factorplot('Pclass','Survived',hue='Sex',data=train_df)
plt.show()
print('Oldest Passenger was of:', train_df['Age'].max(),'Years')
print('Youngest Passenger was of:',train_df['Age'].min(),'Years')
print('Average Age on the ship:', train_df['Age'].mean(),'Years')
print('Oldest Passenger was of:', test_df['Age'].max(),'Years')
print('Youngest Passenger was of:',test_df['Age'].min(),'Years')
print('Average Age on the ship:', test_df['Age'].mean(),'Years')
sns.violinplot("Pclass","Age", hue="Survived", data=train_df,split=True)
plt.title('PClass & Age vs Survival')
plt.show()
sns.violinplot("Sex","Age", hue="Survived", split = True, data=train_df)
plt.title("Sex and Age vs Survival")
plt.show()
train_df['Title'] = 0
for i in train_df:
    train_df['Title']=train_df.Name.str.extract('([A-Za-z]+)\.') 

print(train_df.groupby(train_df["Title"]).size())

print('-'*20)

test_df['Title'] = 0
for i in test_df:
    test_df['Title']=test_df.Name.str.extract('([A-Za-z]+)\.')
    
print(test_df.groupby(test_df["Title"]).size())

print('-'*20)
train_df['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mr'],inplace=True)
test_df['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess','Jonkheer','Col','Rev','Capt','Sir','Dona'],['Miss','Miss','Miss','Mr','Mr','Mrs','Mrs','Other','Other','Other','Mr','Mr','Mrs'],inplace=True)

print(train_df.groupby('Title')['Age'].mean())
print('-'*30)
print(test_df.groupby('Title')['Age'].mean())
train_df.loc[(train_df.Age.isnull())&(train_df.Title=='Mr'),'Age']=33
train_df.loc[(train_df.Age.isnull())&(train_df.Title=='Mrs'),'Age']=36
train_df.loc[(train_df.Age.isnull())&(train_df.Title=='Master'),'Age']=5
train_df.loc[(train_df.Age.isnull())&(train_df.Title=='Miss'),'Age']=22
train_df.loc[(train_df.Age.isnull())&(train_df.Title=='Other'),'Age']=46

train_df['Age_band']=0

train_df.loc[train_df['Age']<=16,'Age_band']=0
train_df.loc[(train_df['Age']>16)&(train_df['Age']<=32),'Age_band']=1
train_df.loc[(train_df['Age']>32)&(train_df['Age']<=48),'Age_band']=2
train_df.loc[(train_df['Age']>48)&(train_df['Age']<=64),'Age_band']=3
train_df.loc[train_df['Age']>64,'Age_band']=4

#train_df = train_df.drop(['Age'], axis = 1)

test_df.loc[(test_df.Age.isnull())&(test_df.Title=='Mr'),'Age']=33
test_df.loc[(test_df.Age.isnull())&(test_df.Title=='Mrs'),'Age']=39
test_df.loc[(test_df.Age.isnull())&(test_df.Title=='Master'),'Age']=8
test_df.loc[(test_df.Age.isnull())&(test_df.Title=='Miss'),'Age']=22
test_df.loc[(test_df.Age.isnull())&(test_df.Title=='Other'),'Age']=43



test_df['Age_band']=0

test_df.loc[test_df['Age']<=16,'Age_band']=0
test_df.loc[(test_df['Age']>16)&(test_df['Age']<=32),'Age_band']=1
test_df.loc[(test_df['Age']>32)&(test_df['Age']<=48),'Age_band']=2
test_df.loc[(test_df['Age']>48)&(test_df['Age']<=64),'Age_band']=3
test_df.loc[test_df['Age']>64,'Age_band']=4

print(train_df.isnull().sum())
print('-'*20)
print(test_df.isnull().sum())
test_df['Fare'].fillna(test_df.Fare.mean(), inplace=True)
test_df.isnull().sum()
sns.heatmap(train_df.corr(), annot = True);
test_df.head()
test_df = test_df.drop(['Name', 'Ticket', 'Cabin', 'Title', 'Age'], axis = 1)
np.shape(test_df)
test_df.head()
train_df = train_df.drop(['Name','Ticket','Cabin', 'Age'], axis = 1)
train_df.head()
sns.factorplot('Pclass','Survived',col='Title',data=train_df)
plt.show()
sns.factorplot('Embarked','Survived',data=train_df);
sns.factorplot('Pclass','Survived',hue='Sex',col='Embarked',data=train_df)
plt.show()
train_df = train_df.drop(['Title'], axis = 1)
print(train_df.Embarked.isnull().sum())
print(test_df.Embarked.isnull().sum())
print(train_df.Embarked.value_counts())
print(test_df.Embarked.value_counts())
train_df['Embarked'].fillna('S', inplace=True)
test_df['Embarked'].fillna('S', inplace=True)
train_df['Sex'].replace(['male','female'],[0,1],inplace=True)
train_df['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
train_df.head(20)
train_df[["SibSp", "Survived"]].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived', ascending=False)
train_df[["Parch", "Survived"]].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived', ascending=False)
pd.crosstab([train_df.SibSp], train_df.Survived)
sns.factorplot('SibSp','Survived',data=train_df);
pd.crosstab(train_df.SibSp,train_df.Pclass)
sns.factorplot('Age_band','Survived', data = train_df)
train_df['Fare_Range']=pd.qcut(train_df['Fare'],4)
train_df.groupby(['Fare_Range'])['Survived'].mean()
train_df['Fare_cat']=0

train_df.loc[train_df['Fare']<=7.91,'Fare_cat']=0
train_df.loc[(train_df['Fare']>7.91)&(train_df['Fare']<=14.454),'Fare_cat']=1
train_df.loc[(train_df['Fare']>14.454)&(train_df['Fare']<=31),'Fare_cat']=2
train_df.loc[(train_df['Fare']>31)&(train_df['Fare']<=513),'Fare_cat']=3

test_df['Fare_cat']=0

test_df.loc[test_df['Fare']<=7.91,'Fare_cat']=0
test_df.loc[(test_df['Fare']>7.91)&(test_df['Fare']<=14.454),'Fare_cat']=1
test_df.loc[(test_df['Fare']>14.454)&(test_df['Fare']<=31),'Fare_cat']=2
test_df.loc[(test_df['Fare']>31)&(test_df['Fare']<=513),'Fare_cat']=3
train_df = train_df.drop(['Fare', 'Fare_Range', 'Title'], axis = 1)
test_df = test_df.drop(['Fare'], axis = 1)
print(test_df.head())
print(train_df.head())
test_df['Embarked'].replace(['S','C','Q'],[0,1,2],inplace=True)
test_df['Sex'].replace(['male','female'],[0,1],inplace=True)
test_df.head()
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.metrics import confusion_matrix
y_train = train_df['Survived']
x_train = train_df.drop(['Survived'], axis = 1)
x_test = test_df
x_train.shape, y_train.shape, x_test.shape
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)

acc_log = round(logreg.score(x_train, y_train) * 100, 2)
acc_log
coef_df =  pd.DataFrame(train_df.columns.delete(0))
coef_df.columns = ['Feature']
coef_df["Correlation"] = pd.Series(logreg.coef_[0])
coef_df.sort_values(by='Correlation', ascending=False)

from sklearn.svm import SVC
classifier_svc = SVC()
classifier_svc.fit(x_train, y_train)
y_pred_svc = classifier_svc.predict(x_test)

acc_svc = round(classifier_svc.score(x_train, y_train) * 100, 2)
acc_svc
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_test)
acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)
acc_gaussian
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
y_pred = knn.predict(x_test)
acc_knn = round(knn.score(x_train, y_train) * 100, 2)
acc_knn
random_forest = RandomForestClassifier(n_estimators=100)
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)
random_forest.score(x_train, y_train)
acc_random_forest = round(random_forest.score(x_train, y_train) * 100, 2)
acc_random_forest
models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 
              'Random Forest', 'Naive Bayes'],
    'Score': [acc_svc, acc_knn, acc_log, acc_random_forest, acc_gaussian]})
models.sort_values(by='Score', ascending=False)
submission = pd.DataFrame({"Survived": y_pred})