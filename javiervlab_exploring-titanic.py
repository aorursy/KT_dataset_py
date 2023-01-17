#Libraries to import



import numpy

import pandas

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline



sns.set_style("whitegrid")

sns.set_context("notebook", font_scale=1.5)
# Import dataset 



titanic_train = pandas.read_csv('../input/train.csv')

titanic_test = pandas.read_csv('../input/test.csv')
# the train information

titanic_train.info()
titanic_train.head()
titanic_train.describe()
titanic_train['SibSp'].value_counts()
titanic_train['Parch'].value_counts()
titanic_train['Survived'].value_counts().plot(kind='bar', title='Survival Counts')
survived = titanic_train['Survived'].value_counts()[0]/titanic_train.shape[0]

death = titanic_train['Survived'].value_counts()[1]/titanic_train.shape[0]

print("{0} {1:0.2f}".format("Survived % = ", survived))

print("{0} {1:0.2f}".format("Dead % = ", death))
sns.barplot(x="Sex", y="Survived", data=titanic_train)

sns.plt.title('Survival by Gender')
sns.barplot(x="Pclass", y="Survived", data=titanic_train)

sns.plt.title('Survival by Passenger Class')
titanic_train["Age"].hist(bins=20)

titanic_train[titanic_train['Survived']==1]["Age"].hist(bins=20)

sns.plt.xlabel('Age')

sns.plt.ylabel('Number of persons')

sns.plt.title('Distribution Survival by Age')
plt.scatter(titanic_train[titanic_train['Survived']==1]['PassengerId'],

            titanic_train[titanic_train['Survived']==1]['Fare'], c = 'b')

plt.scatter(titanic_train[titanic_train['Survived']==0]['PassengerId'],

            titanic_train[titanic_train['Survived']==0]['Fare'], c = 'r')



plt.xlabel('PassengerId')

plt.ylabel('Fare')

plt.xlim([0,900])

plt.ylim([0,550])

plt.legend(('Survived','Dead'),loc='upper left',fontsize=15,bbox_to_anchor=(1.05, 1))
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

axes[0].scatter(titanic_train[(titanic_train['Survived']==0)&(titanic_train['Pclass']==3)]['Age'],

            titanic_train[(titanic_train['Survived']==0)&(titanic_train['Pclass']==3)]['Fare'], c = 'r')

axes[0].scatter(titanic_train[(titanic_train['Survived']==0)&(titanic_train['Pclass']==2)]['Age'],

            titanic_train[(titanic_train['Survived']==0)&(titanic_train['Pclass']==2)]['Fare'], c = 'g')

axes[0].scatter(titanic_train[(titanic_train['Survived']==0)&(titanic_train['Pclass']==1)]['Age'],

            titanic_train[(titanic_train['Survived']==0)&(titanic_train['Pclass']==1)]['Fare'], c = 'b')

axes[0].set_xlabel('Age')

axes[0].set_ylabel('Fare')

axes[0].set_xlim([0,80])

axes[0].set_ylim([0,300])



axes[1].scatter(titanic_train[(titanic_train['Survived']==1)&(titanic_train['Pclass']==3)]['Age'],

            titanic_train[(titanic_train['Survived']==1)&(titanic_train['Pclass']==3)]['Fare'], c = 'r')

axes[1].scatter(titanic_train[(titanic_train['Survived']==1)&(titanic_train['Pclass']==2)]['Age'],

            titanic_train[(titanic_train['Survived']==1)&(titanic_train['Pclass']==2)]['Fare'], c = 'g')

axes[1].scatter(titanic_train[(titanic_train['Survived']==1)&(titanic_train['Pclass']==1)]['Age'],

            titanic_train[(titanic_train['Survived']==1)&(titanic_train['Pclass']==1)]['Fare'], c = 'b')

axes[1].set_xlabel('Age')

axes[1].set_ylabel('Fare')

axes[1].set_xlim([0,80])

axes[1].set_ylim([0,300])

axes[1].legend(('Pclass 3','Pclass 2', 'Pclass 1'),fontsize=15,bbox_to_anchor=(1.05, 1))
sns.barplot(x="Embarked", y="Survived", data=titanic_train)

sns.plt.title('Survival by Embarked')
#Creation of Dummy features for Sex, Embarker, Pclass



def dummy_features(df):

    new_embarked = pandas.get_dummies(df['Embarked'],prefix='Embarked')

    new_sex = pandas.get_dummies(df['Sex'])

    new_Pclass = pandas.get_dummies(df['Pclass'],prefix='Pclass')

    new_df = pandas.concat([new_embarked,new_sex,new_Pclass],axis=1)

    return new_df
train = pandas.concat([titanic_train,dummy_features(titanic_train)],axis=1)
train = train[train['Fare'] < 450]
#train.columns



#All the columns

#col = ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',

#       'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked', 'Embarked_C',

#       'Embarked_Q', 'Embarked_S', 'female', 'male', 'Pclass_1', 'Pclass_2',

#       'Pclass_3']

#

#New columns



col = ['PassengerId', 'Survived', 'Age', 'SibSp', 'Parch', 'Fare',

       'Embarked_C', 'Embarked_Q', 'Embarked_S', 

       'female', 'male', 

       'Pclass_1', 'Pclass_2', 'Pclass_3']
corr = train[train['Age']==train['Age']][col].corr()

corr.style.format("{:.2f}")
# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn import tree



from sklearn.metrics import accuracy_score
cls = []

cls.append(LogisticRegression()) 

cls.append(SVC())

cls.append(RandomForestClassifier())

cls.append(KNeighborsClassifier())

cls.append(GaussianNB())

cls.append(tree.DecisionTreeClassifier())

col = ['Fare',

       'Embarked_C', 'Embarked_Q', 'Embarked_S', 

       'female', 'male', 

       'Pclass_1', 'Pclass_2', 'Pclass_3']
limit = int(0.8 * train.shape[0])

X_train = train[col].iloc[:limit,:]

X_test = train[col].iloc[limit:,:]

y_test = train['Survived'].iloc[limit:]

y_train = train['Survived'].iloc[:limit]
for cl in cls:

    cl.fit(X_train,y_train)

    print(cl)

    print("score: ", accuracy_score(y_test,cl.predict(X_test)))
test = pandas.concat([titanic_test,dummy_features(titanic_test)],axis=1)
#Refit the classifier with all the train set



cls[2].fit(train[col],train['Survived'])
test[test['Fare']!=test['Fare']]
#I looking for a person with the same characteristic in train data set

train[(train['Pclass']==3)&(train['Embarked']=='S')&(train['male']==1.0)&(train['Age']>60)]
test['Fare'] = test['Fare'].fillna(6.24)
prediction = cls[2].predict(test[col])
test[test['Fare']>400]
#I check that she is alive

prediction[343]
submission = pandas.DataFrame({'PassengerId': test['PassengerId'],

                               'Survived': prediction})
submission.to_csv("submission_0.csv",index=False)