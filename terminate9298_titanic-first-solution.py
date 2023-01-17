%matplotlib inline



#data processing



import numpy as np

import pandas as pd



#data visualization



import matplotlib.pyplot as plt

import seaborn as sns



#data predications



from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
database = "../input/train.csv"

titanic = pd.read_csv(database)
titanic.columns
titanic.head()
#for getting data of surviving people

sns.countplot(x = 'Survived' ,data = titanic)
sns.countplot(x = 'Survived' , hue = 'Sex' , data = titanic)
sns.countplot( x = 'Survived' , hue = 'Pclass' , data= titanic)
sns.distplot(titanic['Age'].dropna(),bins = 50)
titanic['Fare'].hist(bins = 50)
titanic['Age'].hist(bins=50)
#Cleaning the Data

titanic.columns
titanic.tail()
#sns.boxplot(x='SibSp',y='Age',data=titanic,palette='winter')

sns.boxplot(x = 'Pclass' , y = 'Age' , data = titanic)
#sns.lmplot(x='Age', y='Parch', data=titanic)

#i am doing this according to answer but i will trying again

def defineage(cols):

    age= cols[0]

    pclass = cols[1]

    if pd.isnull(age):

        if pclass==1:

            return 37

        elif pclass==2:

            return 29

        elif pclass==3:

            return 24

    else:

        return age
def definecatage(cols):

    age = cols[0]

    if age < 5:

        return 0

    elif age < 10:

        return 1

    elif age < 15:

        return 2

    elif age < 20:

        return 3

    elif age < 25:

        return 4

    elif age < 30:

        return 5

    elif age < 35:

        return 6

    elif age < 40:

        return 7

    elif age < 45:

        return 8

    elif age < 50:

        return 9

    elif age < 55:

        return 10

    elif age < 60:

        return 11

    elif age < 65:

        return 12

    elif age < 70:

        return 13

    elif age < 75:

        return 14

    elif age < 80:

        return 15

    else:

        return 16
titanic['Age'] = titanic[['Age' , 'Pclass']].apply(defineage , axis = 1)
titanic.info()
def defcabin(cols):

    cabin = cols[0]

    if type(cabin)==str:

        return 1

    else:

        return 0

titanic['Cabin'] = titanic[['Cabin']].apply(defcabin , axis = 1)
titanic['Embarked'].unique()
titanic_new = titanic.dropna(axis=0)
titanic_new.info()
sex = pd.get_dummies(titanic_new['Sex'],drop_first = True)

embarked = pd.get_dummies(titanic_new['Embarked'], drop_first = True)

titanic_new.drop(['Sex','Embarked', 'Ticket', 'Name'],axis =1,inplace= True)
titanic_new = pd.concat([titanic_new,sex,embarked] ,axis =1)
titanic_new['Age'] = titanic_new[['Age']].apply(definecatage , axis = 1)
titanic_new.columns
t_train, t_test ,s_train ,s_test = train_test_split(titanic_new.drop('Survived',axis=1),titanic_new['Survived'] , test_size = 0.20 , random_state=101)
model = LogisticRegression()

model.fit(t_train , s_train)

#model.fit(titanic_new.drop('Survived',axis=1),titanic_new['Survived'])
pred = model.predict(t_test)
print(accuracy_score(s_test, pred))
#Trying tree Model 

from sklearn import tree

generalized_tree = tree.DecisionTreeClassifier(random_state =70 , max_depth =7 , min_samples_split = 4)

#generalized_tree = tree.DecisionTreeClassifier()

#generalized_tree.fit(t_train , s_train )

generalized_tree.fit(titanic_new.drop('Survived',axis=1),titanic_new['Survived'])
pred_tree = generalized_tree.predict(t_test)

print(accuracy_score(s_test, pred_tree))
rfc = RandomForestClassifier(n_estimators = 50 , bootstrap=False, min_samples_leaf = 4 )

rfc.fit(titanic_new.drop('Survived',axis=1),titanic_new['Survived'])

#rfc.fit(t_train,s_train)

#round(rfc.score(t_test , s_test)*100 , 2)
test_database = "../input/test.csv"

test_titanic = pd.read_csv(test_database)
#test_titanic_new.isna().sum()

test_titanic.describe()
test_titanic['Age'] = test_titanic[['Age' , 'Pclass']].apply(defineage , axis = 1)

test_titanic['Cabin'] = test_titanic[['Cabin']].apply(defcabin , axis = 1)

test_titanic.describe()
#test_titanic_new["Fare"] = test_titanic["Fare"].fillna((test_titanic["Fare"].median()),inplace= True)

#test_titanic_new = test_titanic.dropna(axis=0)

test_titanic['Fare'] = test_titanic['Fare'].fillna(test_titanic['Fare'].median())

test_titanic.describe()
sex = pd.get_dummies(test_titanic['Sex'],drop_first = True)

embarked = pd.get_dummies(test_titanic['Embarked'], drop_first = True)

test_titanic.drop(['Sex','Embarked', 'Ticket', 'Name'],axis =1,inplace= True)

test_titanic = pd.concat([test_titanic,sex,embarked] ,axis =1)

test_titanic['Age'] = test_titanic[['Age']].apply(definecatage , axis = 1)
test_titanic.describe()
answer = model.predict(test_titanic)
answer = rfc.predict(test_titanic)
#print(accuracy_score(answer, test_titanic_new['Survived']))
answer_tree = generalized_tree.predict(test_titanic)

#print(accuracy_score(answer_tree, test_titanic_new['Survived']))
output = pd.DataFrame({'PassengerId': test_titanic['PassengerId'],

                        'Survived': answer})
output.to_csv('submission.csv', index=False)
output = pd.DataFrame({'PassengerId': test_titanic['PassengerId'],

                        'Survived': answer_tree})

output.to_csv('submission_tree.csv', index=False)
output = pd.DataFrame({'PassengerId': test_titanic['PassengerId'],

                        'Survived': answer})

output.to_csv('submission_randomforest.csv', index=False)