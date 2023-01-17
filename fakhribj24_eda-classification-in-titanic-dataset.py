import pandas as pd 

train = pd.read_csv('../input/titanic/train.csv')

test = pd.read_csv('../input/titanic/test.csv')

gender = pd.read_csv('../input/titanic/gender_submission.csv')
train.head()
gender.head()
test.head()
test = pd.merge(test,gender,on=['PassengerId'])

test.head()
train.info()

test.info()
for i in train['Age'] : 

    train['Age'].fillna(train['Age'].median(), inplace = True)

for j in test['Age'] : 

    test['Age'].fillna(test['Age'].median(), inplace = True)



for k in train['Embarked'] :

    train['Embarked'].fillna(train['Embarked'].mode()[0], inplace = True)

for l in test['Embarked'] :

    test['Embarked'].fillna(test['Embarked'].mode()[0], inplace = True)
train.info()

test.info()
#Changing the sex column with 0 and 1

def gen(Gen):

    if Gen == 'male':

        return 0

    elif Gen == 'female':

        return 1

train['Sex'] = train['Sex'].apply(gen)

test['Sex'] = test['Sex'].apply(gen)
train['Title'] = [i.split('.')[0] for i in train.Name.astype('str')]

train['Title'] = [i.split(',')[1] for i in train.Title.astype('str')]

train.head()
test['Title'] = [i.split('.')[0] for i in test.Name.astype('str')]

test['Title'] = [i.split(',')[1] for i in test.Title.astype('str')]

test.head()
test['Embarked'].unique()
import seaborn as sns 

import matplotlib.pyplot as plt

plt.subplots(figsize=(10,8))

sns.countplot(x="Embarked",data=train,hue = "Survived").set_title("Embarked in Titanic ")
train['Embarked'].value_counts()
train['Pclass'].unique()
train['Pclass'].value_counts()
plt.subplots(figsize=(10,8))

sns.countplot(x="Pclass",data=train,hue = "Survived").set_title("Pclass in Titanic ")
train['Title'].unique()
def Title(t):

    if t == ' the Countess' or t == ' Mlle' or t == ' Sir' or t == ' Ms' or t ==' Lady' or t ==' Mme':

        return "special"

    elif t == ' Mrs':

        return ' Mrs'

    elif t == ' Miss':

        return ' Miss'

    elif t == ' Master':

        return ' Master'

    elif t == ' Col':

        return ' Col'

    elif t == ' Major':

        return ' Major'

    elif t == ' Dr':

        return ' Dr'

    elif t == ' Mr':

        return ' Mr'

    else:

        return 'another'



train['Title'] = train['Title'].apply(Title)

test['Title'] = test['Title'].apply(Title)
train['Title'].value_counts()
plt.subplots(figsize=(10,8))

sns.countplot(x="Title",data=train).set_title("People in Titanic based on the title")
#graph distribution of quantitative data

plt.figure(figsize=[16,12])





plt.subplot(234)

plt.hist(x = [train[train['Survived']==1]['Fare'], train[train['Survived']==0]['Fare']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Fare Histogram by Survival')

plt.xlabel('Fare ($)')

plt.ylabel('# of Passengers')

plt.legend()



plt.subplot(235)

plt.hist(x = [train[train['Survived']==1]['Age'], train[train['Survived']==0]['Age']], 

         stacked=True, color = ['g','r'],label = ['Survived','Dead'])

plt.title('Age Histogram by Survival')

plt.xlabel('Age')

plt.ylabel('# of Passengers')

plt.legend()

drop_column = ['PassengerId']

train.drop(drop_column, axis=1, inplace = True)

test.drop(drop_column, axis=1, inplace = True)
train.groupby('Survived').mean()
train = pd.get_dummies(train, columns = ['Embarked'])

test = pd.get_dummies(test, columns = ['Embarked'])
drop_column = ['Cabin', 'Ticket']

train.drop(drop_column, axis=1, inplace = True)

test.drop(drop_column, axis=1, inplace = True)
train = pd.get_dummies(train, columns = ['Title'])

test = pd.get_dummies(test, columns = ['Title'])
train.head()
x = train.iloc[:,3:]  #delete target column from train dataset

y = train['Survived'] # test dataset  
from sklearn.model_selection import train_test_split

# divide dataset into 65% train, and other 35% test.

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=0)
train['Survived'].unique()
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import confusion_matrix 

classifier1 = KNeighborsClassifier(n_neighbors=2)

classifier1.fit(x_train, y_train)

#Predicting the Test set results 

y_pred = classifier1.predict(x_test)

#Making the confusion matrix 

cm = confusion_matrix(y_test,y_pred)





sns.heatmap(cm, annot=True, linewidth=5, cbar=None)

plt.title('KNN Classifier confusion matrix')

plt.ylabel('True label')

plt.xlabel('predicted label')
print('accuracy of train dataset is',classifier1.score(x_train, y_train))

print('accuracy of test dataset is',classifier1.score(x_test, y_test))
#classification report for the test set

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
from sklearn.svm import SVC 

classifier2 = SVC(kernel = 'rbf', random_state = 0)

classifier2.fit(x_train, y_train)

#Predicting the Test set results 

y_pred = classifier2.predict(x_test)

#Making the confusion matrix 

#from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(y_test,y_pred)







sns.heatmap(cm, annot=True, linewidth=5, cbar=None)

plt.title('SVM with rbf kernel Classifier confusion matrix')

plt.ylabel('True label')

plt.xlabel('predicted label')
print('accuracy of train dataset is',classifier3.score(x_train, y_train))

print('accuracy of test dataset is',classifier3.score(x_test, y_test))
#classification report for the test set

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
from sklearn.tree import DecisionTreeClassifier

classifier3 = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

classifier3.fit(x_train, y_train)

#Predicting the Test set results 

y_pred = classifier3.predict(x_test)

#Making the confusion matrix 

#from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(y_test,y_pred)



sns.heatmap(cm, annot=True, linewidth=5, cbar=None)

plt.title('Decision Tree Classifier confusion matrix')

plt.ylabel('True label')

plt.xlabel('predicted label')
print('accuracy of train dataset is',classifier3.score(x_train, y_train))

print('accuracy of test dataset is',classifier3.score(x_test, y_test))
#classification report for the test set

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
from sklearn.ensemble import RandomForestClassifier 



classifier4 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

classifier4.fit(x_train,y_train)

#Predicting the Test set results 

y_pred = classifier4.predict(x_test)

#Making the confusion matrix 

#from sklearn.metrics import confusion_matrix 

cm = confusion_matrix(y_test,y_pred)



sns.heatmap(cm, annot=True, linewidth=5, cbar=None)

plt.title('RF with with entropy impurity Classifier confusion matrix')

plt.ylabel('True label')

plt.xlabel('predicted label')
print('accuracy of train dataset is',classifier4.score(x_train, y_train))

print('accuracy of test dataset is',classifier4.score(x_test, y_test))