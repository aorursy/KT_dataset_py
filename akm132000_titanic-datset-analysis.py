import numpy as np

import pandas as pd

import os



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
submission_format = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
submission_format.head()
train_data_orig = pd.read_csv("/kaggle/input/titanic/train.csv")

train_data=train_data_orig.copy()

train_data.head()
test_data_orig = pd.read_csv("/kaggle/input/titanic/test.csv")

test_data=test_data_orig.copy()

test_data.head()
distinct_classes,value_counts=np.unique(train_data.Survived,return_counts=True)
train_data.shape
train_data.isnull().sum()
test_data.isnull().sum()
train_data.drop('Cabin',axis=1,inplace=True)

test_data.drop('Cabin',axis=1,inplace=True)


train_data.isnull().sum()
test_data.isnull().sum()
train_data.head()
train_data.drop('PassengerId',axis=1,inplace=True)
testPIds=test_data['PassengerId']

test_data.drop('PassengerId',axis=1,inplace=True)
train_data.head()
test_data.head()
def change_gender_num(s):

    if (s=='male'):

        return 1

    else:

        return 0

    

train_data['Sex']=train_data['Sex'].apply(change_gender_num)

test_data['Sex']=test_data['Sex'].apply(change_gender_num)
train_data.head()
test_data.head()
train_data['relatives']=train_data['SibSp']+train_data['Parch']+1

train_data.drop(['SibSp','Parch'],axis=1,inplace=True)
test_data['relatives']=test_data['SibSp']+test_data['Parch']+1

test_data.drop(['SibSp','Parch'],axis=1,inplace=True)
train_data.head()
test_data.head()
len(train_data.Ticket.unique())
train_data.drop('Ticket',axis=1,inplace=True)
test_data.drop('Ticket',axis=1,inplace=True)
train_data.head()
test_data.head()
def get_title(name):

    return name.strip().split(',')[1].split('.')[0]

    

train_data['title']=train_data['Name'].apply(get_title)

train_data.drop('Name',axis=1,inplace=True)



test_data['title']=test_data['Name'].apply(get_title)

test_data.drop('Name',axis=1,inplace=True)
train_data.head()
test_data.head()
import matplotlib

import matplotlib.pyplot as plt

matplotlib.rcParams['figure.figsize']=(15,10)
(keys,values)=np.unique(train_data[train_data.Survived==1].title,return_counts=True)
keys
plt.bar(keys,values,align='center', alpha=0.5,color='r')

plt.xlabel('Title')

plt.ylabel('Number of people survived')

plt.title('Survived vs Title')

plt.show()
train_data.Age.fillna(train_data.Age.mean(),inplace=True)
test_data.Age.fillna(test_data.Age.mean(),inplace=True)
train_data.isnull().sum()
test_data.isnull().sum()
train_data.fillna(train_data.Embarked.value_counts().idxmax(),inplace=True)
test_data.fillna(test_data.Fare.mean(),inplace=True)
survived_train_data=train_data[train_data.Survived==1]
survived_train_data.Embarked.unique()
(keys,values)=np.unique(survived_train_data.Embarked,return_counts=True)
plt.bar(keys,values,align='center', alpha=0.5,color='g')

plt.xlabel('Port')

plt.ylabel('Number of people survived')

plt.title('Survived vs Port')

plt.show()
dummies=pd.get_dummies(train_data.Embarked)
dummies_test=pd.get_dummies(test_data.Embarked)
train_data2=pd.concat([train_data,dummies],axis='columns')
test_data2=pd.concat([test_data,dummies_test],axis='columns')
train_data2.drop('Embarked',axis=1,inplace=True)
test_data2.drop('Embarked',axis=1,inplace=True)
train_data2.shape
test_data2.shape
train_data2.head()
test_data2.shape
thresh_title_counts=10

title_names=(train_data2.title.value_counts() < thresh_title_counts)
train_data2['title']=train_data2.title.apply(lambda x:'Misc' if title_names.loc[x]==True else x)
distinct_keys=list(train_data2['title'].value_counts().keys())
distinct_keys.remove('Misc')
distinct_keys
test_data2['title']=test_data2.title.apply(lambda x:'Misc' if x not in distinct_keys else x)
test_data2['title'].value_counts()
train_data2.shape
test_data2.shape
dummies_embarked_train=pd.get_dummies(train_data2['title'])
dummies_embarked_test=pd.get_dummies(test_data2['title'])
train_data3=pd.concat([train_data2,dummies_embarked_train],axis='columns')
test_data3=pd.concat([test_data2,dummies_embarked_test],axis='columns')
train_data3.drop('title',axis=1,inplace=True)
test_data3.drop('title',axis=1,inplace=True)
train_data3.head()
test_data3.head()
train_data3.isnull().sum()
test_data3.isnull().sum()
yTrain=train_data3.Survived

train_data4=train_data3.drop('Survived',axis=1)

xTrain=train_data4.values
xTest=test_data3.values
print (xTrain.shape)

print (xTest.shape)
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix,classification_report
xTrain_small,xTest_small,yTrain_small,yTest_small=train_test_split(xTrain,yTrain)
xTrain_small.shape
from sklearn.svm import SVC

from sklearn.model_selection import GridSearchCV
clf_best_fit=SVC(kernel='linear')
clf=SVC(kernel='linear')
clf.fit(xTrain_small,yTrain_small)
clf.score(xTest_small,yTest_small)
yTest_predicted_small=clf.predict(xTest_small)
confusion_matrix(yTest_small,yTest_predicted_small)
print (classification_report(yTest_small,yTest_predicted_small))
yPredicted=clf.predict(xTest)
from sklearn.ensemble import RandomForestClassifier



clf = RandomForestClassifier(random_state=0)

clf.fit(xTrain_small,yTrain_small)
clf.score(xTest_small,yTest_small)
from sklearn.linear_model import LogisticRegression



clf_logistic_regression=LogisticRegression(max_iter=1000)

clf_logistic_regression.fit(xTrain_small,yTrain_small)
clf.score(xTest_small,yTest_small)
final_dataFrame=pd.DataFrame()

final_dataFrame['PassengerId']=testPIds
final_dataFrame.head()
from sklearn.neural_network import MLPClassifier
clf=MLPClassifier(hidden_layer_sizes=(20,10),max_iter=1000,activation='logistic')

clf.fit(xTrain_small,yTrain_small)
clf.score(xTest_small,yTest_small)
yPredicted_MLP_Small=clf.predict(xTest_small)
print (classification_report(yTest_small,yPredicted_MLP_Small))
print (confusion_matrix(yTest_small,yPredicted_MLP_Small))
yPredicted_MLP=clf.predict(xTest)
final_dataFrame['Survived']=yPredicted_MLP
final_dataFrame.to_csv('titanic.csv', index=False)

print("Your submission was successfully saved!")