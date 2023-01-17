# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



train=pd.read_csv("../input/train.csv")
test=pd.read_csv("../input/test.csv")
train.head()
test.head()
#----------Survived/Died by Pclass-------------------

survived_class = train[train['Survived']==1]['Pclass'].value_counts()

dead_class = train[train['Survived']==0]['Pclass'].value_counts()

s_class = pd.DataFrame([survived_class,dead_class])

s_class.index =['Survived','Dead']

s_class.plot(kind='bar',stacked=True,title='Survived/Died by class')



from IPython.display import display

display(s_class)



class1 = s_class.iloc[0,0]/s_class.iloc[:,0].sum()*100

class2 = s_class.iloc[0,1]/s_class.iloc[:,1].sum()*100

class3 = s_class.iloc[0,2]/s_class.iloc[:,2].sum()*100

print('% of class 1 survived',class1)

print('% of class 2 survived',class2)

print('% of class 3 survived',class3)
#----------Survived/Died by Sex-------------------

survived_sex = train[train['Survived']==1]['Sex'].value_counts()

died_sex = train[train['Survived']==0]['Sex'].value_counts()

ss_class = pd.DataFrame([survived_sex,died_sex])

ss_class.index = ['survived_sex','died_sex']

ss_class.plot(kind='bar',stacked=True,title = 'Survived/Died by Sex')



from IPython.display import display

display(ss_class)



female_survived = ss_class.iloc[0,0]/ss_class.iloc[:,0].sum()*100

male_survived = ss_class.iloc[0,1]/ss_class.iloc[:,1].sum()*100

print('% of female survived:',round(female_survived))

print('% of male survived:',round(male_survived))
X = train.drop(['PassengerId','Cabin','Ticket','Fare', 'Parch', 'SibSp'], axis=1)

y = X.Survived

X=X.drop(['Survived'],axis =1)



X.head()
from sklearn.preprocessing import LabelEncoder



LabelEncoder_X = LabelEncoder()

X.Sex = LabelEncoder_X.fit_transform(X.Sex)



print(sum(X.Embarked.isnull()))



row_index = X.Embarked.isnull()

X.loc[row_index,'Embarked'] = 'S'



Embarked = pd.get_dummies(X.Embarked,prefix = 'Embarked')

X = X.drop(['Embarked'],axis = 1)

X = pd.concat([X,Embarked],axis = 1)

X = X.drop(['Embarked_S'],axis = 1)



X.head()
#X["Age"].fillna(X["Age"].mean(), inplace=True)

X.Age.fillna(X.Age.mean(), inplace=True)

print(sum(X.Age.isnull()))
X = X.drop(['Name'],axis = 1)

#---------------Logistic Regression------------------------

from sklearn.linear_model import LogisticRegression



classifier = LogisticRegression(penalty='l2',random_state = 0)



from sklearn.model_selection import cross_val_score



accuracies = cross_val_score(classifier,X = X,y=y,cv = 10)

print('Logistic Regression accuracy : ',accuracies.mean())

#---------------K-Nearest Neighbors------------------------

from sklearn.neighbors import KNeighborsClassifier



classifier = KNeighborsClassifier(n_neighbors = 9)



from sklearn.model_selection import cross_val_score



accuracies = cross_val_score(classifier,X = X,y=y,cv = 10)

print('K-NN accuracy : ',accuracies.mean())
#---------------SVM------------------------

from sklearn.svm import SVC



classifier = SVC(kernel = 'rbf',random_state=0)



from sklearn.model_selection import cross_val_score



accuracies = cross_val_score(classifier,X = X,y=y,cv = 10)

print('SVM accuracy : ',accuracies.mean())
#---------------Naive Bayes------------------------

from sklearn.naive_bayes import GaussianNB



classifier = GaussianNB()



from sklearn.model_selection import cross_val_score



accuracies = cross_val_score(classifier,X = X,y=y,cv = 10)

print('Naive Bayes accuracy : ',accuracies.mean())
#---------------Random Forest Classifier------------------------

from sklearn.ensemble import RandomForestClassifier



model = RandomForestClassifier(n_estimators = 100,criterion = 'entropy',random_state = 0)



from sklearn.model_selection import cross_val_score



accuracies = cross_val_score(model,X = X,y=y,cv = 10)

print('Random Forest accuracy : ',accuracies.mean())



from sklearn.model_selection import train_test_split 



Xtrain, Xvalidation, Ytrain, Yvalidation = train_test_split(X, y, test_size=0.2, random_state=True)

model.fit(Xtrain,Ytrain)
Ypredict = model.predict(Xvalidation)



from sklearn.metrics import accuracy_score

accuracy_score(Yvalidation, Ypredict)
test=pd.read_csv("../input/test.csv")

test.head()

test_final = test.drop(['Name','PassengerId','Cabin','Ticket','Fare', 'Parch', 'SibSp'],axis = 1)





from sklearn.preprocessing import LabelEncoder

LabelEncoder_X = LabelEncoder()

test_final.Sex = LabelEncoder_X.fit_transform(test_final.Sex)

test_final.head()

print(sum(test.Embarked.isnull()))



#row_index = X.Embarked.isnull()

#X.loc[row_index,'Embarked'] = 'S'



Embarked = pd.get_dummies(test_final.Embarked,prefix = 'Embarked')

test_final = test_final.drop(['Embarked'],axis = 1)

test_final = pd.concat([test_final,Embarked],axis = 1)

test_final = test_final.drop(['Embarked_S'],axis = 1)



test_final.head()


print(sum(test_final.Embarked_C.isnull()))

print(sum(test_final.Embarked_Q.isnull()))
submission = pd.DataFrame()

test1=pd.read_csv("../input/test.csv")

test_final.Age.fillna(test_final.Age.mean(), inplace=True)

submission['PassengerId'] = test1['PassengerId']



submission['Survived'] = model.predict(test_final)



submission.to_csv('submission.csv',index = False)



submission.head()