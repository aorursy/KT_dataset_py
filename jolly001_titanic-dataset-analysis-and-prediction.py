# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_set = pd.read_csv("/kaggle/input/titanic/train.csv")
test_set = pd.read_csv("/kaggle/input/titanic/test.csv")
submission_set = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
train_set.head()
train_set.columns
train_set.describe()
train_set.info()
train_set.isna().sum()
import matplotlib.pyplot as plt
import seaborn as sns
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(train_set.corr(), vmax=.8, square=True, cmap="BuPu")
sns.countplot(y = train_set.Survived)
sns.distplot(train_set.Age.dropna(),kde = False)
sns.countplot(x = train_set.SibSp, hue = train_set.Sex)
sns.countplot(x = train_set.SibSp, hue = train_set.Survived)
sns.countplot(x = train_set.SibSp)
sns.scatterplot(data = train_set.Fare)
sns.distplot(train_set.Fare, kde = False)
print(pd.crosstab(index = train_set.Survived , columns = train_set.Pclass))

pd.crosstab(index = train_set.Survived , columns = train_set.Pclass).plot(kind = 'bar')

print(pd.crosstab(index = train_set.Survived , columns = train_set.Embarked))

pd.crosstab(index = train_set.Survived , columns = train_set.Embarked).plot(kind = 'bar')
print(pd.crosstab(index = train_set.Survived , columns = train_set.Sex))
    
pd.crosstab(index = train_set.Survived , columns = train_set.Sex).plot(kind = 'bar')
bins = [0,1,2,3]
label = ['High Class','Medium Class', 'Low Class']
train_set.Pclass_bins = pd.cut(train_set.Pclass,bins,labels = label)

train_set.Pclass_bins.value_counts().plot(kind = 'bar')
bins = [0,25,60,90]
label = ['young','not young','very old']
train_set.Age_bins = pd.cut(train_set.Age,bins,labels = label)

train_set.Age_bins.value_counts().plot(kind = 'bar')

sns.boxplot(x = train_set.Pclass, y = train_set.Age)
def age_value(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
        
        elif Pclass == 2:
            return 29
        
        else:
            return 24
    else:
        return Age
    
train_set.Age = train_set[['Age','Pclass']].apply(age_value,axis = 1)

train_set.Embarked.fillna(train_set.Embarked.mode()[0], inplace=True)
train_set.isnull().sum()
sex = pd.get_dummies(train_set.Sex,drop_first = True)
embark = pd.get_dummies(train_set.Embarked,drop_first = True)
train_set.drop(['PassengerId','Cabin','Name','Ticket','Sex','Embarked'],axis = 1,inplace = True)
train = pd.concat([train_set,sex,embark], axis = 1)
train_survived = train.Survived
train.drop(['Survived'],axis = 1,inplace = True)
from sklearn.preprocessing import StandardScaler
train_x = StandardScaler()
train = train_x.fit_transform(train)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train,train_survived,test_size = 0.3)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
#Prediction with the help of Logistic Regression
classifier = LogisticRegression() 
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

#Prediction with the help of Decision Tree
tree_classifier = DecisionTreeClassifier(criterion = 'entropy')
tree_classifier.fit(X_train,y_train)
y_pred1 = tree_classifier.predict(X_test)

#Prediction with the help of SVC
svc_classifier = SVC(kernel = 'poly')
svc_classifier.fit(X_train,y_train)
y_pred2 = svc_classifier.predict(X_test)

#Prediction with the help of Random Forest Regression
ensemble_classifier = RandomForestClassifier(n_estimators = 20)
ensemble_classifier.fit(X_train,y_train)
y_pred3 = ensemble_classifier.predict(X_test)

#Prediction with the help of Naive Bayes
bayes_classifier = GaussianNB()
bayes_classifier.fit(X_train,y_train)
y_pred4 = bayes_classifier.predict(X_test)


print('The accuracy for Loisgtic Regression is :', accuracy_score(y_test,y_pred) , '\n' ,
     'The accuracy for Decision Tree is :',accuracy_score(y_test,y_pred1) , '\n',
     'The accuracy for SVC is :',accuracy_score(y_test,y_pred2) , '\n',
     'The accuracy for Random Forest Regression is :',accuracy_score(y_test,y_pred3) , '\n',
     'The accuracy for Naive Bayes is :', accuracy_score(y_test,y_pred4))
passengerID = test_set['PassengerId']
test_set.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_set.isnull().sum()
test_set.Fare.fillna(test_set.Fare.mode()[0], inplace=True)
test_set.Age = test_set[['Age','Pclass']].apply(age_value,axis = 1)

sex = pd.get_dummies(test_set.Sex,drop_first = True)
embark = pd.get_dummies(test_set.Embarked,drop_first = True)

test = pd.concat([test_set,sex,embark], axis = 1)

test.drop(['Sex','Embarked'],axis = 1,inplace = True)

test = train_x.fit_transform(test)
survived = svc_classifier.predict(test)

df_svc_submission = pd.DataFrame(data = [passengerID,survived]).transpose()
df_svc_submission.rename(columns = {'Unnamed 0':'Survived'}, inplace = True) 
df_svc_submission.to_csv('df_svc_submission.csv',index = False)