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
import matplotlib.pyplot as plt

import seaborn as sns

#import dependancies

%matplotlib inline

sns.set_style('whitegrid')

#ignore warnings

import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

#view the test data

train.head()
#view the test data

test.head()
train.describe()
train.info()
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')


sns.heatmap(test.isnull(), yticklabels=False, cbar=False, cmap='viridis')
train.corr()
#by using Pclass we can remove NaN value of age column
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):

        if Pclass == 1:

            return 37   #near to mean of Age

        elif Pclass == 2:

            return 29   #near to mean of Age

        else:

            return 24   #near to mean of Age

    else:

        return Age
#impute the new function with old column

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
#impte the new function with old column

test['Age'] = test[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
train.drop('Cabin',axis=1,inplace=True)#drop cabin from train
test.drop('Cabin',axis=1,inplace=True)#drop cabin from test column
train.dropna(inplace=True)#drop any NaN value from train
test.dropna(inplace=True) #drop any NaN value from test
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
sns.heatmap(test.isnull(),yticklabels=False,cbar=False,cmap='viridis')
#get descriptive statistics on "object "datatype

train.describe(include=['object'])
#get descriptive statistick on "number" datatype

train.describe(include=['number'])
train.Survived.value_counts(normalize=True)
fig,axes = plt.subplots(2,4,figsize=(16,10))

sns.countplot('Survived',data=train,ax=axes[0,0])

sns.countplot('Pclass',data=train,ax=axes[0,1])

sns.countplot('Sex',data=train,ax=axes[0,2])

sns.countplot('SibSp',data=train,ax=axes[0,3])

sns.countplot('Parch',data=train,ax=axes[1,0])

sns.countplot('Embarked',data=train,ax=axes[1,1])

sns.distplot(train['Fare'],kde=True,ax=axes[1,2])

sns.distplot(train['Age'],kde=True,ax=axes[1,3])
sns.jointplot(x="Age",y="Fare",data=train)
sex = pd.get_dummies(train['Sex'],drop_first=True)

embark =pd.get_dummies(train['Embarked'],drop_first=True)
#concat dummies to train

train = pd.concat([train,sex,embark],axis=1)
train.head()
#same process applying for test 
sex = pd.get_dummies(test['Sex'],drop_first=True)

embark = pd.get_dummies(test['Embarked'],drop_first=True)
test = pd.concat([test,sex,embark],axis=1)
#drop column that dont need for model

train.drop(['PassengerId','Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

#same process for test 

test.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

train.head(2)
test.head()
#x = train.drop(['Survived'],axis=1)

x = train[['Pclass','Age','SibSp','Parch','Fare','male','Q','S']]  #predictors

y = train['Survived'] #target
from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.22, random_state = 42)

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score

logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)
print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
logmodel_pred = logmodel.predict(x_test)
acc_logmodel = round(accuracy_score(logmodel_pred, y_test) * 100, 2)

print(acc_logmodel)

from sklearn.svm import SVC



svc = SVC()

svc.fit(x_train, y_train)

svc_pred = svc.predict(x_test)

acc_svc = round(accuracy_score(svc_pred, y_test) * 100, 2)

print(acc_svc)
from sklearn.svm import LinearSVC



linear_svc = LinearSVC()

linear_svc.fit(x_train,y_train)

linear_svc_pred = linear_svc.predict(x_test)

acc_linear_svc = round(accuracy_score(linear_svc_pred,y_test) * 100,2)

print(acc_linear_svc)
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score





gaussian = GaussianNB()

gaussian.fit(x_train,y_train)

gaussian_pred = gaussian.predict(x_test)

acc_gaussian = round(accuracy_score(gaussian_pred,y_test) * 100,2)

print(acc_gaussian)
from sklearn.linear_model import Perceptron



perceptron = Perceptron()

perceptron.fit(x_train,y_train)

perception_pred = perceptron.predict(x_test)

acc_perceptron = round(accuracy_score(perception_pred,y_test) * 100,2)

print(acc_perceptron)
from sklearn.tree import DecisionTreeClassifier



decisiontree = DecisionTreeClassifier()

decisiontree.fit(x_train,y_train)

decisiontree_pred = decisiontree.predict(x_test)

acc_decisiontree = round(accuracy_score(decisiontree_pred,y_test) * 100,2)

print(acc_decisiontree)
from sklearn.ensemble import RandomForestClassifier



randomforest = RandomForestClassifier()

randomforest.fit(x_train,y_train)

randomforest_pred = randomforest.predict(x_test)

acc_randomforest = round(accuracy_score(randomforest_pred,y_test) *100,2)

print(acc_randomforest)

from sklearn.neighbors import KNeighborsClassifier



knn = KNeighborsClassifier()

knn.fit(x_train,y_train)

knn_pred = knn.predict(x_test)

acc_knn = round(accuracy_score(knn_pred,y_test) *100,2)

print(acc_knn)
from sklearn.linear_model import SGDClassifier



sgd = SGDClassifier()

sgd.fit(x_train,y_train)

y_pred = sgd.predict(x_test)

acc_sgd = round(accuracy_score(y_pred,y_test) * 100,2)

print(acc_sgd)


from sklearn.ensemble import GradientBoostingClassifier

gbk = GradientBoostingClassifier()

gbk.fit(x_train,y_train)

y_pred = gbk.predict(x_test)

acc_gbk = round(accuracy_score(y_pred,y_test) *100,2)

print(acc_gbk)

models = pd.DataFrame({'Model' :['Support Vector Machin','KNN','Logistic Regression','Random Forest','Naive Bayes','Perceptron','Linear SVC','Decision Tree','Stochastic Gradient Descent','Gradient Boosting Classifier'],

                       'Score' :[acc_svc,acc_knn,acc_logmodel,acc_randomforest,acc_gaussian,acc_perceptron,acc_linear_svc,acc_decisiontree,acc_sgd,acc_gbk]})

models.sort_values(by = 'Score',ascending=True)
#set ids as PassengerId and predict survival 

ids = test['PassengerId']

predictions = gbk.predict(test.drop('PassengerId', axis=1))



#set the output as a dataframe and convert to csv file named submission.csv

output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })

output.to_csv('submission.csv', index=False)

output