import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#import numpy as np # linear algebra

#import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#rint(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')
train.head()
train.describe()
train.describe(include=['O'])
sns.countplot(x=train['Survived'],data=train,hue='Sex')
sns.countplot(x=train['Survived'],data=train,hue='Pclass')
sns.countplot(x=train['Survived'],data=train,hue='Embarked')
sns.distplot(train['Age'].dropna(), color='red',bins=100)
sns.countplot(x=train['SibSp'],data=train)
sns.countplot(x=train['Survived'],data=train,hue='SibSp')
sns.countplot(x=train['Parch'],data=train)
sns.countplot(x=train['Survived'],data=train,hue='Parch')
sns.boxplot(x='Pclass',y='Age',data=train)
train.groupby(['Pclass'])['Age'].mean()
def impute_age(cols):

    Age = cols[0]

    Pclass = cols[1]

    if pd.isnull(Age):

        if Pclass == 1:

            return 38

        elif Pclass == 2:

            return 30

        else:

            return 25

    else:

        return Age
train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)
print(train.isnull().sum())
train['Embarked'].fillna('S', inplace=True)
train.drop(['Cabin','Ticket','Name'],axis=1,inplace=True)
sex = pd.get_dummies(train['Sex'],drop_first=True)

embark = pd.get_dummies(train['Embarked'],drop_first=True)

train.drop(['Sex','Embarked'],axis=1,inplace=True)

train = pd.concat([train,sex,embark],axis=1)
train.head()
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 

                                                    train['Survived'], test_size=0.30, 

                                                    random_state=101)
class color:

   PURPLE = '\033[95m'

   CYAN = '\033[96m'

   DARKCYAN = '\033[36m'

   BLUE = '\033[94m'

   GREEN = '\033[92m'

   YELLOW = '\033[93m'

   RED = '\033[91m'

   BOLD = '\033[1m'

   UNDERLINE = '\033[4m'

   END = '\033[0m'
def modelResults(model):

    model.fit(X_train,y_train)

    y_pred = model.predict(X_test)

    modelStr=str(model)[:str(model).find("(")]

    print("\n")

    print (color.BOLD + color.RED + modelStr + color.END)

    print("\n")

    print (color.BOLD + color.UNDERLINE + "Classification Report" + color.END)

    print(classification_report(y_test,y_pred))

    print("\n")

    print (color.BOLD + color.UNDERLINE + "Confusion Matrix" + color.END)

    print(confusion_matrix(y_test, y_pred))

    print("\n")

    if modelStr=="LogisticRegression":

        print(color.BOLD + color.UNDERLINE + "Accuracy" + color.END)

        print(logmodel.score(X_test,y_test))
from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.svm import SVC

from sklearn.decomposition import PCA



NBclassifier = GaussianNB()

logmodel = LogisticRegression()

rfModel = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)

dtModel = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)

svmModel = SVC(kernel = 'linear', random_state = 0)

knnModel = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)



modelResults(NBclassifier)

modelResults(logmodel)

modelResults(rfModel)

modelResults(dtModel)

modelResults(svmModel)

modelResults(knnModel)
train_test = pd.read_csv('../input/test.csv')
train_test.groupby(['Pclass'])['Age'].mean()
def impute_age_test(cols):

    Age = cols[0]

    Pclass = cols[1]

    

    if pd.isnull(Age):



        if Pclass == 1:

            return 40



        elif Pclass == 2:

            return 29



        else:

            return 24



    else:

        return Age
train_test['Age'] = train_test[['Age','Pclass']].apply(impute_age_test,axis=1)
train_test.drop(['Cabin','Name','Ticket'],axis=1,inplace=True)

sex = pd.get_dummies(train_test['Sex'],drop_first=True)

embark = pd.get_dummies(train_test['Embarked'],drop_first=True)

train_test = pd.concat([train_test,sex,embark],axis=1)

train_test.drop(['Sex','Embarked'],axis=1,inplace=True)
train_test.isnull().sum()
train_test[train_test.Fare.isnull()]
train_test.groupby(['Pclass'])['Fare'].mean()
train_test['Fare'].fillna('12.45', inplace=True)
def finalResult(model):

    y_pred = model.predict(train_test)

    train_test_result = train_test.copy(deep=True)

    train_test_result['Survived'] = y_pred

    train_test_result.drop(['Pclass','Age','SibSp','Parch','Fare','male','Q','S'],axis=1,inplace=True)

    submissionFileName = str(model)[:str(model).find("(")]

    submissionFileName = 'titanic_submission_' + submissionFileName + '.csv'

    train_test_result.to_csv(submissionFileName,index=False)
finalResult(NBclassifier)

finalResult(logmodel)

finalResult(rfModel)

finalResult(dtModel)

finalResult(svmModel)

finalResult(knnModel)