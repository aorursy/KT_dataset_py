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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt 

import seaborn as sns

sns.set()

df=pd.read_csv("../input/titanic/train.csv")

test=pd.read_csv("../input/titanic/test.csv")
df.head()
df.info()
df.describe()
df.isnull().sum()
dfnew=df

dfnew['Age'].fillna(dfnew['Age'].mean(), inplace = True)

dfnew.isnull().sum()
dfnew4=dfnew.drop(['PassengerId','Survived','Name','Ticket','Cabin'],axis=1)

y=dfnew.Survived

print(dfnew4)
test.head()
test.info()
test.isnull().sum()
test1=test

testnew=test1['Age'].fillna(test1['Age'].mean(), inplace = True)

testnew=test1['Fare'].fillna(test1['Fare'].mean(), inplace = True)

test1.isnull().sum()
testfix=test1.drop(['PassengerId','Name','Ticket','Cabin'],axis=1)

testfix
print(dfnew4.Embarked.unique())

print(testfix.Embarked.unique())
train_objs_num = len(dfnew4)

dataset = pd.concat(objs=[dfnew4, testfix], axis=0)

dataset = pd.get_dummies(dataset,drop_first=True)

train1 = dataset[:train_objs_num]

test1 = dataset[train_objs_num:]

print(train1.columns)

print(test1.columns)
sns.countplot(x=dfnew.Survived,hue="Pclass",data=dfnew)

plt.title("no of passenger survived")
sns.countplot(x=dfnew.Survived,hue="Sex",data=dfnew)
a=dfnew[dfnew.Sex=='female']['Survived']

female_survive_percent=sum(a==1)*100/len(a)

female_survive_percent
b=dfnew[dfnew.Sex=='male']['Survived']

male_survive_percent=sum(b==1)*100/len(b)

male_survive_percent
a=dfnew[dfnew.Age<=18]

a

sns.countplot(x=a.Survived,hue="Sex",data=a)
dfnew.groupby('Pclass')['Age'].mean()
a=dfnew.corr()

sns.heatmap(a,annot=True,cmap="RdYlGn")
from sklearn.ensemble import ExtraTreesRegressor

model=ExtraTreesRegressor()

model.fit(train1,y)
model.feature_importances_

check=pd.Series(model.feature_importances_,index=train1.columns)

check.nlargest(6).plot(kind='barh')

plt.show()
from sklearn.model_selection import train_test_split



X_train, y_train, X_test ,y_test = train_test_split(train1,y,test_size=0.2,random_state=0)

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)

def models(X_train,X_test):

    from sklearn import preprocessing

    from sklearn import utils

    lab_enc = preprocessing.LabelEncoder()

    training_scores_encoded = lab_enc.fit_transform(X_test)

    print(training_scores_encoded)

    print(utils.multiclass.type_of_target(X_test))

    print(utils.multiclass.type_of_target(X_test.astype('int')))

    print(utils.multiclass.type_of_target(training_scores_encoded))

    from sklearn.linear_model import LogisticRegression

    log=LogisticRegression(random_state=0)

    log.fit(X_train,training_scores_encoded)

    

    from sklearn.neighbors import KNeighborsClassifier

    knn=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)

    knn.fit(X_train,training_scores_encoded)

    

    from sklearn.svm import SVC

    svc_lin=SVC(kernel='linear',random_state=0)

    svc_lin.fit(X_train,training_scores_encoded)

    

    from sklearn.svm import SVC

    svc_rbf=SVC(kernel='rbf',random_state=0)

    svc_rbf.fit(X_train,training_scores_encoded)

    

    from sklearn.naive_bayes import GaussianNB

    gauss=GaussianNB()

    gauss.fit(X_train,training_scores_encoded)

    

    from sklearn.tree import DecisionTreeClassifier

    tree=DecisionTreeClassifier(criterion='entropy',random_state=0)

    tree.fit(X_train,training_scores_encoded)

    

    from sklearn.ensemble import RandomForestClassifier

    forest=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)

    forest.fit(X_train,training_scores_encoded)

    

    print('[0] logistic regression training accuracy:  ', log.score(X_train,training_scores_encoded))

    print('[1] k regression training accuracy:  ', knn.score(X_train,training_scores_encoded))

    print('[2] svc linear regression training accuracy:  ', svc_lin.score(X_train,training_scores_encoded))

    print('[3] svc rbf regression training accuracy:  ', svc_rbf.score(X_train,training_scores_encoded))

    print('[4] gaussian regression training accuracy:  ', gauss.score(X_train,training_scores_encoded))

    print('[5] decision regression training accuracy:  ', tree.score(X_train,training_scores_encoded))

    print('[6] randomforest regression training accuracy:  ', forest.score(X_train,training_scores_encoded))

    

    

    return log,knn,svc_lin,svc_rbf,gauss,tree,forest





model=models(X_train,X_test)
import numpy as np

from sklearn.metrics import confusion_matrix



for i in range(len(model)):

    cm=confusion_matrix(y_test,model[i].predict(y_train))

    TN,FP,FN,TP=confusion_matrix(y_test,model[i].predict(y_train)).ravel()

    test_score=(TP+TN)/(TP+TN+FN+FP)

    print(cm)

    print('Model[{}]  testing accuracy="{}"' .format(i,test_score))

    

    

    print()

    
precision=TP/(TP+FP)

precision
recall=TP/(TP+FN)

recall
F1_score=2*precision*recall/(precision+recall)

F1_score
from sklearn.model_selection import GridSearchCV





from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()

parameters = {

                  'n_estimators' : [400,500,600,700,800,900,1000],

                  'max_depth'    : [5,5.5,6,6.5,7,7.5,8],

                'criterion':['gini', 'entropy'],

                 'max_features':['auto', 'sqrt', 'log2']

                 }

grid = GridSearchCV(estimator=model, param_grid = parameters, cv = 2, n_jobs=-1,verbose=1)

grid.fit(train1,y)



# Results from Grid Search

print("\n========================================================")

print(" Results from Grid Search " )

print("========================================================")

print("\n The best estimator across ALL searched params:\n",

          grid.best_estimator_)

print("\n The best score across ALL searched params:\n",

          grid.best_score_)

print("\n The best parameters across ALL searched params:\n",

          grid.best_params_)

print("\n ========================================================")
forest=RandomForestClassifier(criterion= 'entropy', max_depth= 7, max_features= 'sqrt', n_estimators= 1000)

a=forest.fit(train1,y)

b=a.predict(test1)

b
output = pd.DataFrame({'PassengerId': test.PassengerId, 'Survived': b})

output.to_csv('submission1234.csv', index=False)