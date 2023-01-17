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
df_train=pd.read_csv('/kaggle/input/credit-risk/original.csv')
df_train.head()
df_train.size
df_train.columns
df_train.describe()
df_train.info()
df_train.count()
df_train['default']
df_train.isnull().sum()
df_train['age']
df_train.isnull().sum()
df_train["age"]=df_train["age"].fillna(df_train['age'].median())
df_train.isnull().sum()
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
df_train_len=len(df_train)
test=df_train[:df_train_len]

test.drop(labels=["default"],axis=1,inplace=True)

test.head()
train=df_train[:df_train_len]

X_train=train.drop(labels=["default"],axis=1)

y_train=train["default"]

X_train,X_test ,y_train,y_test=train_test_split(X_train,y_train,test_size=0.33,random_state=42)

print("X_train",len(X_train))

print("X_test",len(X_test))

print("y_train",len(y_train))

print("y_test",len(y_test))

print("test",len(test))
logreg=LogisticRegression()

logreg.fit(X_train,y_train)

acc_log_train=round(logreg.score(X_train,y_train)*100,2)

acc_log_test=round(logreg.score(X_test,y_test)*100,2)

print("Training Accuray:% {}".format(acc_log_train))

print("Testing Accuray:% {}".format(acc_log_test))


random_state=42

classifier=[DecisionTreeClassifier(random_state=random_state),

           SVC(random_state=random_state),

           RandomForestClassifier(random_state=random_state),

           LogisticRegression(random_state=random_state),

           KNeighborsClassifier()]

dt_param_grid={"min_samples_split":range(10,500,20),

              "max_depth":range(1,20,2)}

svc_param_grid={"kernel":["rbf"],

               "gamma":[0.001,0.01,0.1,1],

               "C":[1,10,50,100,200,300,1000]}

rf_param_grid={"max_features":[1,3,10],

              "min_samples_split":[2,3,10],

              "min_samples_leaf":[1,3,10],

              "bootstrap":[False],

              "n_estimators":[100,300],

              "criterion":["gini"]}

logreg_param_grid={"C":np.logspace(-3,3,7),

                  "penalty":["l1","l2"]}

knn_param_grid={"n_neighbors":np.linspace(1,19,10,dtype=int).tolist(),

               "weights":["uniform","distance"],

               "metric":["euclidean","manhattan"]}

classifier_param=[dt_param_grid,

                 svc_param_grid,

                 rf_param_grid,

                 logreg_param_grid,

                 knn_param_grid]

cv_result=[]

best_estimators=[]

for i in range(len(classifier)):

    clf=GridSearchCV(classifier[i],param_grid=classifier_param[i],cv=StratifiedKFold(n_splits=10),scoring="accuracy",n_jobs=-1,verbose=1)

    clf.fit(X_train,y_train)

    cv_result.append(clf.best_score_)

    best_estimators.append(clf.best_estimator_)

    print(cv_result[i])
import seaborn as sns

cv_results=pd.DataFrame({"Cross Validation Means":cv_result,"ML Models":["DecisionTreeClassifier","SVM","RandomForestClassifier","LogisticRegression","KNeighborsClassifier"]})

g=sns.barplot("Cross Validation Means","ML Models",data=cv_results)

g.set_xlabel("Mean Accuracy")

g.set_title("Cross Validation Score")
votingC=VotingClassifier(estimators=[("dt",best_estimators[0]),

                                     ("rfc",best_estimators[2]),

                                     ("lr",best_estimators[3])],

                                      voting="soft",n_jobs=-1)

votingC=votingC.fit(X_train,y_train)

print('Highest Precison:',100*accuracy_score(votingC.predict(X_test),y_test))