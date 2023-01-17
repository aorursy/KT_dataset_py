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
filepath_test="../input/titanic/test.csv"

filepath_train="../input/titanic/train.csv"



test=pd.read_csv(filepath_test)

train=pd.read_csv(filepath_train)
train.head()
import matplotlib.pyplot as plt

import seaborn as sns



train.hist(figsize=(10,10),color="green",bins=20)
drop_features=['Cabin','Name','Ticket','Embarked','Fare']

train=train.drop(drop_features,axis=1)

train.head()
from sklearn.preprocessing import OneHotEncoder

# creating instance of one-hot-encoder

enc = OneHotEncoder(handle_unknown='error',drop='first')



# passing the gender column 

enc_df = pd.DataFrame(enc.fit_transform(train[['Sex']]).toarray())



# merge with main df bridge_df on key values

train =train.join(enc_df)
train.head()
train["Male"]=train[0]

train=train.drop(["Sex",0],axis=1)
train.head()
plt.figure(figsize=(7,7))

sns.countplot(x="Survived",hue="Male",data=train)

plt.title("0=Female , 1=Male")
plt.figure(figsize=(7,7))

sns.countplot(x="Survived",hue="Pclass",data=train)
train.fillna(value=train.Age.median(),inplace=True)
sns.heatmap(train.isnull())
train
train.set_index(['PassengerId'],inplace=True)



x=train[["Pclass","Age","SibSp","Parch","Male"]]

y=train["Survived"]



from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.25,train_size=0.75)
from sklearn.svm import SVC

svc=SVC()
from sklearn.model_selection import GridSearchCV

parameters=[{'C':[1, 2,3,5,10],'kernel':['linear']},

           {'C':[1, 2,3,5,10],'kernel':['rbf'],'gamma':[0.3, 0.4, 0.5, 0.6, 0.7]}]

grid_search=GridSearchCV(estimator=svc,

                        param_grid=parameters,

                        scoring='accuracy',

                        cv=10,

                        n_jobs= -1)

grid_search=grid_search.fit(X_train,Y_train)

accuracy=grid_search.best_score_

best_param=grid_search.best_params_



print(accuracy)

print(best_param)
svc_para= SVC(C=2, kernel='rbf',gamma=0.3)

svc_para.fit(X_train,Y_train)
from sklearn.metrics import confusion_matrix



pred=svc_para.predict(X_test)



svc_cm=confusion_matrix(Y_test,pred)

print("Confusion Matrix Is:\n",svc_cm)
svc_para.score(X_test,Y_test)
import xgboost as xgb

model=xgb.XGBClassifier()

params={'max_depth':[3,5,10,20,30,40,50,100],

        'learning_rate':[0.01,0.05,0.1,0.15,0.2],

        'n_estimators':[100,500,1000],

        'min_child_weight ':[1,2,3,4,5]

        }
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint



def hypertuning_fun(model,params,n,x,y):

    rdmsearch=RandomizedSearchCV(model,param_distributions=params,n_jobs=-1,n_iter=n,cv=10)

    rdmsearch.fit(x,y)

    ht_params=rdmsearch.best_params_

    ht_score=rdmsearch.best_score_

    return ht_params,ht_score



model_para,model_score=hypertuning_fun(model,params,40,x,y)
model_para
model=xgb.XGBClassifier(n_estimators= 100,

                    min_child_weight = 2,

                    max_depth= 10,

                    learning_rate= 0.05,

                    verbosity =3,

                    n_jobs=-1

                   )
model.fit(x,y)
x
test.head()
DFeatures=['Name','Ticket','Fare','Cabin','Embarked']



test=test.drop(DFeatures,axis=1)

enc_test=pd.DataFrame(enc.fit_transform(test[['Sex']]).toarray())

test=test.join(enc_test)
test
test['Male']=test[0]

test=test.drop([0,'Sex'],axis=1)
test
test.fillna(test.Age.median(),inplace=True)

test
sns.heatmap(test.isnull())
test.set_index(["PassengerId"],inplace=True)
test
test_predict=model.predict(test)
test_predict=pd.Series(test_predict)
test_predict
test.reset_index(inplace=True)
test
predict=test["PassengerId"]
predict
test_predict
predict=pd.concat([predict,test_predict],axis=1)
predict
predict.rename(columns={0:"Survived"},inplace=True)
predict
predict.to_csv("my_submission.csv",index=False)
sns.countplot(predict.Survived)