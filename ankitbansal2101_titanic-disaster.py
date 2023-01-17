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
df_train=pd.read_csv("../input/titanic/train.csv")
df_test=pd.read_csv("../input/titanic/test.csv")
df_train.head()
df_test.head()
df_train.describe()  # 
df_train.shape
df_test.shape
df_test.describe()
df_train.info()
df_test.info()
df_train.columns
df_train.Survived.value_counts()
# unnecessary columns

df_train=df_train.drop(["Ticket","Cabin"],axis=1)
df_test=df_test.drop(["Ticket","Cabin"],axis=1)
df_train.shape
df_test.shape
df_train.isnull().sum()
df_test.isnull().sum()
# there are lot of missing values in age column , we cant drop them, we have to fill those na with some appropriate values
df_train.corr()
# Replacing the null values in the Age column with Mean
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
# Fit and transform to the parameters
df_train['Age'] = imputer.fit_transform(df_train[['Age']])
df_test['Age'] = imputer.fit_transform(df_test[['Age']])

df_train.isnull().sum()
df_test.isnull().sum()
df_test.Fare.describe()
df_test['Fare'] = imputer.fit_transform(df_test[['Fare']])
df_test.isnull().sum()    # no null values remaining in test dataset
df_train.isnull().sum()
df_train.Embarked.value_counts()
# we can replace missing values of embarked with S
df_train['Embarked'] = df_train['Embarked'].fillna(df_train['Embarked'].value_counts().index[0])
df_train.isnull().sum()  # no missing values in train
df_train.shape
df_test.shape
df_train.head()
df_train=df_train.drop("Name",axis=1)
df_test=df_test.drop("Name",axis=1)
df_train.info()
df_test.info()
# we should convert pclass to categorical as it defines classes 1,2,3
df_train["Pclass"]=df_train["Pclass"].astype(str)
df_test["Pclass"]=df_test["Pclass"].astype(str)
df_train.info()
df_train=df_train.drop("PassengerId",axis=1)
submission_id=df_test["PassengerId"]
df_test=df_test.drop("PassengerId",axis=1)
df_train=pd.get_dummies(df_train)
df_test=pd.get_dummies(df_test)
df_train.head()
df_test.head()
df_train.shape
df_test.shape
submission_id
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,auc,roc_curve,confusion_matrix, classification_report,roc_auc_score
# splitting the data
y=df_train["Survived"]
X=df_train.drop("Survived",axis=1)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train.shape,y_train.shape,X_test.shape,y_test.shape
model_lr=LogisticRegression(max_iter=2000)
model_lr.fit(X_train,y_train)
y_pred_lr=model_lr.predict(X_test)
confusion_matrix(y_test,y_pred_lr)
acc_lr=143/179
acc_lr
# logistic regression accuracy is 79.88826
model_lr.score(X_test,y_test)
model_tree=DecisionTreeClassifier(max_depth=3,random_state=42)
model_tree.fit(X_train,y_train)
y_pred_tree=model_tree.predict(X_test)
confusion_matrix(y_pred_tree,y_test)
acc_tree=143/179
acc_tree
hyperparameters={"max_depth":np.arange(3,10)}
model=DecisionTreeClassifier()
model_tree_tune=GridSearchCV(model,hyperparameters,cv=5).fit(X_train,y_train)
model_tree_new=model_tree_tune.best_estimator_
y_pred_tree_new=model_tree_new.predict(X_test)
confusion_matrix(y_test,y_pred_tree_new)
model_tree_new.score(X_test,y_test)
from sklearn.neighbors import KNeighborsClassifier
for i in range(1,10):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    print(knn.score(X_test,y_test))
# k=5 have max accuracy but still less the logitic regression and decisiontreeclassifier
model_rf=RandomForestClassifier(n_estimators=500,oob_score=True,verbose=1).fit(X_train,y_train)
print(model_rf.score(X_test,y_test))
model_rf
# getting good accuracies then logisticregression and decisiontre3eclassifier
parameters = {'max_features':np.arange(1,10),'max_depth':np.arange(1,6)}
tune_model = GridSearchCV(model_rf,parameters,cv=5,scoring='accuracy').fit(X_train,y_train)
tune_model_rf=tune_model.best_estimator_
y_pred_rf_tuned=tune_model_rf.predict(X_test)
confusion_matrix(y_test,y_pred_rf_tuned)
accuracy_score(y_test,y_pred_rf_tuned)
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
model_xgb =  XGBClassifier(n_estimators=500, objective= 'binary:logistic',seed=80).fit(X_train,y_train)
y_pred_xgb=model_xgb.predict(X_test)
accuracy_score(y_test,y_pred_xgb)
y_pred_submit=model_xgb.predict(df_test)
submission_df1=pd.DataFrame({"PassengerId":pd.Series(submission_id),"Survived":pd.Series(y_pred_submit)})

submission_df1
submission_df1.to_csv("submission_titanic3.csv",index=False)