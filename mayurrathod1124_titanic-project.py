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
import warnings
warnings.filterwarnings('ignore')
train=pd.read_csv("/kaggle/input/titanic/train.csv")
test=pd.read_csv("/kaggle/input/titanic/test.csv")
display(train.head()),display(test.head())
train.isnull().sum(),test.isnull().sum()

train.drop(["Cabin"],axis=1,inplace=True)
test.drop(["Cabin"],axis=1,inplace=True)
train.columns
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
def is_alone(x):
    if  (x['SibSp'] + x['Parch'])  > 0:
        return 1
    else:
        return 0

train['Is_alone'] = train.apply(is_alone, axis = 1)
test['Is_alone'] = test.apply(is_alone, axis = 1)

g = sns.catplot(x="Is_alone", col = 'Survived', data=train, kind = 'count', palette='deep')
g = sns.FacetGrid(train, col='Survived')
g = g.map(sns.distplot, "Age")
train.head()
train.corr()
train.drop(['PassengerId','Name','SibSp','Parch','Ticket'],axis=1,inplace=True)
test.drop(['Name','SibSp','Parch','Ticket'],axis=1,inplace=True)
print(train.columns),print(test.columns)
train['Survived'].value_counts()
feature=train.drop("Survived",axis=1)
target=train['Survived']
target.shape
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(feature,target, test_size = 0.3, random_state = 23)
print(x_train.shape),print(y_train.shape)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
import numpy as np
#from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

numerical_features = ['Age','Fare','Is_alone'] 
categorical_features = ['Pclass','Sex','Embarked']
preprocessor=make_column_transformer(
             (make_pipeline(
              SimpleImputer(strategy='median'),
              KBinsDiscretizer(n_bins=3)), numerical_features),
             
              (make_pipeline(
               SimpleImputer(strategy='constant',fill_value='missing'),
              OneHotEncoder(categories='auto',handle_unknown='ignore')), categorical_features)
)
from sklearn.linear_model import LogisticRegression
log_model=make_pipeline(preprocessor,LogisticRegression())
log_model.fit(x_train,y_train)
log_pred=log_model.predict(x_test)
log_pred
from sklearn import metrics
from sklearn.metrics import accuracy_score
acc_log=accuracy_score(log_pred,y_test)
acc_log
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict

result_log = cross_val_score(log_model,feature, target, cv=10, scoring="accuracy" )
print("Cross val Score ", result_log.mean())
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

rf_model=make_pipeline(preprocessor,RandomForestClassifier(n_estimators=500))
rf_model.fit(x_train,y_train)
rf_pred=rf_model.predict(x_test)
print(rf_pred)
acc_rf=accuracy_score(rf_pred,y_test)
print("Accuracy= ",acc_rf)
result_rf = cross_val_score(rf_model,feature, target, cv=10, scoring="accuracy" )
print("Cross val Score ", result_rf.mean())
svc_model=make_pipeline(preprocessor,SVC(kernel='rbf',C=1))
svc_model.fit(x_train,y_train)
svc_pred=svc_model.predict(x_test)
print(svc_pred)
acc_svc=accuracy_score(svc_pred,y_test)
print("Accuracy",acc_svc)
result_svc = cross_val_score(svc_model,feature, target, cv=10, scoring="accuracy" )
print("Cross val Score ", result_svc.mean())
from sklearn.neighbors import KNeighborsClassifier
knn_model=make_pipeline(preprocessor,KNeighborsClassifier(n_neighbors=8))
knn_model.fit(x_train,y_train)
knn_pred=knn_model.predict(x_test)
print(knn_pred)
acc_knn=accuracy_score(knn_pred,y_test)
print("Accuracy",acc_knn)
result_knn = cross_val_score(knn_model,feature, target, cv=10, scoring="accuracy" )
print("Cross val Score ", result_knn.mean())
models=pd.DataFrame({ 'model':['Logistic','Random forest','SVM','KNN'],
                    'Accuracy':[acc_log,acc_rf,acc_svc,acc_knn],
                    'Cross_Val_Score':[result_log.mean(),result_rf.mean(),result_svc.mean(),result_knn.mean()]})
models
models.sort_values(by='Accuracy',ascending=True)
models.sort_values(by='Cross_Val_Score',ascending=True)
test.head()
x_test=test.drop('PassengerId',axis=1)
x_test.head()
sub_data=pd.read_csv("/kaggle/input/titanic/gender_submission.csv")
sub_data.head()
y_test=sub_data.iloc[:,1]
y_test
pred_sub=svc_model.predict(x_test)
print(pred_sub)
acc_sub=accuracy_score(y_test, pred_sub)
print('Accuracy= ', acc_sub)

df = pd.DataFrame()
df['PassengerId'] = test['PassengerId']
df['Survived'] = pred_sub
df.to_csv('/kaggle/working/titanic_final2.csv', index=False)
pd.set_option('display.max_rows', 500)
results = pd.read_csv('/kaggle/working/titanic_final2.csv')
#results
results