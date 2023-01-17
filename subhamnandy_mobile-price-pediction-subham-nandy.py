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
import warnings
warnings.filterwarnings("ignore")
train=pd.read_csv("/kaggle/input/mobile-price-range-prediction-is2020-v2/train_data.csv")
test=pd.read_csv("/kaggle/input/mobile-price-range-prediction-is2020-v2/test_data.csv")
sample_submission=pd.read_csv("/kaggle/input/mobile-price-range-prediction-is2020-v2/sample_submission.csv")
train.head()
test.head()
print(train.shape,test.shape)
train.info()
test.info()
train.isnull().sum()
test.isnull().sum()
train_clean =train.drop(columns=["id"])
test_clean=test.drop(columns=["id"])
train_clean
test_clean
x_train=train_clean.drop(columns=["price_range"])
y_train=train_clean["price_range"]
print(y_train.shape)
print(type(y_train))
import sklearn
from sklearn.preprocessing import StandardScaler
x_train_std=StandardScaler().fit_transform(x_train)
pd.DataFrame(x_train_std).head()
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression().fit(x_train_std,y_train)
y_pred=lr.predict(test_clean)
from sklearn.metrics import classification_report
print(classification_report(y_pred,sample_submission["price_range"]))
from sklearn.model_selection import cross_val_score
scores=cross_val_score(LogisticRegression(C=1),x_train_std,y_train,cv=5)
print(scores)
print(scores.mean())
from sklearn.model_selection import GridSearchCV
grid={"C":[0.6,0.7,0.8,0.9,1],"penalty":["l1","l2"]}
grid
score=GridSearchCV(LogisticRegression(solver='liblinear'),grid).fit(x_train_std,y_train)
print(score.best_params_)
print(score.best_score_)
test_clean_std= StandardScaler().fit_transform(test_clean)
lr=LogisticRegression(solver="liblinear",penalty='l1',C=0.9).fit(x_train_std,y_train)
y_pred=lr.predict(test_clean_std)
lr.predict_proba(test_clean_std)
final_output = test.assign(price_range = y_pred)[['id','price_range']]
final_output
result=pd.DataFrame(final_output)
result.to_csv("/kaggle/working/result.lr.csv",index=False)
from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier(criterion='gini',n_estimators=25,random_state=0)
RF=RF.fit(x_train_std,y_train)
y_pred_ref=RF.predict(test_clean)
from sklearn.metrics import classification_report
scores=cross_val_score(RandomForestClassifier(),x_train,y_train,cv=5)
print(scores)
print(scores.mean())
model=RandomForestClassifier().fit(x_train,y_train)
y_pred_RF=model.predict(test_clean)
final_output_RF = test.assign(price_range = y_pred_RF)[['id','price_range']]
final_output_RF
result1=pd.DataFrame(final_output_RF)
result1.to_csv("/kaggle/working/result.rf.csv",index=False)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
dtc=DecisionTreeClassifier(criterion="gini",max_depth=4,random_state=0)
scores=cross_val_score(dtc,x_train,y_train,cv=5)
print(scores)
print(scores.mean())
model1=dtc.fit(x_train,y_train)
y_pred_dtc=model1.predict(test_clean)
final_output_dtc = test.assign(price_range = y_pred_dtc)[['id','price_range']]
final_output_dtc
result2=pd.DataFrame(final_output_dtc)
result2.to_csv("/kaggle/working/result.dtc.csv",index=False)
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
svc=SVC(kernel='linear',C=1,random_state=0)
scores=cross_val_score(svc,x_train_std,y_train,cv=5)
print(scores)
print(scores.mean())
model2=svc.fit(x_train_std,y_train)
y_pred_svm=model2.predict(test_clean_std)
final_output_svm = test.assign(price_range = y_pred_svm)[['id','price_range']]
final_output_svm
result3=pd.DataFrame(final_output_svm)
result3.to_csv("/kaggle/working/result.svm.csv",index=False)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
knn = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
knn=knn.fit(x_train_std,y_train)
scores=cross_val_score(knn,x_train_std,y_train,cv=5)
print(scores)
print(scores.mean())
model3=knn.fit(x_train_std,y_train)
y_pred_knn=model3.predict(test_clean_std)
final_output_knn = test.assign(price_range = y_pred_knn)[['id','price_range']]
final_output_knn
result4=pd.DataFrame(final_output_knn)
result4.to_csv("/kaggle/working/result.knn.csv",index=False)
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
nb=GaussianNB()
nb=nb.fit(x_train_std,y_train)
scores=cross_val_score(svc,x_train_std,y_train,cv=5)
print(scores)
print(scores.mean())
model4=nb.fit(x_train_std,y_train)
y_pred_nb=model4.predict(test_clean_std)
final_output_nb = test.assign(price_range = y_pred_nb)[['id','price_range']]
final_output_nb
result5=pd.DataFrame(final_output_nb)
result5.to_csv("/kaggle/working/result.nb.csv",index=False)