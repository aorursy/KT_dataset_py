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
test=pd.read_csv("/kaggle/input/janatahack-crosssell-prediction/test.csv")
train=pd.read_csv("/kaggle/input/janatahack-crosssell-prediction/train.csv")
print(train.shape)
print(test.shape)

data = pd.concat([train, test], axis=0)
data
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
data.Gender=encoder.fit_transform(data.Gender)
data.Vehicle_Damage=encoder.fit_transform(data.Vehicle_Damage)
data.head()
vehicle_age={"< 1 Year": 0,'1-2 Year':1,'> 2 Years':2}
data["Vehicle_Age"]=data["Vehicle_Age"].replace(vehicle_age)
data["Vehicle_Age"].unique()
data=data.drop(columns=["Age","Vehicle_Damage","id"],axis=1,inplace=False)
Train=data.iloc[:381109,:]
Test=data.iloc[381109: ,:]
Train["Response"]=train["Response"]
train["Response"].value_counts()
y=Train.Response
Train = Train.drop(columns='Response')
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(Train,y,test_size=0.2,random_state=1)
print(y_train.value_counts())
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

over = RandomOverSampler(sampling_strategy=0.4)
under = RandomUnderSampler(sampling_strategy=0.6)

x_train, y_train = over.fit_resample(x_train, y_train)
print(y_train.value_counts())
x_train, y_train = under.fit_resample(x_train, y_train)
print(y_train.value_counts())
Mod_train=data.iloc[:381109,:]
Mod_test=data.iloc[381109:,:]
Mod_train
Mod_train["Response"].value_counts()
Mod_train["Response"]=train["Response"]
from sklearn.utils import resample
min_data=Mod_train[Mod_train["Response"]==1]
maj_data=Mod_train[Mod_train["Response"]==0]
mod_min_data=resample(min_data,n_samples=334399,replace=True)
mod_data=pd.concat([maj_data,mod_min_data])
mod_data.shape
Y=mod_data.Response
X=mod_data.drop("Response",axis=1)

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=1)
import seaborn as sns
sns.countplot(y_train)
from sklearn.preprocessing import MinMaxScaler
scale = MinMaxScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)

from lightgbm import LGBMClassifier
model_1 = LGBMClassifier(
    boosting_type='gbdt',
    objective='binary',
 learning_rate =0.01,
 n_estimators=5000,
 colsample_bytree=0.3)
#model.fit(X_train, y_train)
model_1.fit(x_train, y_train, eval_metric='auc', 
          eval_set=[(x_test, y_test)], early_stopping_rounds=500, verbose=100)
lgbm_model = LGBMClassifier( colsample_bytree=0.3,
               learning_rate=0.01, n_estimators=2990, objective='binary')

lgbm_model.fit(x_train, y_train)
pred = lgbm_model.predict(x_test)
from sklearn.metrics import classification_report
print(classification_report(pred, y_test))

from sklearn.metrics import f1_score, roc_auc_score,accuracy_score,confusion_matrix, precision_recall_curve, auc, roc_curve, recall_score, classification_report 
import matplotlib.pyplot as plt

y_score = lgbm_model.predict_proba(x_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_score)
print ('Area under curve (AUC): ', auc(fpr,tpr))
from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(x_train,y_train)
Y_pred=model.predict(x_test)
print(classification_report(Y_pred, y_test))



y_score = model.predict_proba(x_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_score)
print ('Area under curve (AUC): ', auc(fpr,tpr))
# from sklearn.model_selection import cross_val_score
# score=cross_val_score(lgbm_model,X,Y,cv=5,scoring="roc_auc")
# print(score.mean())
Mod_test = Mod_test.drop(columns='Response')
predictions=lgbm_model.predict(Mod_test)
result=pd.DataFrame(test["id"],columns=["id","Response"])
result["Response"]=predictions
result.to_csv("sub.csv",index=0)