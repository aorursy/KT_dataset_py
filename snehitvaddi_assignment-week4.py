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
train=pd.read_csv("/kaggle/input/mobile-price-range-prediction-is2020/train_data.csv")

train.info()
test=pd.read_csv("/kaggle/input/mobile-price-range-prediction-is2020/test_data.csv")

test.info()
submission=pd.read_csv("/kaggle/input/mobile-price-range-prediction-is2020/sample_submission.csv")

print(submission.head(10))
train.head(10)

test.head(10)
train.columns
train['price_range']
y_train=train['price_range']

x_train=train.drop(columns=['price_range','id'])

x_train
y_train.value_counts()
x_test=test.drop(columns=['id'])

x_test.columns
# Normalizing data convergence. This brings all the values under

# one range without any distortion.

from sklearn.preprocessing import StandardScaler as ss

x_train=ss().fit_transform(x_train)

x_test=ss().fit_transform(x_test)
# Logistic Regression

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import cross_val_score as cvs

lr = LogisticRegression().fit(x_train, y_train)

loges_pred = lr.predict(x_test)

scores_logistic=cvs(LogisticRegression(),x_train,y_train,cv=3)

print(scores_logistic)

print(scores_logistic.mean())
# Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier as rfc

from sklearn.model_selection import cross_val_score as cvs

rand_for=rfc().fit(x_train,y_train)

rf_pred=rand_for.predict(x_test)

scores_randF=cvs(rfc(),x_train,y_train,cv=3)

print(scores_randF)

print(scores_randF.mean())
# Support Vector Machines

from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score as cvs

svm=SVC()

svm=svm.fit(x_train,y_train)

svm_pred=svm.predict(x_test)

scores_svm=cvs(SVC(),x_train,y_train,cv=3)

print(scores_svm)

print(scores_svm.mean())
# K-Nearest Neighbours

from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score as cvs

knn = KNeighborsClassifier()

knn.fit(x_train, y_train)

knn_pred = knn.predict(x_test)

scores_knn=cvs(knn,x_train,y_train,cv=3)

print(scores_knn)

print(scores_knn.mean())
res_logi=pd.DataFrame({'id':test['id'],'price_range':loges_pred})

res_logi.to_csv('/kaggle/working/result_logesticReg.csv',index=False)
res_RF=pd.DataFrame({'id':test['id'],'price_range':rf_pred})

res_RF.to_csv('/kaggle/working/result_randf.csv',index=False)
res_svm=pd.DataFrame({'id':test['id'],'price_range':svm_pred})

res_svm.to_csv('/kaggle/working/result_svm.csv',index=False)
res_knn=pd.DataFrame({'id':test['id'],'price_range':knn_pred})

res_knn.to_csv('/kaggle/working/result_knn.csv',index=False)
df = pd.read_csv('/kaggle/working/result_logesticReg.csv')

df
res1=pd.DataFrame({'id':Test_data['id'],'price_range':y_prediction_svc})

res1.to_csv('/kaggle/working/result_dtc.csv',index=False)
reg=lr(C=1000,penalty='l2')

reg.fit(x_trainscale,y_train)

y_pred=reg.predict(x_testscale)

value=cvs(lr(),x_trainscale,y_train,cv=3)

print(value)
res2=pd.DataFrame({'id':Test_data['id'],'price_range':y_pred})

res2.to_csv('/kaggle/working/result_lr.csv',index=False)
