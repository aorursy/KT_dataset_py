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

import warnings

warnings.filterwarnings('ignore')

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
Train_data=pd.read_csv("/kaggle/input/mobile-price-range-prediction-is2020/train_data.csv")

Test_data=pd.read_csv("/kaggle/input/mobile-price-range-prediction-is2020/test_data.csv")

Submission_data=pd.read_csv("/kaggle/input/mobile-price-range-prediction-is2020/sample_submission.csv")
x_train=Train_data.drop(columns=['price_range','id'])

y_train=Train_data['price_range']
print(x_train,y_train,sep="\n")
x_test=Test_data.drop(columns=['id'])

print(x_test)
y_train.value_counts()
from sklearn.preprocessing import StandardScaler as ss

x_trainscale=ss().fit_transform(x_train)

x_testscale=ss().fit_transform(x_test)
pd.DataFrame(x_trainscale).head()
pd.DataFrame(x_testscale).head()
from sklearn.linear_model import LogisticRegression as lr

from sklearn.ensemble import RandomForestClassifier as rfc

from sklearn.model_selection import cross_val_score as cvs

ranfor=rfc().fit(x_trainscale,y_train)

y_prediction=ranfor.predict(x_testscale)

value=cvs(rfc(),x_trainscale,y_train,cv=3)

print(value)
print(value.mean())
res=pd.DataFrame({'id':Test_data['id'],'price_range':y_prediction})

res.to_csv('/kaggle/working/result_rf.csv',index=False)
from sklearn.svm import SVC

svc=SVC(kernel='linear',C=1)

y_prediction_svc=svc.fit(x_trainscale,y_train).predict(x_testscale)

value=cvs(svc,x_trainscale,y_train,cv=3)

print(value)

print(value.mean())

res1=pd.DataFrame({'id':Test_data['id'],'price_range':y_prediction_svc})

res1.to_csv('/kaggle/working/result_dtc.csv',index=False)
from sklearn.model_selection import GridSearchCV

gsc={'C':np.logspace(-3,3,7),'penalty':['l1','l2']}

value=GridSearchCV(lr(),gsc).fit(x_trainscale,y_train)

print(value.best_params_)

print(value.best_score_)
reg=lr(C=1000,penalty='l2')

reg.fit(x_trainscale,y_train)

y_pred=reg.predict(x_testscale)

value=cvs(lr(),x_trainscale,y_train,cv=3)

print(value)
res2=pd.DataFrame({'id':Test_data['id'],'price_range':y_pred})

res2.to_csv('/kaggle/working/result_lr.csv',index=False)