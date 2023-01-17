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
trainData=pd.read_csv("/kaggle/input/mobile-price-range-prediction-is2020-v2/train_data.csv")
trainData
testData=pd.read_csv("/kaggle/input/mobile-price-range-prediction-is2020-v2/test_data.csv")
testData
submissionData=pd.read_csv("/kaggle/input/mobile-price-range-prediction-is2020-v2/sample_submission.csv")
submissionData
x_train=trainData.drop(columns=['price_range','id'])
y_train=trainData['price_range']
x_test=testData.drop(columns=['id'])
print(x_train,y_train,x_test,sep="\n")

from sklearn.preprocessing import StandardScaler as ss
x_trainscale =ss().fit_transform(x_train)
x_testscale =ss().fit_transform(x_test)
from sklearn.linear_model import LogisticRegression as lr
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.model_selection import cross_val_score as cvs
r = rfc().fit(x_trainscale,y_train)
y_pred=r.predict(x_testscale)
v=cvs(rfc(),x_trainscale,y_train,cv=3)
res=pd.DataFrame({'id':testData['id'],'price_range':y_pred})
res.to_csv('/kaggle/working/result_dtc.csv',index=False)
reg=lr(C=1000,penalty='l2')
reg.fit(x_trainscale,y_train)
y_pred=reg.predict(x_testscale)
value=cvs(lr(),x_trainscale,y_train,cv=3)
print(value)