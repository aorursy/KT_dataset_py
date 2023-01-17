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
warnings.filterwarnings('ignore')
train=pd.read_csv("/kaggle/input/mobile-price-range-prediction-is2020-v2/train_data.csv")
test=pd.read_csv("/kaggle/input/mobile-price-range-prediction-is2020-v2/test_data.csv")
submissiond=pd.read_csv("/kaggle/input/mobile-price-range-prediction-is2020-v2/sample_submission.csv")
train.info()
test.info()
x_train=train.drop(columns=['price_range','id'])
x_test=test.drop(columns=['id'])
y_train=train['price_range']
print(" x train:{} y train{}".format(x_train,y_train))


y_train.value_counts()


print(x_test)
import sklearn 
from sklearn.preprocessing import StandardScaler 
x_trains=StandardScaler().fit_transform(x_train)
x_tests=StandardScaler().fit_transform(x_test)

print("X trains {}".format(x_trains))
print("X tests {}".format(x_tests))

pd.DataFrame(x_trains).head()

pd.DataFrame(x_tests).head()
from sklearn.linear_model import LogisticRegression as l
#from sklearn.ensemble import RandomForestClassifier as l
LR=l().fit(x_trains,y_train)
y_pred=LR.predict(x_tests)

from sklearn.model_selection import cross_val_score as c
#from sklearn.metrics import classification_report
val=c(l(),x_trains,y_train,cv=3,scoring='accuracy')
print(val)
print(val.mean())
result=pd.DataFrame({'id':test['id'],'price_range':y_pred})
result.to_csv('/kaggle/working/result_rf4.csv',index=False)