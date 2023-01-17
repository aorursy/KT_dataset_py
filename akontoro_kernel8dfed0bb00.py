# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.import pandas as pd
from sklearn import preprocessing
import numpy as np
from xgboost import XGBClassifier



model = XGBClassifier(model = XGBClassifier(objective='binary:logistic', n_estimators=100, seed=123, eval_metric = "auc", max_delta_step=1, max_depth = 3,    min_child_weight = 5, subsample=0.55, colsample_bytree=0.85))


pd_train = pd.read_csv('../input/saftey_efficay_myopiaTrain.csv')
pd_test = pd.read_csv('../input/saftey_efficay_myopiaTest.csv')
pd_train = pd_train[['T_L_Treatment_ZO','T_L_Max_Abl._Depth','T_L_Laser_Type', 'T_L_Treatment_Type', 'T_L_Temp',
'T_L_Op.Time', 'T_L_Therapeutic_Cont._L.', 'Pre_L_Est_UCVA_', 'Pre_L_Est_BCVA', 'Class']]
pd_test = pd_test[['T_L_Treatment_ZO','T_L_Max_Abl._Depth','T_L_Laser_Type', 'T_L_Treatment_Type', 'T_L_Temp',
'T_L_Op.Time', 'T_L_Therapeutic_Cont._L.', 'Pre_L_Est_UCVA_', 'Pre_L_Est_BCVA']]
le = preprocessing.LabelEncoder()
X, y = pd_train.iloc[:,:-1], pd_train.iloc[:,-1]
X_train = X.dropna(how='all')
X_train = X_train.dropna(how='all')
y = y.dropna(how='all')
pd_test = pd_test.fillna(lambda x: x.mean())
# Train our classifier
for column in X_train.columns:
    if X_train[column].dtype == type(object):
        X_train[column] = X_train[column].factorize()[0]
        le = preprocessing.LabelEncoder()
        X_train[column] = le.fit_transform(X_train[column])

model = model.fit(X_train, y)

# Train our classifier
for column in pd_test.columns:
    if pd_test[column].dtype == type(object):
        pd_test[column] = pd_test[column].factorize()[0]
        le = preprocessing.LabelEncoder()
        pd_test[column] = le.fit_transform(pd_test[column])
# Make predictions
preds = model.predict_proba(pd_test)
submission = pd.DataFrame(preds)
submission = submission[1]
submission = submission.reset_index()
submission['Id'] = range(1, len(submission) + 1)
submission = pd.DataFrame(submission).rename(columns={1: 'Class'})
submission = submission[['Id', 'Class']]
pd.DataFrame(submission).to_csv( "./submission.csv", index=False)








