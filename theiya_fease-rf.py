from pandas import Series,DataFrame
import pandas as pd

# numpy, matplotlib, seaborn
import numpy as np

names=[
'Field6',
'Field7',
'Field8',
'Field10',
'CoverageField2A',
'CoverageField2B',
'CoverageField3A',
'CoverageField4B',
'CoverageField5A',
'CoverageField5B',
'CoverageField6A',
'CoverageField6B',
'CoverageField8',
'CoverageField11A',
'CoverageField11B',
'SalesField1A',
'SalesField1B',
'SalesField2A',
'SalesField2B',
'SalesField3',
'SalesField4',
'SalesField6',
'SalesField7',
'SalesField8',
'SalesField9',
'SalesField10',
'SalesField12',
'SalesField13',
'SalesField14',
'SalesField15',
'PersonalField1',
'PersonalField2',
'PersonalField4B',
'PersonalField5',
'PersonalField6',
'PersonalField7',
'PersonalField8',
'PersonalField9',
'PersonalField10A',
'PersonalField10B',
'PersonalField11',
'PersonalField12',
'PersonalField13',
'PersonalField15',
'PersonalField16',
'PersonalField17',
'PersonalField18',
'PersonalField19',
'PersonalField22',
'PersonalField23',
'PersonalField25',
'PersonalField27',
'PersonalField29',
'PersonalField33',
'PersonalField34',
'PersonalField36',
'PersonalField37',
'PersonalField38',
'PersonalField39',
'PersonalField40',
'PersonalField41',
'PersonalField42',
'PersonalField47',
'PersonalField48',
'PersonalField49',
'PersonalField50',
'PersonalField51',
'PersonalField52',
'PersonalField53',
'PersonalField56',
'PersonalField57',
'PersonalField59',
'PersonalField60',
'PersonalField62',
'PersonalField63',
'PersonalField64',
'PersonalField66',
'PersonalField69',
'PersonalField70',
'PersonalField71',
'PersonalField74',
'PersonalField75',
'PersonalField77',
'PersonalField81',
'PersonalField82',
'PersonalField83',
'PersonalField84',
'PropertyField1A',
'PropertyField1B',
'PropertyField2A',
'PropertyField2B',
'PropertyField3',
'PropertyField4',
'PropertyField6',
'PropertyField7',
'PropertyField8',
'PropertyField9',
'PropertyField10',
'PropertyField11B',
'PropertyField12',
'PropertyField13',
'PropertyField14',
'PropertyField15',
'PropertyField16B',
'PropertyField18',
'PropertyField19',
'PropertyField20',
'PropertyField21B',
'PropertyField22',
'PropertyField23',
'PropertyField24B',
'PropertyField25',
'PropertyField26A',
'PropertyField26B',
'PropertyField27',
'PropertyField28',
'PropertyField29',
'PropertyField30',
'PropertyField31',
'PropertyField32',
'PropertyField33',
'PropertyField34',
'PropertyField35',
'PropertyField36',
'PropertyField37',
'PropertyField38',
'PropertyField39A',
'GeographicField1A',
'GeographicField2B',
'GeographicField4A',
'GeographicField4B',
'GeographicField5A',
'GeographicField6A',
'GeographicField8A',
'GeographicField11A',
'GeographicField13B',
'GeographicField15A',
'GeographicField16B',
'GeographicField17A',
'GeographicField17B',
'GeographicField18A',
'GeographicField20B',
'GeographicField21B',
'GeographicField22A',
'GeographicField22B',
'GeographicField23A',
'GeographicField23B',
'GeographicField24A',
'GeographicField26A',
'GeographicField27A',
'GeographicField29B',
'GeographicField30B',
'GeographicField32A',
'GeographicField33B',
'GeographicField36B',
'GeographicField37B',
'GeographicField38A',
'GeographicField39B',
'GeographicField41A',
'GeographicField41B',
'GeographicField42B',
'GeographicField43A',
'GeographicField44A',
'GeographicField45A',
'GeographicField45B',
'GeographicField46B',
'GeographicField48A',
'GeographicField48B',
'GeographicField50B',
'GeographicField52B',
'GeographicField53B',
'GeographicField54B',
'GeographicField55B',
'GeographicField56A',
'GeographicField59A',
'GeographicField59B',
'GeographicField60A',
'GeographicField60B',
'GeographicField61A',
'GeographicField61B',
'GeographicField62A',
'GeographicField62B',
'GeographicField63',
'Year',
'Month'
]


import random
from datetime import datetime
import pandas as pd
from pandas import DataFrame as df
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import train_test_split,cross_val_score
from sklearn import preprocessing

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train_sample = np.random.choice(train.index.values,130000)   
train = train.ix[train_sample]

# Converting date into datetime format
train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))
# Dropping original date column
train = train.drop('Original_Quote_Date', axis=1)   

test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))
test = test.drop('Original_Quote_Date', axis=1)

## Seperating date into 3 columns
train['Year'] = train['Date'].apply(lambda x: int(str(x)[:4]))
train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))
train['weekday'] = train['Date'].dt.dayofweek

test['Year'] = test['Date'].apply(lambda x: int(str(x)[:4]))
test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))
test['weekday'] = test['Date'].dt.dayofweek 

train = train.drop('Date', axis=1)
test = test.drop('Date', axis=1)    

## Filing NA values with -1

train = train.fillna(-1)
test = test.fillna(-1)
test_ori=test

y = train.QuoteConversion_Flag.values

#columns choice--gmm
train=DataFrame(train,columns=names)
test=DataFrame(test,columns=names)

for f in train.columns:
    if train[f].dtype=='object':
        print(f)
        lbl=preprocessing.LabelEncoder()
        lbl.fit(list(train[f].values)+list(test[f].values))
        train[f]=lbl.transform(list(train[f].values))
        test[f]=lbl.transform(list(test[f].values))

import xgboost as xgb

X_train = train
Y_train =y
X_test  = test

params = {"objective": "binary:logistic"}
T_train_xgb = xgb.DMatrix(X_train, Y_train)
X_test_xgb  = xgb.DMatrix(X_test)
gbm = xgb.train(params, T_train_xgb, 20)
Y_pred = gbm.predict(X_test_xgb)
# Create submission
submission = pd.DataFrame()
submission["QuoteNumber"]          = test_ori["QuoteNumber"]
submission["QuoteConversion_Flag"] = Y_pred
submission.to_csv('homesite.csv', index=False)
      























