import numpy as np

import pandas as pd

import os

os.listdir('../input')
train = pd.read_csv('../input/train.csv',nrows=1000000)

train=train.drop( ['DefaultBrowsersIdentifier','Wdft_IsGamer','Census_InternalBatteryNumberOfCharges','Wdft_RegionIdentifier','Census_IsFlightingInternal','Census_IsWIMBootEnabled','Census_InternalBatteryType','Census_ThresholdOptIn','ProductName','MachineIdentifier'],axis=1)

#train=train.dropna()



pd.set_option('display.max_columns',None)

train.head()

y = train['HasDetections']

x = train.drop(['HasDetections'],axis=1)

x.dropna()



#print(x.head())

y.dropna()

y.head()
x.OrganizationIdentifier.dtype
x.fillna(0)

x.head()
x
from sklearn.preprocessing import LabelEncoder

#X.head()



from sklearn.preprocessing import LabelEncoder

for i in x:

    if(x[i].dtype==object):

        x[i] = LabelEncoder().fit_transform(x[i].astype(str))

x.isnull().sum()

x.fillna(method='ffill', inplace=True)
x
x.isnull().sum()
from sklearn.model_selection import train_test_split

x_train, X_test, y_train, y_test = train_test_split(x, y, test_size =0.2)



from sklearn.ensemble import RandomForestClassifier

from lightgbm import LGBMClassifier



lgbm = LGBMClassifier(random_state=5)
d = RandomForestClassifier()
model=lgbm.fit(x_train,y_train)
from sklearn.metrics import roc_auc_score

roc_auc_score(y_test,model.predict(X_test))