import numpy as np

import pandas as pd

import lightgbm
data=pd.read_csv('../input/creditcardfraud/creditcard.csv')
data.columns
data.head
y=data.Class

data=data.drop('Class',axis=1)

a=data.columns
from sklearn.preprocessing import StandardScaler, RobustScaler



std_scaler = StandardScaler()

rob_scaler = RobustScaler()



data['scaled_amount'] = rob_scaler.fit_transform(data['Amount'].values.reshape(-1,1))

data['scaled_time'] = rob_scaler.fit_transform(data['Time'].values.reshape(-1,1))



data.drop(['Time','Amount'], axis=1, inplace=True)
scaled_amount = data['scaled_amount']

scaled_time = data['scaled_time']



data.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)

data.insert(0, 'scaled_amount', scaled_amount)

data.insert(1, 'scaled_time', scaled_time)



data.head()
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x1,y1,test_size=0.2, random_state=42)
from imblearn.over_sampling import SMOTE

smt=SMOTE(random_state=42)

x1,y1=smt.fit_resample(data,y)
df=pd.DataFrame(x1)
df
df.columns=a
df
df['Class'] = y1
df
categorical_features = [c for c, col in enumerate(data.columns) if 'cat' in col]

train_data = lightgbm.Dataset(x_train,label=y_train,categorical_feature=categorical_features)

test_data = lightgbm.Dataset(x_test,label=y_test)
parameters = {

    'application': 'binary',

    'objective': 'binary',

    'metric': 'auc',

    'is_unbalance': 'true',

    'boosting': 'gbdt',

    'num_leaves': 31,

    'feature_fraction': 0.5,

    'bagging_fraction': 0.5,

    'bagging_freq': 20,

    'learning_rate': 0.05,

    'verbose': 0

}
model = lightgbm.train(parameters,

                       train_data,

                       valid_sets=test_data,

                       num_boost_round=5000,

                       early_stopping_rounds=100)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import *

model=LogisticRegression()

model.fit(x_train,y_train)

pred=model.predict(x_test)

target_names=['class 0','class 1']

print(classification_report(y_test,pred,target_names=target_names))

from sklearn import datasets

import xgboost as xgb

D_train = xgb.DMatrix(x_train, label=y_train)

D_test = xgb.DMatrix(x_test, label=y_test)

param = {

    'eta': 0.3, 

    'max_depth': 3,  

    'objective': 'multi:softprob',  

    'num_class': 3} 



steps = 20



model = xgb.train(param, D_train, steps)



preds2 = model.predict(D_test)

best_preds = np.asarray([np.argmax(line) for line in preds2])



target_names=['class 0','class 1']

print(classification_report(y_test,best_preds,target_names=target_names))
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import *

model=RandomForestClassifier()

model.fit(x_train,y_train)

pred=model.predict(x_test)

target_names=['class 0','class 1']

print(classification_report(y_test,pred,target_names=target_names))