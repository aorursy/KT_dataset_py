import pandas as pd

import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split

import xgboost as xgb

import warnings

warnings.filterwarnings('ignore')
train_df= pd.read_csv("../input/sf-crime/train.csv.zip")

test_df = pd.read_csv("../input/sf-crime/test.csv.zip")
leCrime =LabelEncoder()

crime = leCrime.fit_transform(train_df.Category)
days = pd.get_dummies(train_df.DayOfWeek)

district = pd.get_dummies(train_df.PdDistrict)
train_df.Dates=pd.to_datetime(train_df.Dates)
hour = train_df.Dates.dt.hour

hour = pd.get_dummies(hour)
trainDate = pd.concat([hour,days,district],axis=1)

trainDate['crime']=crime
days = pd.get_dummies(test_df.DayOfWeek)

district = pd.get_dummies(test_df.PdDistrict)

test_df.Dates=pd.to_datetime(test_df.Dates)

hour = test_df.Dates.dt.hour

hour = pd.get_dummies(hour)
testDate = pd.concat([hour,days,district],axis=1)
# 添加犯罪的小时时间点作为特征

features = ['Friday', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',

'Wednesday', 'BAYVIEW', 'CENTRAL', 'INGLESIDE', 'MISSION',

'NORTHERN', 'PARK', 'RICHMOND', 'SOUTHERN', 'TARAVAL', 'TENDERLOIN']

hourFea = [x for x in range(0,24)]

features = features + hourFea
training,validation = train_test_split(trainDate,train_size=.8)
from sklearn.naive_bayes import BernoulliNB,MultinomialNB

from sklearn.linear_model import LogisticRegression

import time

from sklearn.metrics import log_loss
model = BernoulliNB()

nbstart= time.time()

model.fit(training[features],training['crime'])

nbCostTime = time.time()-nbstart

predicted = np.array(model.predict_proba(validation[features]))

print("耗时: ",nbCostTime)

# print("log_loss:  ",log_loss(validation['crime'],predicted))

# print(model.score(validation[features],validation['crime']))
predict = model.predict_proba(testDate)
submission=pd.DataFrame(np.c_[test_df.Id,predict],

                              columns=['Id']+list(leCrime.classes_))
submission['Id']=submission['Id'].astype(int)
submission.to_csv("submission.csv",index=False)