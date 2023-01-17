import numpy as np
import pandas as pd
import time
import warnings
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import os
print(os.listdir("../input"))
warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.describe()
train['HF1'] = train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Fire_Points']
train['HF2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Fire_Points'])
train['HR1'] = abs(train['Horizontal_Distance_To_Hydrology']+train['Horizontal_Distance_To_Roadways'])
train['HR2'] = abs(train['Horizontal_Distance_To_Hydrology']-train['Horizontal_Distance_To_Roadways'])
train['FR1'] = abs(train['Horizontal_Distance_To_Fire_Points']+train['Horizontal_Distance_To_Roadways'])
train['FR2'] = abs(train['Horizontal_Distance_To_Fire_Points']-train['Horizontal_Distance_To_Roadways'])

# Pythagoras theorem
train['slope_hyd'] = (train['Horizontal_Distance_To_Hydrology']**2+train['Vertical_Distance_To_Hydrology']**2)**0.5
train.slope_hyd=train.slope_hyd.map(lambda x: 0 if np.isinf(x) else x)

# Means
train['Mean_Amenities']=(train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology + train.Horizontal_Distance_To_Roadways) / 3  
train['Mean_Fire_Hyd']=(train.Horizontal_Distance_To_Fire_Points + train.Horizontal_Distance_To_Hydrology) / 2 

# Testing data

test['HF1'] = test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Fire_Points']
test['HF2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Fire_Points'])
test['HR1'] = abs(test['Horizontal_Distance_To_Hydrology']+test['Horizontal_Distance_To_Roadways'])
test['HR2'] = abs(test['Horizontal_Distance_To_Hydrology']-test['Horizontal_Distance_To_Roadways'])
test['FR1'] = abs(test['Horizontal_Distance_To_Fire_Points']+test['Horizontal_Distance_To_Roadways'])
test['FR2'] = abs(test['Horizontal_Distance_To_Fire_Points']-test['Horizontal_Distance_To_Roadways'])

# Pythagoras theorem
test['slope_hyd'] = (test['Horizontal_Distance_To_Hydrology']**2+test['Vertical_Distance_To_Hydrology']**2)**0.5
test.slope_hyd=test.slope_hyd.map(lambda x: 0 if np.isinf(x) else x) # remove infinite value if any

# Means
test['Mean_Amenities']=(test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology + test.Horizontal_Distance_To_Roadways) / 3 
test['Mean_Fire_Hyd']=(test.Horizontal_Distance_To_Fire_Points + test.Horizontal_Distance_To_Hydrology) / 2
train.shape
test.shape
y_train = train['Cover_Type']
x_train = train.drop(['Cover_Type'],axis=1)
preds = pd.DataFrame()
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn import ensemble 
from sklearn.naive_bayes import BernoulliNB
m1 = ensemble.AdaBoostClassifier(ensemble.ExtraTreesClassifier(n_estimators=500), n_estimators=250, learning_rate=0.01, algorithm='SAMME')  
m1.fit(x_train, y_train) 
preds["Model1"] = m1.predict(test)
m2 = ensemble.ExtraTreesClassifier(n_estimators=550)  
m2.fit(x_train, y_train)
preds["Model2"] = m2.predict(test)
m3 = XGBClassifier(max_depth=20, n_estimators=1000)  
m3.fit(x_train, y_train)
preds["Model3"] = m3.predict(test)
m4 = LGBMClassifier(n_estimators=2000, max_depth=15)
m4.fit(x_train, y_train)
preds["Model4"] = m4.predict(test)
m5 = ensemble.AdaBoostClassifier(ensemble.GradientBoostingClassifier(n_estimators=1000, max_depth=10), n_estimators=1000, learning_rate=0.01, algorithm="SAMME")
m5.fit(x_train, y_train)
preds["Model5"] = m5.predict(test)
m6 = SGDClassifier(loss='hinge')
m6.fit(x_train, y_train)
preds["Model6"] = m6.predict(test)
m7 = BernoulliNB()
m7.fit(x_train,y_train)
preds['Model7'] = m7.predict(test)

pred = preds.mode(axis=1)
sub = pd.DataFrame({"Id": test['Id'],"Cover_Type": pred[0].astype('int').values})
sub.to_csv("result.csv", index=False)