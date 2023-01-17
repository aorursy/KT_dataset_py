# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_filepath = "../input/train.csv"
test_filepath = "../input/test.csv"
Building_Structure_filepath = "../input/Building_Structure.csv"
Building_Ownership_Use_filepath = "../input/Building_Ownership_Use.csv"
res_train_filepath = "merged_train.csv"
res_test_filepath = "merged_test.csv"

train = pd.read_csv(train_filepath)
Building_Ownership_Use = pd.read_csv(Building_Ownership_Use_filepath)
test = pd.read_csv(test_filepath)
Building_Structure = pd.read_csv(Building_Structure_filepath)

Building_Info = pd.merge(Building_Ownership_Use,Building_Structure,on=['building_id','vdcmun_id','district_id','ward_id'],how='inner')

res_train = pd.merge(train,Building_Info,on=['building_id','vdcmun_id','district_id'],how='inner')
res_test = pd.merge(test,Building_Info,on=['building_id','vdcmun_id','district_id'],how='inner')

res_train.to_csv(res_train_filepath,index=False)
res_test.to_csv(res_test_filepath,index=False)
# read the merged dataset
mtrain_filepath = "merged_train.csv"
mtest_filepath = "merged_test.csv"

train = pd.read_csv(mtrain_filepath)
test = pd.read_csv(mtest_filepath)

train.head(n=4)
test.head(n=4)
train['dist_vdc_ward_id'] = train['district_id'].map(str) + train['vdcmun_id'].map(str) + train['ward_id'].map(str)
test['dist_vdc_ward_id'] = test['district_id'].map(str) + test['vdcmun_id'].map(str) + test['ward_id'].map(str)

train['count_floors_diff'] = train['count_floors_pre_eq'] - train['count_floors_post_eq']
test['count_floors_diff'] = test['count_floors_pre_eq'] - test['count_floors_post_eq']

train['height_ft_diff'] = train['height_ft_pre_eq'] - train['height_ft_post_eq']
test['height_ft_diff'] = test['height_ft_pre_eq'] - test['height_ft_post_eq']

train['risk_count'] = train['has_geotechnical_risk_other'] + train['has_geotechnical_risk_liquefaction'] + train['has_geotechnical_risk_landslide'] + train['has_geotechnical_risk_flood'] + train['has_geotechnical_risk_rock_fall'] + train['has_geotechnical_risk_land_settlement'] + train['has_geotechnical_risk_fault_crack']
test['risk_count'] = test['has_geotechnical_risk_other'] + test['has_geotechnical_risk_liquefaction'] + test['has_geotechnical_risk_landslide'] + test['has_geotechnical_risk_flood'] + test['has_geotechnical_risk_rock_fall'] + test['has_geotechnical_risk_land_settlement'] + test['has_geotechnical_risk_fault_crack']

from sklearn.preprocessing import Imputer

imput = Imputer(strategy='most_frequent')
imput1 = Imputer(strategy='mean')
train['count_families'] = imput1.fit_transform(train[['count_families']]).astype('int')
train['has_repair_started'] = imput.fit_transform(train[['has_repair_started']]).astype('int')

test['has_repair_started'] = imput.fit_transform(test[['has_repair_started']]).astype('int')

train['has_geotechnical_risk'] = train['has_geotechnical_risk'].map(int)
train['has_secondary_use'] = train['has_secondary_use'].map(int)

test['has_geotechnical_risk'] = test['has_geotechnical_risk'].map(int)
test['has_secondary_use'] = test['has_secondary_use'].map(int)
test['count_families'] = test['count_families'].map(int)

grade_enc = {
	'Grade 1':1,
	'Grade 2':2,
	'Grade 3':3,
	'Grade 4':4,
	'Grade 5':5
}

train_y = pd.Series([grade_enc[i] for i in train['damage_grade']])

train_x = train.drop(columns=['building_id','damage_grade'])
test_x = test.drop(columns=['building_id'])
print(test_x.shape)
from sklearn import preprocessing

label_enc = preprocessing.LabelEncoder()

for col in test_x.columns.values:
    if test_x[col].dtype == 'object':
        data = train_x[col].append(test_x[col])
        label_enc.fit(data)
        train_x[col] = label_enc.transform(train_x[col])
        test_x[col] = label_enc.transform(test_x[col])
import time
from sklearn.ensemble import RandomForestClassifier

prev_time = time.time()

clf = RandomForestClassifier(n_jobs=-1,n_estimators=100,max_features=.33)
clf.fit(train_x,train_y)

new_time = time.time()

total_time = round((new_time - prev_time),2)
print('total time : ',total_time)
import time

prev_time = time.time()

predictions_train = clf.predict(train_x)
predictions_test = clf.predict(test_x)

new_time = time.time()

total_time = round((new_time - prev_time),2)
print('total time : ',total_time)
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

print("Train Accuracy : ",accuracy_score(train_y,predictions_train))

print("\nConfusion Matrix : ")
print(confusion_matrix(train_y,predictions_train))
predictions_test = pd.Series(predictions_test)

predict_test = 'Grade ' + predictions_test.map(str)

subm = {
    'building_id' : test['building_id'],
    'damage_grade' : predict_test
}
submissions = pd.DataFrame(subm)
submissions = submissions.set_index('building_id')

print(submissions.head())

submissions.to_csv('submission.csv')