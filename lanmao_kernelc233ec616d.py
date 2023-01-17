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
loc_train = "../input/train.csv"
loc_test = "../input/test.csv"
loc_submission = "kaggle.rf200.entropy.submission.csv"

loc_train
df_train = pd.read_csv(loc_train)
df_test = pd.read_csv(loc_test)
df_train.head(10)
cols_to_normalize = ['Aspect','Slope','Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
  'Hillshade_9am','Hillshade_Noon','Hillshade_3pm','Horizontal_Distance_To_Fire_Points']
from sklearn.preprocessing  import normalize
import math
df_train[cols_to_normalize] = normalize(df_train[cols_to_normalize])
df_test[cols_to_normalize] = normalize(df_test[cols_to_normalize])
feature_cols = [col for col in df_train.columns if col not in ['Cover_Type','Id']]
  
feature_cols.append('binned_elevation')
feature_cols.append('Horizontal_Distance_To_Roadways_Log')
feature_cols.append('Soil_Type12_32')
feature_cols.append('Soil_Type23_22_32_33')
df_train['binned_elevation'] = [math.floor(v/50.0) for v in df_train['Elevation']]
df_test['binned_elevation'] = [math.floor(v/50.0) for v in df_test['Elevation']]
df_train['Horizontal_Distance_To_Roadways_Log'] = [math.log(v+1) for v in df_train['Horizontal_Distance_To_Roadways']]
df_test['Horizontal_Distance_To_Roadways_Log'] = [math.log(v+1) for v in df_test['Horizontal_Distance_To_Roadways']]

df_train['Soil_Type12_32'] = df_train['Soil_Type32'] + df_train['Soil_Type12']
df_test['Soil_Type12_32'] = df_test['Soil_Type32'] + df_test['Soil_Type12']
df_train['Soil_Type23_22_32_33'] = df_train['Soil_Type23'] + df_train['Soil_Type22'] + df_train['Soil_Type32'] + df_train['Soil_Type33']
df_test['Soil_Type23_22_32_33'] = df_test['Soil_Type23'] + df_test['Soil_Type22'] + df_test['Soil_Type32'] + df_test['Soil_Type33']
 
df_train_1_2 = df_train[(df_train['Cover_Type'] <= 2)]
df_train_3_4_6 = df_train[(df_train['Cover_Type'].isin([3,4,6]))]
X_train = df_train[feature_cols]
X_test = df_test[feature_cols]

X_train_1_2 = df_train_1_2[feature_cols]
X_train_3_4_6 = df_train_3_4_6[feature_cols]

y = df_train['Cover_Type']
y_1_2 = df_train_1_2['Cover_Type']
y_3_4_6 = df_train_3_4_6['Cover_Type']

test_ids = df_test['Id']
from sklearn import ensemble
clf = ensemble.ExtraTreesClassifier(n_estimators=100,n_jobs=-1,random_state=0)
clf.fit(X_train, y)

clf_1_2 = ensemble.RandomForestClassifier(n_estimators=200,n_jobs=-1,random_state=0)
clf_1_2.fit(X_train_1_2, y_1_2)

clf_3_4_6 = ensemble.RandomForestClassifier(n_estimators=200,n_jobs=-1,random_state=0)
clf_3_4_6.fit(X_train_3_4_6, y_3_4_6)
vals_1_2 = {}
for e, val in enumerate(list(clf_1_2.predict_proba(X_test))):
    vals_1_2[e] = val
vals_3_4_6= {}
for e, val in enumerate(list(clf_3_4_6.predict_proba(X_test))):
    vals_3_4_6[e] = val 
vals = {}
for e, val in enumerate(list(clf.predict(X_test))):
    vals[e] = val 
def largest_index(inlist):
    largest = -1
    largest_index = 0
    for i in range(len(inlist)):
        item = inlist[i]
        if item > largest:
            largest = item
            largest_index = i
    return largest_index
with open(loc_submission, "w") as outfile:
    outfile.write("Id,Cover_Type\n")
    for e, val in enumerate(list(clf.predict_proba(X_test))):
      #boost types 1 and 2
        val[0] += vals_1_2[e][0]/1.3
        val[1] += vals_1_2[e][1]/1.1
        #boost types 3,4, and 6
        val[2] += vals_3_4_6[e][0]/3.4
        val[3] += vals_3_4_6[e][1]/4.0
        val[5] += vals_3_4_6[e][2]/3.6
      #val[4] += vals_5_7[e][0]/2.4
      #val[6] += vals_5_7[e][1]/3.4
        i = largest_index(val)
        v = i  + 1
        outfile.write("%s,%s\n"%(test_ids[e],v))
 
