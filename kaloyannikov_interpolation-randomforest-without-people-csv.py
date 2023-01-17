import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestClassifier
###################Data Cleaning#####################



#Loading Data

train = pd.read_csv('../input/act_train.csv')

test = pd.read_csv('../input/act_test.csv')
#Defining dictionary to replace string type

mapping = {}

for i in range(0, 10000):

    mapping['type '+str(i)] = float(i)
#Cleaning train Data

train = train.applymap(lambda s: mapping.get(s) if s in mapping else s)

for i in range(1, 11):

    train['char_'+str(i)][0] = 0.0

for i in range(1, 11):

    train['char_'+str(i)] = train['char_'+str(i)].interpolate(limit=10000)



#Cleaning test data

test = test.applymap(lambda s: float(mapping.get(s)) if s in mapping else s)

for i in range(1, 11):

    test['char_'+str(i)] = test['char_'+str(i)].interpolate(limit=10000)



#If interpolation didn't fill all values

train = train.replace(np.NaN, 0.0)

test = test.replace(np.NaN, 0.0)
#Cleaning dates

def clean(row):

    return row.replace("-","")



train['date'] = train['date'].map(clean)

test['date'] = test['date'].map(clean)
#Training the Random Forest Classifier

features_train = train[['activity_category', 'date', 'char_1', 'char_2', 'char_3', 'char_4', 'char_5', 'char_6', 'char_7', 'char_8', 'char_9', 'char_10']]

target_train = train['outcome']



clf = RandomForestClassifier().set_params(n_estimators=100, n_jobs=-1)

clf.fit(features_train, target_train) 
#Making predictions

features_test = test[['activity_category', 'date', 'char_1', 'char_2', 'char_3', 'char_4', 'char_5', 'char_6', 'char_7', 'char_8', 'char_9', 'char_10']]

predictions = clf.predict(features_test)
#Saving result

csvBuild = pd.DataFrame({'activity_id': test['activity_id'], 'outcome': predictions})

csvBuild.to_csv("submissionRF.csv", index=False)