import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
a_train = pd.read_csv('../input/act_train.csv', parse_dates=['date'])

a_test = pd.read_csv('../input/act_test.csv', parse_dates=['date'])

people = pd.read_csv('../input/people.csv', parse_dates=['date'])
train_date_id = a_train[['activity_id','date', 'people_id', 'outcome']]

test_date_id = a_test[['activity_id','date', 'people_id']]
train_date_id.head()
a_train.head()
# Save the test IDs for Kaggle submission

test_ids = a_test['activity_id']



def preprocess_acts(data, train_set=True):

    

    # Getting rid of data feature for now

    data = data.drop(['date', 'activity_id'], axis=1)

    if(train_set):

        data = data.drop(['outcome'], axis=1)

    

    ## Split off _ from people_id

    data['people_id'] = data['people_id'].apply(lambda x: x.split('_')[1])

    data['people_id'] = pd.to_numeric(data['people_id']).astype(int)

    

    columns = list(data.columns)

    

    # Convert strings to ints

    for col in columns[1:]:

        data[col] = data[col].fillna('type 0')

        data[col] = data[col].apply(lambda x: x.split(' ')[1])

        data[col] = pd.to_numeric(data[col]).astype(int)

    return data



def preprocess_people(data):

    

    

    data = data.drop(['date'], axis=1)

    data['people_id'] = data['people_id'].apply(lambda x: x.split('_')[1])

    data['people_id'] = pd.to_numeric(data['people_id']).astype(int)

    

    #  Values in the people df is Booleans and Strings    

    columns = list(data.columns)

    bools = columns[11:]

    strings = columns[1:11]

    

    for col in bools:

        data[col] = pd.to_numeric(data[col]).astype(int)        

    for col in strings:

        data[col] = data[col].fillna('type 0')

        data[col] = data[col].apply(lambda x: x.split(' ')[1])

        data[col] = pd.to_numeric(data[col]).astype(int)

    return data
# Preprocess each df

pp_people = preprocess_people(people)

pp_train = preprocess_acts(a_train)

pp_test = preprocess_acts(a_test, train_set=False)
pp_test.isnull().sum()
# Merege into a unified table



# Training 

features = pp_train.merge(pp_people, how='left', on='people_id')

labels = train_date_id['outcome']

# Testing

test = pp_test.merge(pp_people, how='left', on='people_id')



# Check it out...

features.sample(10)
features.info()
## Split Training Data

from sklearn.cross_validation import train_test_split



X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size = 0.2, random_state=42)



## Out of box random forest

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.grid_search import GridSearchCV



clf = RandomForestClassifier()

clf.fit(X_train, y_train)
## Training Predictions

proba = clf.predict_proba(X_test)

preds = proba[:,1]

score = roc_auc_score(y_test, preds)

print("Area under ROC {0}".format(score))
# Test Set Predictions

test_proba = clf.predict_proba(test)

test_preds = test_proba[:,1]



# Format for submission

submission_redhat_0 = pd.DataFrame({ 'activity_id' : test_ids, 'outcome': test_preds })

submission_redhat_0.to_csv('subm_redhat_0.csv', index = False)