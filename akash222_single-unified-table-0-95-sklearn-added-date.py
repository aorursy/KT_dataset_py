import numpy as np

import pandas as pd

import datetime



act_train = pd.read_csv('../input/act_train.csv')

act_test = pd.read_csv('../input/act_test.csv')

people = pd.read_csv('../input/people.csv')

people.sample(10)
def process_dates(data,min_date):

    #min_date=data.min()

    min_date 

    data=data.apply(lambda x: (datetime.datetime.strptime(x,"%Y-%m-%d")

                                    -datetime.datetime.strptime(min_date,"%Y-%m-%d")).days)

    data

    return data
# Save the test IDs for Kaggle submission

test_ids = act_test['activity_id']



def preprocess_acts(data,min_date, train_set=True):

    

    # Getting rid of data feature for now

    dates=data['date']

    dates=process_dates(dates,min_date)

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

#    for column in columns[1:]:

#        dummies = pd.get_dummies(data[column])

#        data[dummies.columns] = dummies

    data['dates']=dates

    return data



def preprocess_people(data,min_date):

    dates=data['date']

    dates=process_dates(dates,min_date)

    # TODO refactor this duplication

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

    #data = data.drop(['group_1'], axis=1)

#    for column in strings:

#        dummies = pd.get_dummies(data[column])

#        data[dummies.columns] = dummies

    data['dates']=dates

    return data
#find minimum date

min_date=pd.concat([people['date'],act_train['date'],act_test['date']]).min()

min_date
# Preprocess each df

min_date=pd.concat([people['date'],act_train['date'],act_test['date']]).min()

peeps = preprocess_people(people,min_date)

actions_train = preprocess_acts(act_train,min_date,train_set=True)

actions_test = preprocess_acts(act_test,min_date,train_set=False)

print (peeps.columns)

print (actions_train.columns)

peeps.sample(10)
actions_train.sample(10)
#run k-means to find the nearby groups
# Merege into a unified table



# Training 

features = actions_train.merge(peeps, how='left', on='people_id')

features=features.drop(['people_id'],axis=1)

labels = act_train['outcome']



# Testing

test = actions_test.merge(peeps, how='left', on='people_id')

test=test.drop(['people_id'],axis=1)

# Check it out...

features.sample(10)
columnss=list(features.columns)

columnss

#features['group_1'].nunique()
## Split Training Data

from sklearn.cross_validation import train_test_split



num_test = 0.10

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=num_test, random_state=23)



## Out of box random forest

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, roc_auc_score

#from sklearn.grid_search import GridSearchCV

#clf=GradientBoostingClassifier()

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

test_res=clf.predict(test)



# Format for submission

output = pd.DataFrame({ 'activity_id' : test_ids, 'outcome': test_preds })

output1 = pd.DataFrame({ 'activity_id' : test_ids, 'outcome': test_res })

output.head()
output.to_csv('redhat.csv', index = False)

output1.to_csv('redhat_noprpba.csv', index = False)