# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#from subprocess import check_output

#print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn.cross_validation import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,roc_auc_score



def cleanPeople(people):

    

    people = people.drop(['date'],axis=1)

    people['people_id'] = people['people_id'].apply(lambda x : x.split('_')[1])

    people['people_id'] = pd.to_numeric(people['people_id']).astype(int)

    

    fields = list(people.columns)

    cat_data = fields[1:11]

    bool_data = fields[11:]

    

    for data in cat_data:

        people[data] = people[data].fillna('type 0')

        people[data] = people[data].apply(lambda x: x.split(' ')[1])

        people[data] = pd.to_numeric(people[data]).astype(int)

    

    for data in bool_data:

        people[data] = pd.to_numeric(people[data]).astype(int)

        

    

    return people



def cleanAct(data, train=False):

    

    data = data.drop(['date'],axis = 1)

    if train:

        data = data.drop(['outcome'],axis=1)

        

    data['people_id'] = data['people_id'].apply(lambda x : x.split('_')[1])

    data['people_id'] = pd.to_numeric(data['people_id']).astype(int)

    

    data['activity_id'] = data['activity_id'].apply(lambda x: x.split('_')[1])

    data['activity_id'] = pd.to_numeric(data['activity_id']).astype(int)

    

    fields = list(data.columns)

    cat_data = fields[2:13]

    

    for column in cat_data:

        data[column] = data[column].fillna('type 0')

        data[column] = data[column].apply(lambda x : x.split(' ')[1])

        data[column] = pd.to_numeric(data[column]).astype(int)

     

    return data    







people = pd.read_csv("../input/people.csv")

people = cleanPeople(people)



act_train = pd.read_csv("../input/act_train.csv",parse_dates=['date'])

act_train_cleaned = cleanAct(act_train,train=True)



act_test = pd.read_csv("../input/act_test.csv",parse_dates=['date'])

act_test_cleaned = cleanAct(act_test)









train = act_train_cleaned.merge(people,on='people_id', how='left')

test = act_test_cleaned.merge(people, on='people_id', how='left')







train = train.drop(['people_id'],axis=1)

train = train.drop(['activity_id'],axis=1)





test = test.drop(['people_id','activity_id'],axis=1)





output = act_train['outcome']



X_train, X_test, y_train, y_test = train_test_split(train,output, test_size=0.2, random_state =7)
model = xgb.XGBClassifier(max_depth=8,n_estimators=500,learning_rate=0.1,objective='binary:logistic',seed =7,reg_lambda=1)
model.fit(X_train,y_train)
results = model.predict_proba(X_test)

s = results[:,1]

score = roc_auc_score(y_test,s)

print(score)
results = model.predict_proba(test)

s = results[:,1]

activity = act_test['activity_id']



result = pd.DataFrame({'activity_id': activity, 'outcome': s})

result.to_csv("Result.csv",index=False)