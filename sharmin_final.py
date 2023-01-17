import numpy as np 

import pandas as pd 

from subprocess import check_output

from sklearn.feature_selection import RFE

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from sklearn.metrics import roc_curve, auc

    

def trainSetOverview():

    global train

    print ("Training data overview \n")

    print ("Column Headers:", list(train.columns.values), "\n")

    print (train.dtypes)

    for col in train:

        unique = train[col].unique()

        print ('\n' + str(col) + ' has ' + str(unique.size) + ' unique values')

        if (True in pd.isnull(unique)):

            print (str(col) + ' has ' + str(pd.isnull(train[col]).sum()) + ' missing values \n')



def processData():

    global train, test, people

    print ("Processing the datasets.. \n")

    for data in [train,test]:

        for i in range(1,11):

            data['char_'+str(i)].fillna('type -1', inplace = 'true')

            data['char_'+str(i)] = data['char_'+str(i)].str.lstrip('type ').astype(np.int32)

        

        data['activity_category'] = data['activity_category'].str.lstrip('type ').astype(np.int32)

    

        data['year'] = data['date'].dt.year

        data['month'] = data['date'].dt.month

        data['day'] = data['date'].dt.day

        data.drop('date', axis=1, inplace=True)

    

    for i in range(1,10):

        people['char_' + str(i)] = people['char_' + str(i)].str.lstrip('type ').astype(np.int32)

    for i in range(10, 38):

        people['char_' + str(i)] = people['char_' + str(i)].astype(np.int32)

    

    people['group_1'] = people['group_1'].str.lstrip('group ').astype(np.int32)

    people['year'] = people['date'].dt.year

    people['month'] = people['date'].dt.month

    people['day'] = people['date'].dt.day

    people.drop('date', axis=1, inplace=True)

    

def merge():

    global train, test, people

    print ("Merging the datasets.. \n")



    train = pd.merge(train, people, how='left', on='people_id', left_index=True)

    train.fillna(-1, inplace=True)

    test = pd.merge(test, people, how='left', on='people_id', left_index=True)

    test.fillna(-1, inplace=True)



    train = train.drop(['people_id'], axis=1)

    

def featureRanking():

    global train

    Y = train['outcome']

    X = train.drop(['outcome'], axis=1)

    X = X.iloc[:,1:]

    

    X = X.drop(['char_1_x','char_3_x','char_4_x','char_5_x','char_9_x','char_10_x','day_x','day_y','char_31','char_29'], axis=1)

    

    model(X,Y)

    

def model(X,Y):

    global train,test

    rfc = RandomForestClassifier(n_estimators=96)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)

    print("model")

    clf = rfc.fit(X_train, y_train)

    scores = clf.score(X_test, y_test) 

    print(scores.mean())

    test = test.drop(['people_id'], axis=1)

    test_x = test.iloc[:, 1:]

    test_x = test_x.drop(['char_1_x','char_3_x','char_4_x','char_5_x','char_9_x','char_10_x','day_x','day_y','char_31','char_29'], axis=1)

    predictions = list(map(int, rfc.predict(test_x)))

    test['outcome'] = predictions

    test[['activity_id', 'outcome']].to_csv('submission.csv', index=False)

    

def main():

    trainSetOverview()

    processData()

    merge()

    featureRanking()



print ("Loading input files.. \n")

people = pd.read_csv('../input/people.csv',

                       dtype={'people_id': np.str,

                              'activity_id': np.str,

                              'char_38': np.int32},

                       parse_dates=['date'])

train = pd.read_csv(r'../input/act_train.csv',

                        dtype={'people_id': np.str,

                               'activity_id': np.str,

                               'outcome': np.int8},

                        parse_dates=['date'])

test = pd.read_csv('../input/act_test.csv',

                       dtype={'people_id': np.str,

                              'activity_id': np.str},

                       parse_dates=['date'])



missing_values = []





main()