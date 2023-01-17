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
# Getting the Input names

names_list = pd.read_csv('../input/NationalNames.csv')
name_gender = names_list[['Name','Gender']].drop_duplicates().copy()



name_count = name_gender.groupby('Name',as_index=False).Gender.count().copy()



name_count_1 = name_count[(name_count.Gender == 1)].copy()



name_count_2 = name_count_1.drop('Gender',axis = 1).copy()



name_gender_1 = name_gender[(name_gender.Name.isin(name_count_2.Name))].copy()



name_gender_1.Name = name_gender_1.Name.str.lower()



vowels = ('a','e','i','o','u')



#Checking if name starts/ends with vowels/consonants

name_gender_1.loc[:,'last_char_vow'] = np.where(name_gender_1.Name.str.endswith(vowels),1,0)

name_gender_1.loc[:,'first_char_vow'] = np.where(name_gender_1.Name.str.startswith(vowels),1,0)



name_gender_1.loc[:,'name_len'] = name_gender_1.Name.str.len()



unq = []



for name in name_gender_1['Name']:

    unq.append(len(list(set(name))))



name_gender_1['unq_char'] = unq



import re

rep_char = []



for name in name_gender_1['Name']:

    rep_char.append(len(re.findall(r"((.)\2{1,})",name)))



name_gender_1['rep_char_count'] = rep_char



name_gender_1.loc[:,'rep_char_pres'] = np.where(name_gender_1.rep_char_count > 0,1,0)



rep_char_start = []

rep_char_end = []

for name in name_gender_1['Name']:

    l = list(name)

    l_start = l[:2]

    l_end = l[-2:]

    if len(l_start) == len(set(l_start)):

        rep_char_start.append(0)

    else:

        rep_char_start.append(1)

    if len(l_end) == len(set(l_end)):

        rep_char_end.append(0)

    else:

        rep_char_end.append(1)



name_gender_1['rep_char_start_flg'] = rep_char_start 

name_gender_1['rep_char_end_flg'] = rep_char_end





unq_name = []

for name in name_gender_1['Name']:

    if len(list(name)) == len(list(set(name))):

        unq_name.append(1)

    else:

        unq_name.append(0)



name_gender_1['uniq_name'] = unq_name.copy() 

all_names = name_gender_1.copy()
from sklearn.model_selection import train_test_split



num_test = 0.30

train_data,validation_data = train_test_split(all_names, test_size=num_test, random_state=23)
from sklearn import preprocessing

def encode_features(df_train, df_test):

    features = ['Gender','last_char_vow','first_char_vow','rep_char_pres','rep_char_start_flg','rep_char_end_flg','uniq_name']

    df_combined = pd.concat([df_train[features], df_test[features]])

    

    for feature in features:

        le = preprocessing.LabelEncoder()

        le = le.fit(df_combined[feature])

        df_train[feature] = le.transform(df_train[feature])

        df_test[feature] = le.transform(df_test[feature])

    return df_train, df_test

    

data_train, data_test = encode_features(train_data, validation_data)
from sklearn.preprocessing import OneHotEncoder

enc=OneHotEncoder(sparse=False)

data_train_1=data_train

data_test_1=data_test

columns=['last_char_vow','first_char_vow','rep_char_pres','rep_char_start_flg','rep_char_end_flg','uniq_name']

for col in columns:

       # creating an exhaustive list of all possible categorical values

        data=data_train[[col]].append(data_test[[col]])

        enc.fit(data)

       # Fitting One Hot Encoding on train data

        temp = enc.transform(data_train[[col]])

       # Changing the encoded features into a data frame with new column names

        temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col]

            .value_counts().index])

       # In side by side concatenation index values should be same

       # Setting the index values similar to the X_train data frame

        temp=temp.set_index(data_train.index.values)

       # adding the new One Hot Encoded varibales to the train data frame

        data_train_1=pd.concat([data_train_1,temp],axis=1)

       # fitting One Hot Encoding on test data

        temp = enc.transform(data_test[[col]])

       # changing it into data frame and adding column names

        temp=pd.DataFrame(temp,columns=[(col+"_"+str(i)) for i in data[col]

            .value_counts().index])

       # Setting the index for proper concatenation

        temp=temp.set_index(data_test.index.values)

       # adding the new One Hot Encoded varibales to test data frame

        data_test_1=pd.concat([data_test_1,temp],axis=1)
from sklearn.model_selection import train_test_split



X_all = data_train_1

y_all = data_train_1['Gender']



num_test = 0.20

X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=23)



X_train_1 = X_train.drop(['Gender', 'Name','last_char_vow','first_char_vow',

                          'rep_char_pres','rep_char_start_flg','rep_char_end_flg','uniq_name'], axis=1)

X_test_1 = X_test.drop(['Gender', 'Name','last_char_vow','first_char_vow',

                          'rep_char_pres','rep_char_start_flg','rep_char_end_flg','uniq_name'], axis=1)



data_test_1 = data_test_1.drop(['last_char_vow','first_char_vow',

                          'rep_char_pres','rep_char_start_flg','rep_char_end_flg','uniq_name'], axis=1)
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer, accuracy_score

from sklearn.model_selection import GridSearchCV



# Choose the type of classifier. 

clf = RandomForestClassifier()



# Choose some parameter combinations to try

parameters = {'n_estimators': [4, 6, 9], 

              'max_features': ['log2', 'sqrt','auto'], 

              'criterion': ['entropy', 'gini'],

              'max_depth': [2, 3, 5, 10], 

              'min_samples_split': [2, 3, 5],

              'min_samples_leaf': [1,5,8]

             }



# Type of scoring used to compare parameter combinations

acc_scorer = make_scorer(accuracy_score)



# Run the grid search

grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)

grid_obj = grid_obj.fit(X_train_1, y_train)



# Set the clf to the best combination of parameters

clf = grid_obj.best_estimator_



# Fit the best algorithm to the data. 

clf.fit(X_train_1, y_train)



predictions = clf.predict(X_test_1)

print(accuracy_score(y_test, predictions))
from sklearn.model_selection import KFold 

kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None) 



outcomes = []

for train_index, test_index in kf.split(X_train_1):

    print("Train:", train_index, "Validation:",test_index)

    X_train, X_test = X_train_1.values[train_index], X_train_1.values[test_index]

    y_train, y_test = y_all.values[train_index], y_all.values[test_index]

    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    outcomes.append(accuracy)

    print("Accuracy:".format(accuracy))     

    mean_outcome = np.mean(outcomes)

    print("Mean Accuracy: {0}".format(mean_outcome))
ids = data_test_1['Name']



predictions = clf.predict(data_test_1.drop(['Name','Gender'], axis=1))



output = pd.DataFrame({ 'Name' : ids, 'Gender': predictions })

# output.to_csv('titanic-predictions.csv', index = False)

#output.head()
merge = output.merge(data_test_1[['Name','Gender']],left_on='Name', right_on='Name', how='left')



actuals = data_test_1.Gender



accuracy = accuracy_score(actuals,output['Gender'])

print(accuracy)