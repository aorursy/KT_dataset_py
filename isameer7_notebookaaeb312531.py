import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

from sklearn.cross_validation import StratifiedKFold

import xgboost as xgb



act = pd.read_csv('../input/act_train.csv')

people = pd.read_csv('../input/people.csv')

test = pd.read_csv('../input/act_test.csv')



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
# Pre process people data

people.drop('date', axis = 1, inplace = True)



people.rename(columns = lambda x: 'p_'+ x, inplace = True)

people.rename(columns = {'p_people_id' : 'people_id'}, inplace = True)



people['people_id'] = people['people_id'].apply(lambda x: x.split('_')[1])



for col in range(9):

        colname = 'p_char_'+str(col+1)

        people[colname] = people[colname].astype('category')





for col in people.columns.values.tolist():

    if col != 'char_38':

        lbl_enc = LabelEncoder()

        lbl_enc.fit(people[col])

        people[col] = lbl_enc.transform(people[col])
# Pre process act data

outcome = act['outcome']

act.drop(['activity_id', 'date', 'outcome'], axis = 1, inplace = True)



act['people_id'] = act['people_id'].apply(lambda x: x.split('_')[1])



for col in range(10):

    colname = 'char_'+str(col+1)

    act[colname].fillna('type -1', inplace = True)



for col in act.columns.values.tolist():

    if col != 'people_id':

        act[col] = act[col].astype('category')

        lbl_enc = LabelEncoder()

        lbl_enc.fit(act[col])

        act[col] = lbl_enc.transform(act[col])
features = pd.merge(act, people, on = 'people_id', how = 'left')

features.fillna(999, inplace = True)

features.drop(['people_id'], axis = 1, inplace = True)

labels = outcome

del act

del people



#ohe_columns = features.columns.values.tolist()

#ohe_columns.remove('char_10')

#ohe = OneHotEncoder()

#ohe.fit(features[ohe_columns])

#features = ohe.transform(features[ohe_columns])

#ohe.fit(labels)

#features = ohe.transform(labels)
kf = StratifiedKFold(labels, round(1. / eval_size))

train_indices, valid_indices = next(iter(kf))

X_train, y_train = features.loc[train_indices], labels.loc[train_indices]

X_valid, y_valid = features.loc[valid_indices], labels.loc[valid_indices]

del features

del labels
clf = svm.SVC(gamma = 0.01, C=100)

clf.fit(X_train, y_train)

predicted = clf.predict(X_valid)

metrics.roc_auc_score(y_valid, predicted)