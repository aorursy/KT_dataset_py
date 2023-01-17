#  Libraries

import pandas as pd 

import numpy as np

from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score



import warnings

warnings.filterwarnings("ignore")
#Read data

train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')

test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')

test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')
# Merge

train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)

test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)
# Pre-process

train = train.fillna(-999)

test = test.fillna(-999)



label_y = train['isFraud']

del train['isFraud']



# Label Encoding

print('Label Encoding...')

for f in train.columns:

    if train[f].dtype=='object' or test[f].dtype=='object':

        lbl = LabelEncoder()

        lbl.fit(list(train[f].values)+ list(test[f].values))

        train[f] = lbl.transform(list(train[f].values))

        test[f] = lbl.transform(list(test[f].values))
# Create new y label to detect shift covariance

train['origin'] = 0

test['origin'] = 1



# Create a random index to extract random train and test samples

training = train.sample(10000, random_state=12)

testing = test.sample(10000, random_state=11)



## Combine random samples

combi = training.append(testing)

y = combi['origin']

combi.drop('origin',axis=1,inplace=True)



## Modelling

model = RandomForestClassifier(n_estimators = 50, max_depth = 5,min_samples_leaf = 5)

all_scores = []

drop_list = []

score_list =[]

temp = -1

for i in combi.columns:

    temp +=1

    score = cross_val_score(model,pd.DataFrame(combi[i]),y,cv=2,scoring='roc_auc')

    if (np.mean(score) > 0.8):

        drop_list.append(i)

        score_list.append(np.mean(score))

    all_scores.append(np.mean(score))    

    print('Checking feature no {} out of {}'.format(temp, train.shape[1]))

    print(i,np.mean(score))

#Print Top 20 features with possible covariate shift

scores_df = pd.DataFrame({'feature':combi.columns, 

                          'score': all_scores})



scores_df = scores_df.sort_values(by = 'score', ascending = False)

scores_df.head(20)