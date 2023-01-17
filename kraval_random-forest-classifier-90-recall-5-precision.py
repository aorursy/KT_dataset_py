# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/creditcard.csv"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



#pd options

pd.options.display.max_columns = None

pd.options.display.max_colwidth = -1
ccard = pd.read_csv("../input/creditcard.csv")
ccard.describe()
len(ccard[ccard['Class']==1])



#Heavily Unbalanced dataset
#Split the data into training, validation and test



mask = np.random.rand(len(ccard))<0.8

ccard_trainVal = ccard[mask]

ccard_test = ccard[~mask]

mask2 = np.random.rand(len(ccard_trainVal))<0.85

ccard_train = ccard_trainVal[mask2]

ccard_val = ccard_trainVal[~mask2]



print (len(ccard_train), len(ccard_val), len(ccard_test))
len(ccard_train[ccard_train['Class']==1]) #~300 - 340 of total 492 frauds
ccard_train.columns
print (len(ccard_train[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',\

                        'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 

                        'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']].drop_duplicates()))

print (len(ccard_train.drop_duplicates()))



#equal numbers indicate that there does not exist a pair of observation with same values but different classes.

#Before proceeding with undersampling, it would be preferred to remove duplicates.

#Removal of duplicates could be carried out prior to training-validation-test split as well.

ccard_train.drop_duplicates(inplace=True)
def Amount_fractional(r):

    return r['Amount'] - int(r['Amount'])



ccard_train['Amount_fractional'] = ccard_train.apply(Amount_fractional, axis=1)

ccard_ints = ccard_train[ccard_train['Amount_fractional']==0]

ccard_floats = ccard_train[ccard_train['Amount_fractional']>0]

print(len(ccard_ints[ccard_ints['Class']==1]),len(ccard_ints))

print(len(ccard_floats[ccard_floats['Class']==1]),len(ccard_floats))
len(ccard_train[ccard_train['Class']==1])
ccard_train.columns
#Undersampling

fraud = ccard_train[ccard_train['Class']==1]

nonFraud = ccard_train[ccard_train['Class']==0].sample(frac=1).head(len(ccard_train[ccard_train['Class']==1])) # perfectly balanced

frame = [fraud, nonFraud]

ccardUSample = pd.concat(frame)

print("Size of sample", len(ccardUSample))

ccard_features = ccardUSample[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',\

                               'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 

                               'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Amount_fractional']]

ccard_labels = ccardUSample['Class'].tolist()
from sklearn.neighbors import KNeighborsClassifier

NN = KNeighborsClassifier(n_neighbors = 5, weights = 'distance')

NN.fit(ccard_features, ccard_labels)
ccard_val['Amount_fractional'] = ccard_val.apply(Amount_fractional, axis=1)

valid_features = ccard_val[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11',\

                               'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 

                               'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount', 'Amount_fractional']]

valid_labels = ccard_val['Class'].tolist()
pred = NN.predict_proba(valid_features).tolist()

prediction = NN.predict(valid_features).tolist()
prob_fraud = [x[1] for x in pred]
pred_prob = [1*(x>0.8) for x in prob_fraud]
from sklearn import metrics

precision = metrics.precision_score(valid_labels,prediction)

recall = metrics.recall_score(valid_labels,prediction)
print (precision, recall) # A random allotment inder an identical recall would have half the precision(23TP instead of 44 out of 63positives)

#precision = 44/12489 ; recall = 44/63 in one of the runs
def evaluatePerf(true_labels, predicted_labels):

    type2 = 0

    type1 = 0

    true_positive = 0

    true_negative = 0

    for x,y in zip(true_labels, predicted_labels):

        if x == y:

            if x == 1:

                true_positive+=1

            else:

                true_negative+=1

        elif x == 1:

            type2 += 1

        elif x == 0:

            type1 += 1



    print("TP:", true_positive, " TN:", true_negative, " T1Err:", type1, " T2Err:", type2)

evaluatePerf(valid_labels,prediction)
precision = metrics.precision_score(valid_labels,pred_prob)

recall = metrics.recall_score(valid_labels,pred_prob)



print (precision, recall)

#precision = 29/5446; recall = 29/63
evaluatePerf(valid_labels,pred_prob)
from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier(n_estimators = 50)

RFC.fit(ccard_features, ccard_labels)
prediction = RFC.predict(valid_features)

from sklearn import metrics

precision = metrics.precision_score(valid_labels,prediction)

recall = metrics.recall_score(valid_labels,prediction)



print (precision, recall)

print( "reported: ", sum(prediction), ' and total:', len(prediction))
evaluatePerf(valid_labels,prediction)
RFC.feature_importances_
ccard_features.columns
ccard_test['Amount_fractional'] = ccard_test.apply(Amount_fractional, axis=1)

test_feature = ccard_test[['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10', 'V11', 'V12',\

                           'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 'V22', 'V23', 'V24',

                           'V25', 'V26', 'V27', 'V28', 'Amount', 'Amount_fractional']]

test_labels = ccard_test['Class'].tolist()
pred_test = RFC.predict(test_feature)

precision = metrics.precision_score(test_labels,pred_test)

recall = metrics.recall_score(test_labels,pred_test)



print (precision, recall)

print( "reported: ", sum(pred_test), ' and total:', len(pred_test))
evaluatePerf(test_labels,pred_test)