import time

import pandas as pd

from sklearn.cross_validation import train_test_split



from sklearn import ensemble

from sklearn.metrics import accuracy_score

import numpy as np



loc_test = "../input/test.csv"

loc_train = "../input/train.csv"

loc_submission = "forest-cover-type-prediction.AspiringGuru.csv"



df_test = pd.read_csv(loc_test)

df_train = pd.read_csv(loc_train)



print ("type(df_test)=", type(df_test), "df_test.shape=", df_test.shape)

print ("type(df_train)=", type(df_train), "df_train.shape=", df_train.shape)



#build list of all columns except the ones we don't want.

# ('Cover_Type' is the predicted value), 'Id' is a unique row identifier

feature_cols = [col for col in df_train.columns if col not in ['Cover_Type', 'Id']]



#create dataframe of the columns desired from the input data for test and train

X_train = df_train[feature_cols]

X_test = df_test[feature_cols]

#create dataframe of the predicted value to use for building classifier

train_y = df_train['Cover_Type']

#

test_ids = df_test['Id']

#test_y = df_test['Cover_Type']



del df_train

del df_test



print ("creating classifier")

start_time = time.time()

clf = ensemble.RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)

print("--- time to build ensemble.RandomForestClassifier %s seconds ---" % (time.time() - start_time))

#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

#n_jobs=-1 : the number of jobs is set to the number of cores.(runs faster)

#n_estimators = The number of trees in the forest.



print ("fitting from train data")

start_time = time.time()

clf.fit(X_train, train_y)

print("--- time to clf.fit %s seconds ---" % (time.time() - start_time))



print ("predicting from train data")

start_time = time.time()

train_y_predicted = clf.predict(X_train)

print("--- time to clf.predict %s seconds ---" % (time.time() - start_time))

print ("type(train_y_predicted)=", type(train_y_predicted), "len(train_y_predicted)", len(train_y_predicted), "train_y_predicted.shape", train_y_predicted.shape)

print ("type(train_y)=", type(train_y), "train_y.shape=", train_y.shape)

print ("train_y = ", list(train_y[0:20, ]) )

print ("train_y_predicted = ", list(train_y_predicted[0:20, ]) )



print ("predicting from test data")

start_time = time.time()

predicted = clf.predict(X_test)

print ("--- time to clf.predict %s seconds ---" % (time.time() - start_time))

print ("type(predicted)=", type(predicted), "len(predicted)", len(predicted) )

print  ("type(test_ids)=", type(test_ids), "len(test_ids)=", len(test_ids) )





print ("calculating accuracy_score on train data.")

start_time = time.time()

score = accuracy_score(train_y, train_y_predicted)

print ("--- time to calcualte accuracy_score %s seconds ---" % (time.time() - start_time))

print ("type(score)=", type(score), "score=", score )



#for i in range(len(predicted)):



print ("clf.predicting & writing to file")

start_time = time.time()

with open(loc_submission, "w") as outfile:

    outfile.write("Id,Cover_Type\n")

    for e, val in enumerate(list(clf.predict(X_test))):

        outfile.write("%s,%s\n" % (test_ids[e], val))

print("--- time to clf.predict & write to file %s seconds ---" % (time.time() - start_time))


