import pandas as pd

import numpy as np

from sklearn import naive_bayes

from sklearn import linear_model

from sklearn import svm

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.calibration import CalibratedClassifierCV



# I print progress so I can see where RAM is due to large datasets

# Code used to clean the data is shown towards the bottom



# I undersampled so I wouldn't run out of memory. This is Binary Classification with extremely unbalanced data

print("start reading training data")

train_df_isFraud1 = pd.read_csv('../input/df-replace/train_replace.csv')[pd.read_csv('../input/df-replace/train_replace.csv').isFraud == 1]

train_df_isFraud0 = pd.read_csv('../input/df-replace/train_replace.csv')[pd.read_csv('../input/df-replace/train_replace.csv').isFraud == 0].iloc[:200000,:]

train_df = pd.get_dummies(pd.concat([train_df_isFraud0,train_df_isFraud1],axis=0).fillna(0), sparse=True).astype(float)



# Shuffle rows and drop unnecessary column

train_df = train_df.sample(frac=1).reset_index(drop=True)

train_df.drop(train_df.columns[0], axis=1, inplace=True)



print ("Read test data")

test_df = pd.get_dummies(pd.read_csv('../input/df-replace/test_replace.csv').fillna(0), sparse=True).astype(float)

test_df.drop(test_df.columns[0], axis=1, inplace=True)

test_df.drop('TransactionID', axis=1, inplace=True)

print (test_df.shape)



# all_cols will be used later to ensure training and test data have the same feature columns

print("get all_cols")

all_cols = test_df.columns.values

print (len(all_cols))



# Get our target

y = train_df['isFraud']



# Ensure training and test datasets have the same columns, in the same order, then get our finalized feature data

train_df.drop('TransactionID', axis=1, inplace=True)

train_df.drop('isFraud', axis=1, inplace=True)

X_list = train_df.columns.values

new_cols = (list(set(all_cols) - set(X_list)))

X_ready = (pd.concat([train_df, pd.DataFrame(columns=new_cols)], axis=1).fillna(0).astype(float))[test_df.columns]

print ("shape of data to fit")

print (X_ready.shape)



# # Use GridSearchCV to determine the best BernoulliNB alpha

# gridsearch_steps = [('BernoulliNB', naive_bayes.BernoulliNB())]

# gridsearch_pipeline = Pipeline(gridsearch_steps)

# parameters = {'BernoulliNB__alpha':[.01,.1,1,10,100,1000,10000,100000]}

# gm_cv = GridSearchCV(gridsearch_pipeline,parameters)

# gm_cv.fit(X_ready,y)

# print("Tuned BernoulliNB Alpha: {}".format(gm_cv.best_params_))



model = naive_bayes.BernoulliNB(alpha=1000)



print ("start fitting")

model.fit(X_ready,y)



# Start predicting and populating our predictions dataframe that will be output to CSV, 50,000 rows at a time

counter1 = 0

counter2 = 50000

submit_pred_df = pd.DataFrame(columns=['isFraud'])

for i in range(11):

    print ("start predict_proba")

    pred = model.predict_proba(test_df.iloc[counter1:counter2,:])

    # pred = clf.predict_proba(df_concat_dummies.iloc[counter1:counter2,:])

    predicted_prob_true = pred[:,1]

    submit_pred_df = submit_pred_df.append(pd.DataFrame({'isFraud':predicted_prob_true}))

    print (submit_pred_df.tail)

    counter1 = counter1 + 50000

    counter2 = counter2 + 50000

submit_pred_df.to_csv('../working/submit_final.csv')





# The below code was used to perform some cleanup on the starting data





# Identify columns with high cardinality, one column had over 1700 unique text values which would greatly increase the training set after get_dummies

# columns = (train_df.columns.get_values())

# for i in columns:

#     print (i)

#     print (train_df[i].nunique())

#     print (train_df[i].unique())



# Replace values in the high cardinality columns with number of occurrances of the value, so get_dummies doesn't create too many columns

# Define a function to replace column values with the # of occurances, for a list of columns in train_df. To cut down on columns after get_dummies

# Columns = [enter column names]

# def ColNumOccur(ColumnList):

#     for column in ColumnList:

#         for i in train_df[column].unique():

#             train_df[column] = train_df[column].replace({i:train_df[train_df[column] == i].shape[0]})

# ColNumOccur(Columns)



# Output our cleaned up dataframes to CSV after high cardinality columns have values replaced. These are the CSVs used in this code

# train_df.to_csv('../working/train_replace.csv')

# test_df.to_csv('../working/test_replace.csv')