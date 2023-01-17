import pandas as pd

import numpy as np

from sklearn import naive_bayes

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline



# I print progress through the script so I can watch the RAM, due to the large dataset



print ("Reading test data")

df_concat_dummies = pd.get_dummies(pd.read_csv('../input/concatcsvs/test_df_concat.csv'), sparse=True).fillna(0).astype(float)

df_concat_dummies.drop('TransactionID', axis=1, inplace=True)

print (df_concat_dummies.shape)



# all_cols will be used later to ensure training and test data have the same feature columns

print("get all_cols")

all_cols = df_concat_dummies.columns.values

print (len(all_cols))



# I undersampled. The training set went from almost 600,000 rows down to 40,000 with similar performance. This is Binary Classification with extremely unbalanced data

print("start reading training data")

train_df_isFraud1 = pd.read_csv('../input/concatcsvs/train_df_concat.csv')[pd.read_csv('../input/concatcsvs/train_df_concat.csv').isFraud == 1]

train_df_isFraud0 = pd.read_csv('../input/concatcsvs/train_df_concat.csv')[pd.read_csv('../input/concatcsvs/train_df_concat.csv').isFraud == 0].iloc[:20663,:]

train_df = pd.concat([train_df_isFraud0,train_df_isFraud1],axis=0)



# Shuffle rows

train_df = train_df.sample(frac=1).reset_index(drop=True)



# Get our features and target

dummies_df = pd.get_dummies(train_df, sparse=True).fillna(0).astype(float)

y = dummies_df['isFraud']

dummies_df.drop('TransactionID', axis=1, inplace=True)

dummies_df.drop('isFraud', axis=1, inplace=True)



# Ensure training and test datasets have the same columns, in the same order

X_list = dummies_df.columns.values

new_cols = (list(set(all_cols) - set(X_list)))

X_ready = (pd.concat([dummies_df, pd.DataFrame(columns=new_cols)], axis=1).fillna(0).astype(float))[df_concat_dummies.columns]

print ("shape of data to fit")

print (X_ready.shape)



# Use GridSearchCV to determine the best BernoulliNB alpha

gridsearch_steps = [('BernoulliNB', naive_bayes.BernoulliNB())]

gridsearch_pipeline = Pipeline(gridsearch_steps)

parameters = {'BernoulliNB__alpha':[.01,.1,1,10,100,1000,10000,100000]}

gm_cv = GridSearchCV(gridsearch_pipeline,parameters)

gm_cv.fit(X_ready,y)

print("Tuned BernoulliNB Alpha: {}".format(gm_cv.best_params_))



model = naive_bayes.BernoulliNB(alpha=10000)



print ("start fitting")

model.fit(X_ready,y)

print("fitting done")



# Start predicting and populating our predictions dataframe that will be output to CSV, 50,000 rows at a time

counter1 = 0

counter2 = 50000

submit_pred_df = pd.DataFrame(columns=['isFraud'])

for i in range(11):

    print ("start predict_proba")

    pred = model.predict_proba(df_concat_dummies.iloc[counter1:counter2,:])

    predicted_prob_true = pred[:,1]

    submit_pred_df = submit_pred_df.append(pd.DataFrame({'isFraud':predicted_prob_true}))

    counter1 = counter1 + 50000

    counter2 = counter2 + 50000

submit_pred_df.to_csv('../working/submit_final.csv')