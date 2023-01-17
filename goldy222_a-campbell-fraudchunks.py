import pandas as pd

import numpy as np

from sklearn import naive_bayes

from sklearn.model_selection import GridSearchCV

from sklearn.pipeline import Pipeline

from sklearn.metrics import roc_curve

from sklearn.model_selection import train_test_split

from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt





# I print progress through the script so I can watch the RAM, due to the large dataset



print ("Read test data")

df_concat_dummies = pd.get_dummies(pd.read_csv('../input/concatcsvs/test_df_concat.csv'), sparse=True).fillna(0).astype(float)

df_concat_dummies.drop('TransactionID', axis=1, inplace=True)

print (df_concat_dummies.shape)



# all_cols will be used later to ensure training and test data have the same feature columns

print("get all_cols")

all_cols = df_concat_dummies.columns.values

print (len(all_cols))



print("start reading and fitting in chunks")

train_df_concat2 = pd.read_csv('../input/concatcsvs/train_df_concat.csv', chunksize=50000)



model = naive_bayes.BernoulliNB(alpha=1000)



for chunk in train_df_concat2:

                 

                 # Get our features and target

                 dummies_df = pd.get_dummies(chunk, sparse=True).fillna(0).astype(float)

                 y = dummies_df['isFraud']

                 dummies_df.drop('TransactionID', axis=1, inplace=True)

                 dummies_df.drop('isFraud', axis=1, inplace=True)

                 

                 # Ensure training and test datasets have the same columns, in the same order

                 X_list = dummies_df.columns.values

                 new_cols = (list(set(all_cols) - set(X_list)))

                 X_ready = (pd.concat([dummies_df, pd.DataFrame(columns=new_cols)], axis=1).fillna(0).astype(float))[df_concat_dummies.columns]

                

                 # Need to define classes since we're using partial_fit

                 classes = np.unique(y)

                 

                 print ("shape of data to fit")

                 print (X_ready.shape)

                 

#                  # Use GridSearchCV to determine and print the best BernoulliNB alpha, best alpha was 1000

#                  gridsearch_steps = [('BernoulliNB', naive_bayes.BernoulliNB())]

#                  gridsearch_pipeline = Pipeline(gridsearch_steps)

#                  parameters = {'BernoulliNB__alpha':[.01,.1,1,10,100,1000,10000]}

#                  gm_cv = GridSearchCV(gridsearch_pipeline,parameters)

#                  gm_cv.fit(X_ready,y)

#                  print("Tuned BernoulliNB Alpha: {}".format(gm_cv.best_params_))

                 

                 # Start fitting

                 model.partial_fit(X_ready,y, classes=classes)

                 print("partial fit done")



print("all fitting done")



# Check the performance of our fit using the last training chunk, before getting our finalized predictions

X_train, X_test, y_train, y_test = train_test_split(X_ready,y,test_size=0.3,random_state=42)



# Compute predicted probabilities

y_pred_prob = model.predict_proba(X_test)[:,1]



# Generate ROC curve values

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)



# Plot ROC curve

plt.plot([0, 1], [0, 1], 'k--')

plt.plot(fpr, tpr)

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC Curve')

plt.show()



# Compute and print AUC score

print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))



# Start predicting and populating our predictions dataframe that will be output to CSV, 50,000 rows at a time

counter1 = 0

counter2 = 50000

submit_pred_df = pd.DataFrame(columns=['isFraud'])



for i in range(11):

    print ("start predict_proba")

    pred = model.predict_proba(df_concat_dummies.iloc[counter1:counter2,:])

    predicted_prob_true = pred[:,1]

    submit_pred_df = submit_pred_df.append(pd.DataFrame({'isFraud':predicted_prob_true}))

    print (submit_pred_df.shape)

    print (submit_pred_df.tail)

    counter1 = counter1 + 50000

    counter2 = counter2 + 50000



print ("start to_CSV for submit_final")

submit_pred_df.to_csv('../working/submit_final.csv')