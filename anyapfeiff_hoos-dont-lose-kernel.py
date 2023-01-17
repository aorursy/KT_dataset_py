import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

print(os.listdir("../input"))
import seaborn as sns

import matplotlib

from matplotlib import pyplot as plt
bank_test = pd.read_csv('../input/bank-test.csv')

bank_train = pd.read_csv('../input/bank-train.csv')

bank_train_num = bank_train[['id', 'age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.conf.idx', 'euribor3m', 'nr.employed']]

bank_test_num = bank_test[['id', 'age', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.conf.idx', 'euribor3m', 'nr.employed']]

results = bank_train[['y']]
bank_train.describe()
bank_test.describe()
bank_test.describe(include=['O'])
from sklearn import linear_model

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import f1_score

from sklearn.linear_model import Lasso

from sklearn.linear_model import Ridge
lr = LogisticRegression()

lr_fit = lr.fit(bank_train_num, results)

lr_predict = lr.predict(bank_train_num)



#generate predictions on the test set from the logistic regression

lr_predict_test = lr.predict(bank_test_num)



print(classification_report(results, lr_predict))
lasso = Lasso(alpha=5)

lasso_fit = lasso.fit(bank_train_num, results)

lasso_predict = lasso.predict(bank_train_num)



for i in range(len(lasso_predict)):

    if lasso_predict[i] < 0.7:

        lasso_predict[i] = 0

    else:

        lasso_predict[i] = 1



print(classification_report(results, lasso_predict))
ridge = Ridge(alpha=5)

ridge_fit = ridge.fit(bank_train_num, results)

ridge_predict = ridge.predict(bank_train_num)



for i in range(len(ridge_predict)):

    if ridge_predict[i] < 0.7:

        ridge_predict[i] = 0

    else:

        ridge_predict[i] = 1



print(classification_report(results, ridge_predict))
# try a random forest

from sklearn.ensemble import RandomForestClassifier

rand_for = RandomForestClassifier(n_estimators = 500, random_state = 40)

rand_for_fit = rand_for.fit(bank_train_num, results)
#predict values based on the random forest

rand_for_predict_test = rand_for.predict(bank_test_num)
#run feature importance

feat_imp = pd.DataFrame(rand_for.feature_importances_, index=bank_train_num.columns)

print(feat_imp)



#edit the features used in the random forest based on these results

bank_train_num2 = bank_train[['age', 'campaign', 'pdays', 'euribor3m', 'nr.employed']]

bank_test_num2 = bank_test[['age', 'campaign', 'pdays', 'euribor3m', 'nr.employed']]



rand_for_fit2 = rand_for.fit(bank_train_num2, results)

rand_for_predict2 = rand_for.predict(bank_test_num2)
bank_train_d = pd.get_dummies(bank_train)

bank_test_d = pd.get_dummies(bank_test)



#bank_train_d = bank_train_d.drop('poutcome',axis=1)

#bank_test_d = bank_test_d.drop('poutcome',axis=1)

bank_train_d = bank_train_d.drop('duration',axis=1)

bank_test_d = bank_test_d.drop('duration',axis=1)

bank_train_d = bank_train_d.drop('default_yes', axis=1)





bank_train_d = bank_train_d.drop('y',axis=1)



print(bank_test.shape)

print(bank_test_d.shape)



#print(bank_train_d.shape)

#print(bank_test_d.shape)



lr_fit_dummies = lr.fit(bank_train_d, results)

lr_predict_dummies = lr.predict(bank_train_d)



#print(classification_report(results, lr_predict_dummies))



lr_predict_d = lr.predict(bank_test_d)
from sklearn.linear_model import RidgeClassifier

ridge2 = RidgeClassifier(alpha=5)

ridge_fit2 = ridge2.fit(bank_train_d, results)

ridge_predict2 = ridge2.predict(bank_train_d)



for i in range(len(ridge_predict2)):

    if ridge_predict2[i] < 0.7:

        ridge_predict2[i] = 0

    else:

        ridge_predict2[i] = 1



print(classification_report(results, ridge_predict2))



ridge_predict_test2 = ridge2.predict(bank_test_d)

print(bank_test_d.shape)
rand_for_fit_d = rand_for.fit(bank_train_d, results)

rand_for_predict_d = rand_for.predict(bank_test_d)



#feat_imp = pd.DataFrame(rand_for.feature_importances_,index = bank_train_d.columns,columns=['importance']).sort_values('importance',ascending=False)

#print(feat_imp)



print(bank_test_d.shape)
bank_train_d2 = bank_train_d[['id', 'age', 'campaign', 'euribor3m', 'nr.employed', 'cons.conf.idx', 'poutcome_success']]

bank_test_d2 = bank_test_d[['id', 'age', 'campaign', 'euribor3m', 'nr.employed', 'cons.conf.idx', 'poutcome_success']]



rand_for_fit_d2 = rand_for.fit(bank_train_d2, results)

rand_for_predict_d2 = rand_for.predict(bank_test_d2)



bank_test.dtypes

print(bank_test_d.shape)
print(bank_test_d.shape)

submission2 = pd.concat([bank_test_d.id, pd.Series(ridge_predict_test2)], axis = 1)

submission2.columns = ['id', 'Predicted']

submission2.to_csv('submission.csv', index=False)

#submission['id'].astype('int64')

print(len(submission2['id']))

print(len(submission2['Predicted']))