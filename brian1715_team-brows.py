import pandas as pd

import numpy as np

import random as rnd



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import  train_test_split

from sklearn.metrics import classification_report
train_df=pd.read_csv(r"../input/bank-train.csv")

test_df=pd.read_csv(r"../input/bank-test.csv")

combine = [train_df, test_df]
print(train_df.columns.values)
train_df.head()
train_df.info()

print('_'*40)

test_df.info()
train_df.describe()
train = train_df.drop(['id', 'duration'], axis=1)

test = test_df.drop(['id', 'duration'], axis=1)

combine = [train_df, test_df]
train = pd.get_dummies(train)

test = pd.get_dummies(test)

combine = [train, test]
train_corr = train.corr()
train_corr.sort_values('y').head(40)
y=train["y"]
X=train[["nr.employed","pdays","euribor3m","emp.var.rate","poutcome_nonexistent","contact_telephone","cons.price.idx","month_may","default_unknown","job_blue-collar","campaign","education_basic.9y","marital_married","job_services","month_jul","education_basic.6y","day_of_week_mon","job_entrepreneur","month_nov","housing_no","marital_divorced","education_basic.4y","education_high.school","job_housemaid","month_aug","day_of_week_fri","month_jun","job_self-employed","job_unknown","loan_yes","job_technician","job_management","education_professional.course","loan_no","marital_unknown","loan_unknown","housing_unknown","day_of_week_wed","education_illiterate"]]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)  
svc = SVC()

svc.fit(X_train, y_train)

y_pred = svc.predict(X_test)



from sklearn.metrics import f1_score

print("F1 score SVC Model: ", f1_score(y_test, y_pred, average='weighted'))
Xfor_testset = test[["nr.employed","pdays","euribor3m","emp.var.rate","poutcome_nonexistent","contact_telephone","cons.price.idx","month_may","default_unknown","job_blue-collar","campaign","education_basic.9y","marital_married","job_services","month_jul","education_basic.6y","day_of_week_mon","job_entrepreneur","month_nov","housing_no","marital_divorced","education_basic.4y","education_high.school","job_housemaid","month_aug","day_of_week_fri","month_jun","job_self-employed","job_unknown","loan_yes","job_technician","job_management","education_professional.course","loan_no","marital_unknown","loan_unknown","housing_unknown","day_of_week_wed","education_illiterate"]]
predictions = svc.predict(Xfor_testset)

submission = pd.concat([test_df.id, pd.Series(predictions)], axis = 1)

submission.columns = ['id', 'Predicted']

submission.to_csv('submission.csv', index=False)

logreg = LogisticRegression()

logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)



print("F1 score LogReg Model: ", f1_score(y_test, y_pred, average='weighted'))
predictions = logreg.predict(Xfor_testset)

submission = pd.concat([test_df.id, pd.Series(predictions)], axis = 1)

submission.columns = ['id', 'Predicted']

submission.to_csv('submission.csv', index=False)

 
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(X_train, y_train)

y_pred = random_forest.predict(X_test)



print("F1 score Random Forest Model: ", f1_score(y_test, y_pred, average='weighted'))
predictions = random_forest.predict(Xfor_testset)

submission = pd.concat([test_df.id, pd.Series(predictions)], axis = 1)

submission.columns = ['id', 'Predicted']

submission.to_csv('submission.csv', index=False)

linear_svc = LinearSVC()

linear_svc.fit(X_train, y_train)

y_pred = linear_svc.predict(X_test)



print("Linear SVC Model:")

print("F1 score Linear SVC Model: ", f1_score(y_test, y_pred, average='weighted'))
predictions = linear_svc.predict(Xfor_testset)

submission = pd.concat([test_df.id, pd.Series(predictions)], axis = 1)

submission.columns = ['id', 'Predicted']

submission.to_csv('submission.csv', index=False) 