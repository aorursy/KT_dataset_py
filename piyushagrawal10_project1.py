import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import os

print(os.listdir("../input"))

train = pd.read_csv('../input/train_1.csv')

test = pd.read_csv('../input/test_1.csv')

t2=train

rep={'Gender' : {'Male' : 1 , 'Female' : 0},'Married' : {'Yes' : 1 , 'No' : 0},'Education' : {'Graduate' : 1 , 'Not Graduate' : 0},'Self_Employed' : {'Yes' : 1 , 'No' : 0},'Dependents' : {'0' : 0 , '1' : 1 , '2' : 2 , '3+' : 3},'Property_Area' : {'Rural' : 0 , 'Semiurban' : 1 , 'Urban' : 2}, 'Loan_Status' : {'Y' : 1 , 'N' : 0}}

t2=t2.replace(rep)

for column in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Loan_Status','Property_Area','Credit_History'] : t2[column].fillna(t2[column].mode()[0], inplace=True)

t2['TotalIncome'] = t2['ApplicantIncome'] + t2['CoapplicantIncome']

t2['TotalIncomeLog'] = np.log(t2['TotalIncome'])

t2['LoanAmountLog'] = np.log(t2['LoanAmount'])

t2['LoanAmountLog'].fillna(t2['LoanAmountLog'].mean(),inplace=True)

e2=test

rep={'Gender' : {'Male' : 1 , 'Female' : 0},'Married' : {'Yes' : 1 , 'No' : 0},'Education' : {'Graduate' : 1 , 'Not Graduate' : 0},'Self_Employed' : {'Yes' : 1 , 'No' : 0},'Dependents' : {'0' : 0 , '1' : 1 , '2' : 2 , '3+' : 3},'Property_Area' : {'Rural' : 0 , 'Semiurban' : 1 , 'Urban' : 2}}

e2=e2.replace(rep)

for column in ['Gender', 'Married',  'Dependents', 'Education', 'Self_Employed','Property_Area','Credit_History'] : e2[column].fillna(e2[column].mode()[0], inplace=True)

e2['TotalIncome'] = e2['ApplicantIncome'] + e2['CoapplicantIncome']

e2['TotalIncomeLog'] = np.log(e2['TotalIncome'])

e2['LoanAmountLog'] = np.log(e2['LoanAmount'])

e2['LoanAmountLog'].fillna(e2['LoanAmountLog'].mean(),inplace=True)

param = ['Dependents', 'TotalIncomeLog', 'LoanAmountLog','Credit_History','Education', 'Self_Employed']

tr_x = t2[[ 'Gender', 'Married', 'Dependents', 'Education',

       'Self_Employed', 'TotalIncomeLog', 'LoanAmountLog',

        'Credit_History', 'Property_Area']]

tr_y = t2['Loan_Status']

te_x = e2[[ 'Gender', 'Married', 'Dependents', 'Education',

       'Self_Employed', 'TotalIncomeLog', 'LoanAmountLog',

        'Credit_History', 'Property_Area']]





from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import VotingClassifier

import statistics

from sklearn.metrics import accuracy_score

from sklearn.model_selection import RepeatedKFold

rkf = RepeatedKFold(n_splits=4, n_repeats=10, random_state=None)

#model = RandomForestClassifier(n_estimators=200,criterion='entropy',max_depth=4,max_features='log2',min_samples_leaf=10,

#                              min_weight_fraction_leaf=0.05)

#gnb = 

model = GaussianNB()

# rfc= 

model = RandomForestClassifier(n_estimators=100,criterion='entropy',max_depth=4,max_features='log2',min_samples_leaf=10,

                             min_weight_fraction_leaf=0.06)

# clf = rfc 

# #AdaBoostClassifier(n_estimators=100,base_estimator=rfc,learning_rate=2)

# model = VotingClassifier(estimators=[('abc', clf), ('gnb',gnb)], voting='hard')

# model.fit(tr_x,tr_y)

# pre=model.predict(te_x)

# acc1=[]

# # X is the feature set and y is the target

# for i in range(0,5) :

#     acc1=[]

#     for train_index, test_index in rkf.split(t2):

#          X_train, X_test = tr_x.loc[train_index], tr_x.loc[test_index]

#          y_train, y_test = tr_y.loc[train_index], tr_y.loc[test_index]

#          model.fit(X_train,y_train)

#          pred = model.predict(X_test)

#          acc = accuracy_score(y_test,pred)

#          #print(acc)

#          acc1.append(acc)

#     print('Average = ' ,statistics.mean(acc1))



model.fit(tr_x,tr_y)

pre=model.predict(te_x)



id1=e2['Loan_ID']

l = range(0,id1.count())

id1.index = l

pre = pd.Series(pre)

final = pd.concat([id1,pre],axis=1)

final.columns = ['Loan_ID','Loan_Status']

final['Loan_Status']=final['Loan_Status'].astype('str')

eplace = {'Loan_Status' : {'1' : 'Y','0' : 'N'}}

final=final.replace(eplace)

final.to_csv('P1.csv', encoding='utf-8', index=False)