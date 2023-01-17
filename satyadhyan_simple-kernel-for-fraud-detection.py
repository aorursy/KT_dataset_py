import pandas as pd

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn import svm

import numpy as np
%%time



dft = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')

dft2 = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')

dft1 = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')
# selecteding required columns 

sel_cols = ['TransactionID', 'TransactionAmt', 'ProductCD', 'card4', 'card6']

sel_col1 = [col for col in dft if col.startswith('V')] ## Selecting columns starting with 'V'

sel_cols = sel_cols + sel_col1
X_cols = dft[sel_cols].columns

Y_cols = dft2[sel_cols].columns

numericCols = dft._get_numeric_data().columns

numericCols1 = dft2._get_numeric_data().columns

catCols = list(set(X_cols) - set(numericCols))

catCols1 = list(set(Y_cols) - set(numericCols1))

dft[catCols] = dft[catCols].replace({ np.nan:'no card'})

dft2[catCols1] = dft2[catCols1].replace({ np.nan:'no card'})

dft[numericCols] = dft[numericCols].replace({ np.nan:-1})

dft2[numericCols1] = dft2[numericCols1].replace({ np.nan:-1})
X = dft[sel_cols]

y = dft['isFraud'].astype(float)

X['card4'].fillna("no card", inplace=True)

X['card6'].fillna("no card", inplace=True)

pcd = {'C':1, 'H':2, 'R':3, 'S':4, 'W':5}

cards = {'no card':0, 'credit':1, 'debit':2, 'debit or credit':3, 'charge card':4}

ctype = {'no card':0, 'american express':1, 'discover':2, 'mastercard':3, 'visa':4}

X.ProductCD = [pcd[item] for item in X.ProductCD]

X.card4 = [ctype[item] for item in X.card4]

X.card6 = [cards[item] for item in X.card6]

X['ProductCD'] = X.ProductCD.astype(float)

X['card4'] = X.card4.astype(float)

X['card6'] = X.card6.astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state = 0)
model = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight={0:0.85})

model.fit(X, y)

pred = model.predict(X_test)

print(pred)
from sklearn.metrics import classification_report

print(classification_report(y_test, pred))

print('Logistic Score:', model.score(X_test, y_test))
X = dft2[sel_cols]

X['card4'].fillna("no card", inplace=True)

X['card6'].fillna("no card", inplace=True)

pcd = {'C':1, 'H':2, 'R':3, 'S':4, 'W':5}

cards = {'no card':0, 'credit':1, 'debit':2, 'debit or credit':3, 'charge card':4}

ctype = {'no card':0, 'american express':1, 'discover':2, 'mastercard':3, 'visa':4}

X.ProductCD = [pcd[item] for item in X.ProductCD]

X.card4 = [ctype[item] for item in X.card4]

X.card6 = [cards[item] for item in X.card6]

X['ProductCD'] = X.ProductCD.astype(float)

X['card4'] = X.card4.astype(float)

X['card6'] = X.card6.astype(float)
p_op = pd.DataFrame()

p_op['TransactionID'] = dft1['TransactionID']

pred = model.predict_proba(X)[:,1]

print(pred)

#prob = pd.DataFrame([1 - item[0] for item in pred])

#print(prob)

p_op['isFraud'] =  pred

#p_op.loc[p_op['isFraud'] > 0.5, 'isFraud'] = 1.0

#p_op.loc[p_op['isFraud'] <= 0.5, 'isFraud'] = 0.0

p_op['isFraud']

print(p_op)

p_op.to_csv('Sample_Submission_091919.csv')