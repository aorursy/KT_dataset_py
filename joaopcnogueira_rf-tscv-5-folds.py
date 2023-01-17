import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import TimeSeriesSplit, KFold

from sklearn.model_selection import cross_validate

from sklearn.model_selection import GridSearchCV

import time

import warnings



warnings.simplefilter('ignore')

pd.set_option('display.max_columns', None)
train = pd.read_csv('../input/train_transaction.csv')

traini = pd.read_csv('../input/train_identity.csv')

test = pd.read_csv('../input/test_transaction.csv')

testi = pd.read_csv('../input/test_identity.csv')
train['nulls1'] = train.isna().sum(axis=1)

test['nulls1'] = test.isna().sum(axis=1)



traini['nulls2'] = traini.isna().sum(axis=1)

testi['nulls2'] = testi.isna().sum(axis=1)
emails = {'gmail': 'google', 'att.net': 'att', 'twc.com': 'spectrum', 'scranton.edu': 'other', 'optonline.net': 'other', 'hotmail.co.uk': 'microsoft', 'comcast.net': 'other', 'yahoo.com.mx': 'yahoo', 'yahoo.fr': 'yahoo', 'yahoo.es': 'yahoo', 'charter.net': 'spectrum', 'live.com': 'microsoft', 'aim.com': 'aol', 'hotmail.de': 'microsoft', 'centurylink.net': 'centurylink', 'gmail.com': 'google', 'me.com': 'apple', 'earthlink.net': 'other', 'gmx.de': 'other', 'web.de': 'other', 'cfl.rr.com': 'other', 'hotmail.com': 'microsoft', 'protonmail.com': 'other', 'hotmail.fr': 'microsoft', 'windstream.net': 'other', 'outlook.es': 'microsoft', 'yahoo.co.jp': 'yahoo', 'yahoo.de': 'yahoo', 'servicios-ta.com': 'other', 'netzero.net': 'other', 'suddenlink.net': 'other', 'roadrunner.com': 'other', 'sc.rr.com': 'other', 'live.fr': 'microsoft', 'verizon.net': 'yahoo', 'msn.com': 'microsoft', 'q.com': 'centurylink', 'prodigy.net.mx': 'att', 'frontier.com': 'yahoo', 'anonymous.com': 'other', 'rocketmail.com': 'yahoo', 'sbcglobal.net': 'att', 'frontiernet.net': 'yahoo', 'ymail.com': 'yahoo', 'outlook.com': 'microsoft', 'mail.com': 'other', 'bellsouth.net': 'other', 'embarqmail.com': 'centurylink', 'cableone.net': 'other', 'hotmail.es': 'microsoft', 'mac.com': 'apple', 'yahoo.co.uk': 'yahoo', 'netzero.com': 'other', 'yahoo.com': 'yahoo', 'live.com.mx': 'microsoft', 'ptd.net': 'other', 'cox.net': 'other', 'aol.com': 'aol', 'juno.com': 'other', 'icloud.com': 'apple'}

us_emails = ['gmail', 'net', 'edu']



for var in ['P_emaildomain', 'R_emaildomain']:

    train[var + '_bin'] = train[var].map(emails)

    test[var + '_bin'] = test[var].map(emails)

    

    train[var + '_suffix'] = train[var].map(lambda x: str(x).split('.')[-1])

    test[var + '_suffix'] = test[var].map(lambda x: str(x).split('.')[-1])

    

    train[var + '_suffix'] = train[var + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

    test[var + '_suffix'] = test[var + '_suffix'].map(lambda x: x if str(x) not in us_emails else 'us')

    

    

train['email_domain_comp'] = (train['P_emaildomain'].values == train['R_emaildomain'].values).astype(int)

test['email_domain_comp'] = (test['P_emaildomain'].values == test['R_emaildomain'].values).astype(int)

    

train['email_domain_suffix_bin'] = (train['P_emaildomain_bin'].values == train['R_emaildomain_bin'].values).astype(int)

test['email_domain_suffix_bin'] = (test['P_emaildomain_bin'].values == test['R_emaildomain_bin'].values).astype(int)



train['email_domain_suffix_comp'] = (train['P_emaildomain_suffix'].values == train['R_emaildomain_suffix'].values).astype(int)

test['email_domain_suffix_comp'] = (test['P_emaildomain_suffix'].values == test['R_emaildomain_suffix'].values).astype(int)
train = pd.merge(train, traini, how='left', on='TransactionID'); 

del traini

test = pd.merge(test, testi, how='left', on='TransactionID'); 

del testi
y_train = train['isFraud'].copy()



# Drop target, fill in NaNs

X_train = train.drop('isFraud', axis=1)

X_test = test.copy()

X_train = X_train.fillna(-999)

X_test = X_test.fillna(-999)



del train, test
drop_cols = ['TransactionID', 'TransactionDT', 'V300', 'V309', 'V111', 'C3', 'V124', 'V106', 'V125', 'V315', 'V134', 'V102', 'V123', 'V316', 'V113', 'V136', 'V305', 'V110', 'V299', 'V289', 'V286', 'V318', 'V103', 'V304', 'V116', 'V298', 'V284', 'V293', 'V137', 'V295', 'V301', 'V104', 'V311', 'V115', 'V109', 'V119', 'V321', 'V114', 'V133', 'V122', 'V319', 'V105', 'V112', 'V118', 'V117', 'V121', 'V108', 'V135', 'V320', 'V303', 'V297', 'V120']

X_train = X_train.drop(columns=drop_cols, axis=1)

X_test = X_test.drop(columns=drop_cols, axis=1)
# Label Encoding

from sklearn.preprocessing import LabelEncoder



for var in X_train.select_dtypes(include='object').columns:    

    encoder = LabelEncoder()

    encoder.fit(list(X_train[var].values) + list(X_test[var].values))

    X_train[var] = encoder.transform(list(X_train[var].values))

    X_test[var] = encoder.transform(list(X_test[var].values))   
model = RandomForestClassifier(n_estimators=500, bootstrap=False, criterion='entropy',

                               n_jobs=-1, random_state=42, min_samples_leaf=10, max_features=0.2)



tscv = TimeSeriesSplit(n_splits=5)

cv_results = cross_validate(model, X_train, y_train, scoring='roc_auc', cv=tscv)
cv_results['test_score'].mean()
model.fit(X_train, y_train)

preds = model.predict_proba(X_test)[:,1]
sub = pd.read_csv("../input/sample_submission.csv")

sub['isFraud'] = preds

sub.to_csv('rf_no_bootstrap_500.csv', index=False)