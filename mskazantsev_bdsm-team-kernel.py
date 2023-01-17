import pandas as pd

import numpy as np



from datetime import timedelta, date



from sklearn.model_selection import train_test_split, GridSearchCV, KFold

from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

from lightgbm import LGBMClassifier
import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('/kaggle/input/car-loan/car_loan_train.csv')

test = pd.read_csv('/kaggle/input/car-loan/car_loan_test.csv')



train.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in train.columns]

test.columns = ["".join (c if c.isalnum() else "_" for c in str(x)) for x in test.columns]
train = train.fillna('other')

test = test.fillna('other')
train = train.replace({'PERFORM_CNS_SCORE_DESCRIPTION':{'C-Very Low Risk':'Low', 'A-Very Low Risk':'Low',

                                                       'B-Very Low Risk':'Low', 'D-Very Low Risk':'Low',

                                                       'F-Low Risk':'Low', 'E-Low Risk':'Low', 'G-Low Risk':'Low',

                                                       'H-Medium Risk': 'Medium', 'I-Medium Risk': 'Medium',

                                                       'J-High Risk':'High', 'K-High Risk':'High','L-Very High Risk':'High',

                                                       'M-Very High Risk':'High','Not Scored: More than 50 active Accounts found':'Not Scored',

                                                       'Not Scored: Only a Guarantor':'Not Scored','Not Scored: Not Enough Info available on the customer':'Not Scored',

                                                        'Not Scored: No Activity seen on the customer (Inactive)':'Not Scored','Not Scored: No Updates available in last 36 months':'Not Scored',

                                                       'Not Scored: Sufficient History Not Available':'Not Scored', 'No Bureau History Available':'Not Scored'

                                                       }})



test = test.replace({'PERFORM_CNS_SCORE_DESCRIPTION':{'C-Very Low Risk':'Low', 'A-Very Low Risk':'Low',

                                                       'B-Very Low Risk':'Low', 'D-Very Low Risk':'Low',

                                                       'F-Low Risk':'Low', 'E-Low Risk':'Low', 'G-Low Risk':'Low',

                                                       'H-Medium Risk': 'Medium', 'I-Medium Risk': 'Medium',

                                                       'J-High Risk':'High', 'K-High Risk':'High','L-Very High Risk':'High',

                                                       'M-Very High Risk':'High','Not Scored: More than 50 active Accounts found':'Not Scored',

                                                       'Not Scored: Only a Guarantor':'Not Scored','Not Scored: Not Enough Info available on the customer':'Not Scored',

                                                        'Not Scored: No Activity seen on the customer (Inactive)':'Not Scored','Not Scored: No Updates available in last 36 months':'Not Scored',

                                                       'Not Scored: Sufficient History Not Available':'Not Scored', 'No Bureau History Available':'Not Scored'

                                                       }})
#Преобразуем даты

train['Date_of_Birth'] = pd.to_datetime(train['Date_of_Birth'])

train['DisbursalDate'] = pd.to_datetime(train['DisbursalDate'])

test['Date_of_Birth'] = pd.to_datetime(test['Date_of_Birth'])

test['DisbursalDate'] = pd.to_datetime(test['DisbursalDate'])

now = pd.Timestamp('now')



future = train['Date_of_Birth'] > date(year=2050,month=1,day=1)

train.loc[future, 'Date_of_Birth'] -= timedelta(days=365.25*100)

future = test['Date_of_Birth'] > date(year=2050,month=1,day=1)

test.loc[future, 'Date_of_Birth'] -= timedelta(days=365.25*100)



train['birth_year'] = train['Date_of_Birth'].apply(lambda ts: ts.year)

train['birth_month'] = train['Date_of_Birth'].apply(lambda ts: ts.month)

train['birth_day'] = train['Date_of_Birth'].apply(lambda ts: ts.day)

train['birth_dayofweek'] = train['Date_of_Birth'].apply(lambda ts: ts.dayofweek)

train['Disbursal_month'] = train['DisbursalDate'].apply(lambda ts: ts.month)

train['Disbursal_day'] = train['DisbursalDate'].apply(lambda ts: ts.day)

train['Disbursal_dayofweek'] = train['DisbursalDate'].apply(lambda ts: ts.dayofweek)

train['Age'] = (now - train['Date_of_Birth']).dt.days

train['DaysSinceDisbursal'] = (now - train['DisbursalDate']).dt.days



test['birth_year'] = test['Date_of_Birth'].apply(lambda ts: ts.year)

test['birth_month'] = test['Date_of_Birth'].apply(lambda ts: ts.month)

test['birth_day'] = test['Date_of_Birth'].apply(lambda ts: ts.day)

test['birth_dayofweek'] = test['Date_of_Birth'].apply(lambda ts: ts.dayofweek)

test['Disbursal_month'] = test['DisbursalDate'].apply(lambda ts: ts.month)

test['Disbursal_day'] = test['DisbursalDate'].apply(lambda ts: ts.day)

test['Disbursal_dayofweek'] = test['DisbursalDate'].apply(lambda ts: ts.dayofweek)

test['Age'] = (now - test['Date_of_Birth']).dt.days

test['DaysSinceDisbursal'] = (now - test['DisbursalDate']).dt.days



train = train.drop(['Date_of_Birth', 'DisbursalDate'], axis=1)

test = test.drop(['Date_of_Birth', 'DisbursalDate'], axis=1)
def get_nmbr(text):

    return int(text[0:text.find('y')]) * 12 + int(text[text.find(' ')+1:text.find('m')])



train['AVERAGE_ACCT_AGE'] = train['AVERAGE_ACCT_AGE'].apply(get_nmbr)

train['CREDIT_HISTORY_LENGTH'] = train['CREDIT_HISTORY_LENGTH'].apply(get_nmbr)



test['AVERAGE_ACCT_AGE'] = test['AVERAGE_ACCT_AGE'].apply(get_nmbr)

test['CREDIT_HISTORY_LENGTH'] = test['CREDIT_HISTORY_LENGTH'].apply(get_nmbr)
categ_cols = ['branch_id', 'manufacturer_id', 'Employment_Type', 'State_ID', 'PERFORM_CNS_SCORE_DESCRIPTION',

             'Aadhar_flag', 'PAN_flag', 'VoterID_flag', 'Driving_flag', 'Passport_flag']
for col in ['supplier_id', 'Current_pincode_ID', 'Employee_code_ID']:

    train.loc[train[col].value_counts()[train[col]].values < 2, col] = -9999

    test.loc[test[col].value_counts()[test[col]].values < 2, col] = -9999
train.head()
train.drop("UniqueID", axis=1, inplace=True)

test.drop("UniqueID", axis=1, inplace=True)
train.dtypes
train = pd.get_dummies(train)

test = pd.get_dummies(test)
def prod_det(x):

    for i in range(x.shape[1]):

        max = np.max(x[:,i])

        x[:,i] = x[:,i] / max

    return abs(np.linalg.det(np.dot(x.T, x)))
train_cols = train.drop(['target', 'MobileNo_Avl_Flag'], axis=1).columns
var_sets = dict()
for num, col in enumerate(train_cols):

    cols_list = [col]

    i = 0

    while i <= 20:

        dict_det = dict()

        for col_to_add in train_cols:

            if col_to_add not in cols_list:

                df_aux = np.array(train[cols_list])

                dict_det[col_to_add] = prod_det(df_aux)

        cols_list.append(sorted(dict_det.items(), key=lambda x: -x[1])[0][0])

        i += 1

    var_sets[num] = cols_list
var_set = dict()



for i in range(len(var_sets.items())):

    var_set[i] = set(var_sets[i])

    
list_of_sets = [i[1] for i in list(var_set.items())]
list_of_sets = np.unique(list_of_sets)
def test_var_sets(train):

    cat_cols = [i for i in train.columns if i in categ_cols]



    hot = OneHotEncoder(handle_unknown='ignore')

    train_tr = pd.DataFrame(hot.fit_transform(train[cat_cols]).toarray(), columns=hot.get_feature_names())

    #test_tr = pd.DataFrame(hot.transform(test[categ_cols]).toarray(), columns=hot.get_feature_names())



    train = pd.concat([train.drop(cat_cols, axis=1), train_tr], axis=1)

    #test = pd.concat([test.drop(categ_cols, axis=1), test_tr], axis=1)



    del(train_tr)

    #, test_tr)



    y = train['target']

    train = train.drop(['target', 'MobileNo_Avl_Flag'], axis=1)

    #test = test.drop(['MobileNo_Avl_Flag'], axis=1)

    

    X_train, X_valid, y_train, y_valid = train_test_split(train, y, test_size=0.3, random_state = 42)

    

    lgbt = LGBMClassifier(max_depth=4, learning_rate=0.03, n_estimators=1000, random_state=42)

    lgbt.fit(X_train, y_train)

    y_pred = lgbt.predict_proba(X_valid)[:, 1]

    

    return roc_auc_score(y_valid, y_pred)
var_sets_auc = dict()
for varset in list_of_sets:

    trainset = train[list(varset) + ['target', 'MobileNo_Avl_Flag']]

    auc = test_var_sets(trainset)

    var_sets_auc[auc] = varset
best_set = sorted(var_sets_auc.items(), key=lambda x: -x[0])[0][1]
best_set
train_best = train[best_set]

test_best = test[best_set]



cat_cols = [i for i in train_best.columns if i in categ_cols]
train_best.nunique()
hot = OneHotEncoder(handle_unknown='ignore')

train_tr = pd.DataFrame(hot.fit_transform(train_best[cat_cols]).toarray(), columns=hot.get_feature_names())

test_tr = pd.DataFrame(hot.transform(test_best[cat_cols]).toarray(), columns=hot.get_feature_names())



train_best = pd.concat([train_best.drop(cat_cols, axis=1), train_tr], axis=1)

test_best = pd.concat([test_best.drop(cat_cols, axis=1), test_tr], axis=1)



del(train_tr, test_tr)
y = train['target']

#train = train.drop(['target', 'MobileNo_Avl_Flag'], axis=1)

#test = test.drop(['MobileNo_Avl_Flag'], axis=1)
X_train, X_valid, y_train, y_valid = train_test_split(train_best, y, test_size=0.3, random_state = 42)
#%%time

lgbm_params = {'max_depth': [3,5,7],

              'learning_rate':[0.05, 0.01, 0.03],

              'n_estimators':[1000, 1200, 1400]}

lgbt = LGBMClassifier(random_state=42)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

lgbm_grid = GridSearchCV(lgbt, lgbm_params, cv=cv, scoring='roc_auc', verbose=1, n_jobs=-1)

lgbm_grid.fit(X_train, y_train)
lgbm_grid.best_params_
lgbt = LGBMClassifier(max_depth=3, learning_rate=0.05, n_estimators=1000, random_state=42)

lgbt.fit(X_train, y_train)

y_pred = lgbt.predict_proba(X_valid)[:, 1]

y_pred = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred))

roc_auc_score(y_valid, y_pred)
y = train['target']

train = train.drop(['target', 'MobileNo_Avl_Flag'], axis=1)

test = test.drop(['MobileNo_Avl_Flag'], axis=1)

X_train, X_valid, y_train, y_valid = train_test_split(train, y, test_size=0.3, random_state = 42)



lgbm_params = {'max_depth': [4],

              'learning_rate':np.arange(0.03, 0.06, 0.01),

              'n_estimators':[500, 1000, 1500, 2000]}

lgbt = LGBMClassifier(random_state=42)

cv = KFold(n_splits=5, shuffle=True, random_state=42)

lgbm_grid = GridSearchCV(lgbt, lgbm_params, cv=cv, scoring='roc_auc')

lgbm_grid.fit(X_train, y_train)
lgbt = LGBMClassifier(max_depth=4, learning_rate=0.03, n_estimators=1000)

lgbt.fit(X_train, y_train)

y_pred = lgbt.predict_proba(X_valid)[:, 1]

roc_auc_score(y_valid, y_pred)
lgbt = LGBMClassifier(max_depth=4, learning_rate=0.03, n_estimators=1000)

lgbt.fit(train, y)
answer_1 = lgbt.predict_proba(test)[:, 1]

answer_1
answer1 = pd.DataFrame(columns=['ID', 'Predicted'])

answer1['ID'] = test.index

answer1['Predicted'] = answer_1
answer1.to_csv('answer228.csv', index=None)