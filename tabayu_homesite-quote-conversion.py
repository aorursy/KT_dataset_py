import pandas as pd

import numpy as np

import xgboost as xgb

from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
train = train.drop('QuoteNumber', axis=1) 

test = test.drop('QuoteNumber', axis=1)



# Lets play with some dates

train['Date'] = pd.to_datetime(pd.Series(train['Original_Quote_Date']))

train = train.drop('Original_Quote_Date', axis=1)



test['Date'] = pd.to_datetime(pd.Series(test['Original_Quote_Date']))

test = test.drop('Original_Quote_Date', axis=1)



train['Year'] = train['Date'].apply(lambda x: int(str(x)[:4]))

train['Month'] = train['Date'].apply(lambda x: int(str(x)[5:7]))

train['weekday'] = train['Date'].dt.dayofweek



test['Year'] = test['Date'].apply(lambda x: int(str(x)[:4]))

test['Month'] = test['Date'].apply(lambda x: int(str(x)[5:7]))

test['weekday'] = test['Date'].dt.dayofweek



train = train.drop('Date', axis=1)  

test = test.drop('Date', axis=1)
#fill -999 to NAs

train = train.fillna(-999)

test = test.fillna(-999) 



features = list(train.columns[1:])  #la colonne 0 est le quote_conversionflag  

# print(features)





for f in train.columns:

    if train[f].dtype=='object':

        print(f)

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train[f].values) + list(test[f].values))

        train[f] = lbl.transform(list(train[f].values))

        test[f] = lbl.transform(list(test[f].values))
xgb_model = xgb.XGBClassifier()

parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower

              'objective':['binary:logistic'],

              'learning_rate': [0.05], #so called `eta` value

              'max_depth': [6],

              'min_child_weight': [11],

              'silent': [1],

              'subsample': [0.8],

              'colsample_bytree': [0.7],

              'n_estimators': [5], #number of trees, change it to 1000 for better results

              'missing':[-999],

              'seed': [1337]}
from sklearn.model_selection import StratifiedKFold

clf = GridSearchCV(xgb_model, parameters, 

                   cv=StratifiedKFold(n_splits=5, shuffle=True), 

                   scoring='roc_auc',

                   verbose=2, refit=True)



clf.fit(train[features], train["QuoteConversion_Flag"])
# #trust your CV!

# best_parameters,score = max(clf.scorer_, key=lambda x: x[1])

# print('Raw AUC score:', score)

# for param_name in sorted(best_parameters.keys()):

#     print("%s: %r" % (param_name, best_parameters[param_name]))



test_probs = clf.predict_proba(test[features])[:,1]
sample = pd.read_csv('../input/sample_submission.csv')

sample.QuoteConversion_Flag = test_probs

sample.to_csv("xgboost_best_parameter_submission.csv", index=False)