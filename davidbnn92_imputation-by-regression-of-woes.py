import numpy as np 

import pandas as pd 

import os



from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, StratifiedKFold

from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve



from xgboost import XGBRegressor



import category_encoders as ce



%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns



test_features = pd.read_csv("../input/cat-in-the-dat-ii/test.csv")

train_set = pd.read_csv("../input/cat-in-the-dat-ii/train.csv")



train_targets = train_set.target

train_features = train_set.drop(['target'], axis=1)

percentage = train_targets.mean() * 100

print("The percentage of ones in the training target is {:.2f}%".format(percentage))

train_features.head()
columns = [col for col in train_features.columns if col != 'id']



# Encoding training data

df = train_features[columns]

train_encoded = pd.DataFrame()

skf = StratifiedKFold(n_splits=5,shuffle=True).split(df, train_targets)

for tr_in,fold_in in skf:

    encoder = ce.WOEEncoder(cols=columns, handle_missing='return_nan')

    encoder.fit(df.iloc[tr_in,:], train_targets.iloc[tr_in])

    train_encoded = train_encoded.append(encoder.transform(df.iloc[fold_in,:]),ignore_index=False)



train_encoded = train_encoded.sort_index()



# Encoding test data

encoder = ce.WOEEncoder(cols=columns, handle_missing='return_nan', handle_unknown='return_nan')

encoder.fit(df, train_targets)

test_encoded = encoder.transform(test_features[columns])



train_encoded.head()
# Each column is imputed through linear regression (ElasticNetCV), trained on other cols

# Imputation is done on a separate dataset, so that imputation of one col does not affect others

new_train_encoded = train_encoded.copy()

new_test_encoded = test_encoded.copy()



for currently_encoding in columns:

    tr_null = train_encoded.loc[train_encoded[currently_encoding].isnull()].copy()

    tr_not_null = train_encoded.loc[~train_encoded[currently_encoding].isnull()].copy()

    ts_null = test_encoded.loc[test_encoded[currently_encoding].isnull()]

    ts_not_null = test_encoded.loc[~test_encoded[currently_encoding].isnull()]



    temp_tr_feat = tr_not_null.drop(currently_encoding, axis=1).fillna(0)

    temp_tr_targ = tr_not_null[currently_encoding]

    temp_val_feat = tr_null.drop(currently_encoding, axis=1).fillna(0)

    temp_test_feat = ts_null.drop(currently_encoding, axis=1).fillna(0)

    

    parameters = {

        'n_estimators':30,

        'max_depth':15,

        'learning_rate':0.05,

        'reg_lambda':0.03,

        'reg_alpha':0.03,

        'random_state':1728

    }

    regressor = XGBRegressor(**parameters)

    regressor.fit(temp_tr_feat, temp_tr_targ)



    temp_pred = pd.Series(regressor.predict(temp_val_feat))

    temp_pred.index = temp_val_feat.index

    new_train_encoded.loc[train_encoded[currently_encoding].isnull(), [currently_encoding]] = temp_pred



    temp_pred = pd.Series(regressor.predict(temp_test_feat))

    temp_pred.index = temp_test_feat.index

    new_test_encoded.loc[test_encoded[currently_encoding].isnull(), [currently_encoding]] = temp_pred





new_train_encoded.info()
new_test_encoded.info()
# Fitting

regressor = LogisticRegression(solver='lbfgs', max_iter=1000, C=0.6)

regressor.fit(new_train_encoded, train_targets)



# Predicting

probabilities = [pair[1] for pair in regressor.predict_proba(new_test_encoded)]



# Submitting

output = pd.DataFrame({'id': test_features['id'],

                       'target': probabilities})

output.to_csv('submission.csv', index=False)

output.describe()