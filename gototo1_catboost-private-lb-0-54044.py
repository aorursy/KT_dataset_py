#Import Datafiles

import pandas as pd

train = pd.read_csv('../input/shopee-marketing-analytics/train_clean.csv')

test = pd.read_csv('../input/shopee-marketing-analytics/test_clean.csv')



pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows', 200)



print("Train Shape: {}".format(train.shape))

print("Test Shape: {}".format(test.shape))
#Preparing Data for Modelling

train_id = train.row_id

test_id = test.row_id

target = train.open_flag

train.drop(columns=["row_id", "open_flag"], inplace=True)

test_row_ids = test.pop('row_id')
#CatBoost Model from https://www.kaggle.com/wakamezake/starter-code-catboost-baseline



from sklearn.model_selection import KFold, train_test_split

from sklearn.metrics import roc_auc_score

from catboost import Pool, CatBoostClassifier



model = CatBoostClassifier(loss_function="Logloss",

                           eval_metric="AUC",

                           learning_rate=0.01,

                           iterations=10000,

                           random_seed=42,

                           od_type="Iter",

                           depth=10,

                           early_stopping_rounds=500

                          )



n_split = 3

kf = KFold(n_splits=n_split, random_state=42, shuffle=True)



y_valid_pred = 0 * target

y_test_pred = 0



for idx, (train_index, valid_index) in enumerate(kf.split(train)):

    y_train, y_valid = target.iloc[train_index], target.iloc[valid_index]

    X_train, X_valid = train.iloc[train_index,:], train.iloc[valid_index,:]

    _train = Pool(X_train, label=y_train)

    _valid = Pool(X_valid, label=y_valid)

    print( "\nFold ", idx)

    fit_model = model.fit(_train,

                          eval_set=_valid,

                          use_best_model=True,

                          verbose=200,

                          plot=True

                         )

    pred = fit_model.predict_proba(X_valid)[:,1]

    print( "  auc = ", roc_auc_score(y_valid, pred) )

    y_valid_pred.iloc[valid_index] = pred

    y_test_pred += fit_model.predict_proba(test)[:,1]

y_test_pred /= n_split





#Prediction

output_df = pd.DataFrame({'row_id': test_row_ids, 'open_flag': y_test_pred})
#Setting Threshold for Skewed Data

output_df['open_flag'] = output_df['open_flag'].apply(lambda x: int(1) if x >0.261 else int(0))

print(output_df['open_flag'].value_counts())
#Submission

output_df.to_csv('catboost_submission.csv', index=False)