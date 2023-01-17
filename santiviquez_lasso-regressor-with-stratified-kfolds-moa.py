import pandas as pd

import numpy as np

from sklearn import model_selection

from sklearn import linear_model

from sklearn import metrics

from sklearn import preprocessing
folds = 5
# read train and targets datasets

train_df = pd.read_csv('../input/lish-moa/train_features.csv')

targets_df = pd.read_csv('../input/lish-moa/train_targets_scored.csv')



# join both datasets so later we can shuffle the joined df

df = train_df.merge(targets_df)



# suffle dataset

df = df.sample(frac=1).reset_index(drop=True)



train_cols = train_df.columns

target_cols = targets_df.drop('sig_id', axis=1).columns



# lets re-constructed our initial train and targets dataframes from the shuffled df

train_df = df[train_cols]

targets_df = df[target_cols]



# create a new column called kfold

train_df['kfold'] = -1

targets_df['kfold'] = -1



# concat all target values with each other and assing the result on a new column called target_concat

targets_df['target_concat'] = targets_df[target_cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)



# now we want to label encode these target_concat values

# initialize LabelEncoder

lbl = preprocessing.LabelEncoder()

# fit LabelEncoder with all the data

lbl.fit(targets_df['target_concat'])

# transform all the data

targets_df.loc[:, 'target_enc'] = lbl.transform(targets_df['target_concat'])



# fetch the new encoded targets

y = targets_df.target_enc



# initiate the kfold class form model_selection module

kf = model_selection.StratifiedKFold(n_splits=folds)



# fill the new column kfold

for f, (t_, v_) in enumerate(kf.split(X=train_df, y=y)):

    train_df.loc[v_, 'kfold'] = f

    targets_df.loc[v_, 'kfold'] = f



# targets_df.drop(['target_concat', 'target_enc'], axis=1, inplace=True)
train_df.kfold.value_counts()
# now all the folds have the same (or almost the same) target distribution

for f in range(folds):

    print("")

    print(f"FOLD: {f}")

    print(targets_df[targets_df['kfold'] == f]['target_enc'].value_counts())
# write folds datasets

targets_df.drop(['target_concat', 'target_enc'], axis=1, inplace=True)

targets_df.to_csv('train_targets_scored_folds.csv', index=False)

train_df.to_csv('train_features_folds.csv', index=False)
predictions_list = []



def run(fold):

    submission = pd.read_csv('../input/lish-moa/sample_submission.csv')

    train_df = pd.read_csv('./train_features_folds.csv')

    targets_df = pd.read_csv('./train_targets_scored_folds.csv')

    test_df = pd.read_csv('../input/lish-moa/test_features.csv')

    

    # lets label encode categorical columns

    for col in ['cp_type', 'cp_time', 'cp_dose']:

        # initialize LabelEncoder

        lbl = preprocessing.LabelEncoder()

        # fit LabelEncoder with all the data

        lbl.fit(train_df[col])

        # transform all the data

        train_df.loc[:, col] = lbl.transform(train_df[col])

        test_df.loc[:, col] = lbl.transform(test_df[col])



    # get training data using folds

    df_train = train_df[train_df['kfold'] != fold].reset_index(drop=True)

    # get validation data using folds

    df_valid = train_df[train_df['kfold'] == fold].reset_index(drop=True)

        

    # get training targets using folds

    y_train = targets_df[targets_df['kfold'] != fold].reset_index(drop=True)

    # get validation targets using folds

    y_valid = targets_df[targets_df['kfold'] == fold].reset_index(drop=True)

    

    # drop sig_id column

    x_train = df_train.drop(['sig_id', 'kfold'],axis=1)

    x_valid = df_valid.drop(['sig_id', 'kfold'],axis=1)

    x_test = test_df.drop(['sig_id'], axis=1)

    y_train = y_train.drop(['kfold'],axis=1)

    y_valid = y_valid.drop(['kfold'],axis=1)

    

    # initialize regression model

    model = linear_model.Lasso(alpha=0.1, max_iter=3000, random_state=42, selection='random')

    

    # fit model on training data

    model.fit(x_train, y_train)

    

    # predict on validation data

    valid_preds = model.predict(x_valid)    

    

    # get log loss score    

    log_loss = metrics.log_loss(np.ravel(y_valid), np.ravel(valid_preds))

    # print auc

    print("")

    print(f'Fold = {fold}, log loss = {log_loss}')



    # predictions

    predictions = model.predict(x_test)

    predictions_list.append(predictions)
# run for each fold

for fold in range(folds):

    run(fold)
# average predictions of each of the folds models

predictions = (predictions_list[0] + predictions_list[1] + predictions_list[2] +

               predictions_list[3] + predictions_list[4]) / folds

submission = pd.read_csv('/kaggle/input/lish-moa/sample_submission.csv')



for i in range(len(submission.columns)):

    if i != 0:

        col = submission.columns[i]

        submission[col] = predictions[:, i - 1]
submission.to_csv('submission.csv',index=False)