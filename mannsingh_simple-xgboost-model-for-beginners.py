import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.model_selection import KFold

from sklearn.metrics import mean_absolute_error, log_loss



from xgboost import XGBRegressor
sample = pd.read_csv("../input/lish-moa/sample_submission.csv")



test_features = pd.read_csv("../input/lish-moa/test_features.csv")

train_features = pd.read_csv("../input/lish-moa/train_features.csv")

train_score = pd.read_csv("../input/lish-moa/train_targets_scored.csv")
sample.head()
test_features
train_features
train_score
g_features = [feature for feature in train_features.columns if feature.startswith('g-')]

c_features = [feature for feature in train_features.columns if feature.startswith('c-')]

other_features = [feature for feature in train_features.columns if feature not in g_features and feature not in c_features]

                                                            



print(f'Number of g- Features: {len(g_features)}')

print(f'Number of c- Features: {len(c_features)}')

print(f'Number of Other Features: {len(other_features)} ({other_features})')
def preprocess(df):

    df = df.copy()

    df.loc[:, 'cp_type'] = df.loc[:, 'cp_type'].map({'trt_cp': 0, 'ctl_vehicle': 1})

    df.loc[:, 'cp_dose'] = df.loc[:, 'cp_dose'].map({'D1': 0, 'D2': 1})

    del df['sig_id']

    return df



train = preprocess(train_features)

test = preprocess(test_features)

del train_score['sig_id']
test
def metric(y_true, y_pred):

    metrics = []

    metrics.append(log_loss(y_true, y_pred.astype(float), labels=[0,1]))

    return np.mean(metrics)
cols = train_score.columns

submission = sample.copy()

submission.loc[:,train_score.columns] = 0

#test_preds = np.zeros((test.shape[0], train_score.shape[1]))

N_SPLITS = 5

oof_loss = 0



for c, column in enumerate(cols,1):

    y = train_score[column]

    total_loss = 0

    

    for fn, (trn_idx, val_idx) in enumerate(KFold(n_splits = N_SPLITS, shuffle = True).split(train)):

        print('Fold: ', fn+1)

        X_train, X_val = train.iloc[trn_idx], train.iloc[val_idx]

        y_train, y_val = y.iloc[trn_idx], y.iloc[val_idx]

        

        model = XGBRegressor(tree_method = 'gpu_hist',

                         min_child_weight = 31.58,

                         learning_rate = 0.05,

                         colsample_bytree = 0.65,

                         gamma = 3.69,

                         max_delta_step = 2.07,

                         max_depth = 10,

                         n_estimators = 166,

                         subsample = 0.86)

    

        model.fit(X_train, y_train)

        pred = model.predict(X_val)

        #pred = [n if n>0 else 0 for n in pred]

        loss = metric(y_val,pred)

        total_loss += loss

        predictions = model.predict(test)

        #predictions = [n if n>0 else 0 for n in predictions]

        submission[column] += predictions/N_SPLITS

        

    #submission[column] = submission[column]/N_SPLITS

    oof_loss += total_loss/N_SPLITS

    print("Model "+str(c)+": Loss ="+str(total_loss/N_SPLITS))
oof_loss/206
submission
submission.loc[test['cp_type']==1, train_score.columns] = 0
submission.to_csv('submission.csv', index=False)