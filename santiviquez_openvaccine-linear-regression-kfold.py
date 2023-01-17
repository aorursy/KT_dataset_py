import pandas as pd

import numpy as np

from sklearn.model_selection import KFold

from sklearn import preprocessing

from sklearn import linear_model
train = pd.read_json('../input/stanford-covid-vaccine/train.json', lines=True)

test = pd.read_json('../input/stanford-covid-vaccine/test.json', lines=True)

submission = pd.read_csv('../input/stanford-covid-vaccine/sample_submission.csv')
FOLDS = 5

SEED = 43
# suffle dataset

train = train.sample(frac=1, random_state=SEED).reset_index(drop=True)

# create kfold columns

train['kfold'] = -1

# initiate kfold class

kf = KFold(n_splits = FOLDS, random_state=SEED, shuffle=True)



# fill the new column kfold

for f, (t_, v_) in enumerate(kf.split(X=train)):

    train.loc[v_, 'kfold'] = f
train.kfold.value_counts()
# Calculate Means of targets

train['reactivity'] = train['reactivity'].apply(lambda x: np.mean(x))

train['deg_Mg_pH10'] = train['deg_Mg_pH10'].apply(lambda x: np.mean(x))

train['deg_pH10'] = train['deg_pH10'].apply(lambda x: np.mean(x))

train['deg_Mg_50C'] = train['deg_Mg_50C'].apply(lambda x: np.mean(x))

train['deg_50C'] = train['deg_50C'].apply(lambda x: np.mean(x))
def mcrmse_loss(y_true, y_pred, N=3):

    """

    Calculates competition eval metric.

    From: https://www.kaggle.com/kaushal2896/openvaccine-xgboost-baseline

    """

    assert len(y_true) == len(y_pred)

    n = len(y_true)

    return np.sum(np.sqrt(np.sum((y_true - y_pred)**2, axis=0)/n)) / N
features = ['sequence', 'structure', 'predicted_loop_type']

targets = ['reactivity', 'deg_Mg_pH10', 'deg_Mg_50C']

preds_list = []



def run(fold):

    

    # training and validation sets

    X_train = train[train.kfold != fold].reset_index(drop=True)

    X_valid = train[train.kfold == fold].reset_index(drop=True)

    

    # training and validation labels

    y_train = X_train[targets]

    y_valid = X_valid[targets]

    

    # encode features

    # initialize OneHotEncoder from sklearn

    # we use handle_unknown='ignore' just because it is a baseline

    # but we should figure it out a better way to encode the unknown values

    # that we are going to find in the test dataset

    ohe = preprocessing.OneHotEncoder(handle_unknown='ignore')

    

    # fit ohe on training and validation features

    full_data = pd.concat(

        [X_train[features], X_valid[features]],

        axis=0

    )

    ohe.fit(full_data[features])

    

    # transform training and validation data

    X_train = ohe.transform(X_train[features])

    X_valid = ohe.transform(X_valid[features])

    x_test = ohe.transform(test[features])

    

    # initialize regression model

    model = linear_model.LinearRegression()

    

    # fit the model on training data

    model.fit(X_train, y_train)

    

    # predict on validation data

    valid_preds = model.predict(X_valid)

    

    mcrmse = mcrmse_loss(y_valid, valid_preds)

    print(f"FOLD {fold}")

    print(f"Valid MCRMSE: {mcrmse}")

    print("")

    

    # predict on test dataset

    preds = model.predict(x_test)

    preds_list.append(preds)
for f in range(FOLDS):

    run(f)
# average predictions of each of the folds models

predictions = (preds_list[0] + preds_list[1] + preds_list[2] +

               preds_list[3] + preds_list[4]) / FOLDS



predictions = pd.DataFrame(predictions)
# Create submission csv

submission_df = predictions.loc[predictions.index.repeat(list(test['seq_length']))].reset_index(drop=True)

submission_df = submission_df.rename(columns={0: 'reactivity', 1: 'deg_Mg_pH10', 2: 'deg_Mg_50C'})

submission_df['id_seqpos'] = submission['id_seqpos']

submission_df['deg_pH10'] = 0.0

submission_df['deg_50C'] = 0.0

submission_df = submission_df[['id_seqpos', 'reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']]

submission_df.head()
submission_df.to_csv('submission.csv', index=False)