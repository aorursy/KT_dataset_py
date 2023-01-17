!mkdir -p /tmp/pip/cache/

!cp ../input/ngboost/autograd-1.3.xyz /tmp/pip/cache/autograd-1.3.tar.gz

!pip install --no-index --find-links /tmp/pip/cache/ autograd



!pip install "../input/ngboost/autograd_gamma-0.4.2-py3-none-any.whl"

!pip install "../input/ngboost/lifelines-0.25.2-py3-none-any.whl"

!pip install "../input/ngboost/ngboost-0.2.3-py3-none-any.whl"
import os, random

import typing as tp

import numpy as np

import pandas as pd



from tqdm.auto import tqdm

from functools import partial

from itertools import combinations

from itertools import combinations

from sklearn import model_selection

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import mean_squared_error



# NGBoost

# -------------------------------------------

#! pip install ngboost

from ngboost import NGBRegressor

from ngboost.learners import default_tree_learner

from ngboost.distns import Normal

from ngboost.scores import MLE
train = pd.read_csv(f"../input/osic-pulmonary-fibrosis-progression/train.csv")

test = pd.read_csv(f"../input/osic-pulmonary-fibrosis-progression/test.csv")

sample_submission = pd.read_csv(f"../input/osic-pulmonary-fibrosis-progression/sample_submission.csv")
FOLDS = 5

groups = train["Patient"].values

kfold = model_selection.GroupKFold(n_splits=FOLDS)

for n, (train_index, valid_index) in enumerate(kfold.split(train, train["FVC"], groups)):

    train.loc[valid_index, 'kfold'] = int(n)

train["kfold"] = train["kfold"].astype(int)
valid = train.copy()

train = train.copy()
pgb = train.groupby("Patient")

train = pd.DataFrame()

tk0 = tqdm(pgb, total=len(pgb))

for _, user_df in tk0:

    user_df = user_df.reset_index(drop=True)

    for index in list(combinations(user_df.index, 2)):

        df1 = user_df.iloc[[index[0]]].copy()

        df2 = user_df.iloc[[index[1]]].copy()

        # df1

        df1["base_Weeks"] = df2["Weeks"].iloc[0]

        df1["base_FVC"] = df2["FVC"].iloc[0]

        df1["base_Percent"] = df2["Percent"].iloc[0]

        train = pd.concat([train, df1])



        #df2

        df2["base_Weeks"] = df1["Weeks"].iloc[0]

        df2["base_FVC"] = df1["FVC"].iloc[0]

        df2["base_Percent"] = df1["Percent"].iloc[0]

        train = pd.concat([train, df2])

train = train.reset_index(drop=True)
pgb = valid.groupby("Patient")

valid = pd.DataFrame()

tk0 = tqdm(pgb, total=len(pgb))

for _, user_df in tk0:

    user_df = user_df.reset_index(drop=True)

    index = random.choice(user_df.index.values)

    df1 = user_df.iloc[index].copy()

    user_df["base_Weeks"] = df1["Weeks"]

    user_df["base_FVC"] = df1["FVC"]

    user_df["base_Percent"] = df1["Percent"]

    valid = pd.concat([valid, user_df])

valid = valid.reset_index(drop=True)
test = test.rename(columns={'Weeks': 'base_Weeks', 'FVC': 'base_FVC', 'Percent' : 'base_Percent'})

train["Patient_Week"] = train["Patient"] + '_' + train["Weeks"].astype(str)

valid["Patient_Week"] = valid["Patient"] + '_' + valid["Weeks"].astype(str)



sample_submission['Patient'] = sample_submission['Patient_Week'].apply(lambda x: x.split('_')[0])

sample_submission['Weeks'] = sample_submission['Patient_Week'].apply(lambda x: int(x.split('_')[1]))



test = pd.merge(sample_submission, test, how='left', left_on='Patient', right_on='Patient')



sample_submission.drop(columns=['Weeks', 'Patient'], inplace=True)
target = 'FVC'

col_drop = ['Patient', 'Patient_Week', 'kfold', 'Percent']

categorical_columns = []

categorical_dims = {}

for col in train.columns[train.dtypes == object]:

    if col not in col_drop + [target]:

        print(col, train[col].nunique())

        l_enc = LabelEncoder()

        train[col] = train[col].fillna("VV_likely")

        train[col] = l_enc.fit_transform(train[col].values)

        

        valid[col] = valid[col].fillna("VV_likely")

        valid[col] = l_enc.transform(valid[col].values)

        

        test[col] = test[col].fillna("VV_likely")

        test[col] = l_enc.transform(test[col].values)

        

        categorical_columns.append(col)

        categorical_dims[col] = len(l_enc.classes_)
def score(actual_fvc, predicted_fvc, confidence):

    

    sd_clipped = np.maximum(confidence, 70)

    delta = np.minimum(np.abs(actual_fvc - predicted_fvc), 1000)

    metric = - np.sqrt(2) * delta / sd_clipped - np.log(np.sqrt(2) * sd_clipped)

    

    return np.mean(metric)
quantiles = (0.2, 0.5, 0.8)

def quantile_loss(target, preds):

    #assert not target.requires_grad

    assert preds.size(0) == target.size(0)

    losses = []

    for i, q in enumerate(quantiles):

        errors = target[:, i] - preds[:, i]

        losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(1))

    loss = torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))

    return loss
def run_one_fold(clf,train_df,valid_df,test_df,features,targets,categorical=[],fold=0):

    

    trn_idx = train_df[train_df.kfold != fold].index

    val_idx = train_df[train_df.kfold == fold].index

    print(f'len(trn_idx) : {len(trn_idx)}')

    print(f'len(val_idx) : {len(val_idx)}')



    X_train = train_df.iloc[trn_idx][features]

    X_valid = train_df.iloc[val_idx][features]

    X_test  = test_df[features].values

    



    y_train = train_df.iloc[trn_idx][targets].values#.reshape(-1, 1)

    y_valid = train_df.iloc[val_idx][targets].values#.reshape(-1, 1)



    

    fold_oof_df = pd.DataFrame()

    predictions = np.zeros((len(test_df), 2))

    



    clf.fit(X_train, y_train,

        X_val = X_valid , Y_val = y_valid ,

        sample_weight = None ,

        val_sample_weight = None ,

        train_loss_monitor = None ,

        val_loss_monitor = None ,

        early_stopping_rounds = 500

    )

    

    

    fold_oof_df["pred_FVC"] = clf.predict(X_valid.values)

    

    y_dists = clf.pred_dist(X_valid.values)

    #print(np.array(list(y_dists.dist.interval(alpha=0.8)[0])).shape) #- np.array(list(y_dists.dist.interval(0.2))))

    fold_oof_df["Confidence"] = np.abs(np.array(list(y_dists.dist.interval(alpha=0.8)[0])) - np.array(list(y_dists.dist.interval(0.2)[0])))

    #fold_oof_df["Confidence"] = clf.pred_dist(X_valid.values).params['scale']

    fold_oof_df["fold"] = fold

    fold_oof_df["Patient_Week"] = train_df.iloc[val_idx]["Patient_Week"].values

    fold_oof_df["FVC"] = y_valid

    

    y_test_dists = clf.pred_dist(X_test)

    predictions[:, 0] = clf.predict(X_test)

    predictions[:, 1] =  np.abs(np.array(list(y_test_dists.dist.interval(alpha=0.8)[0])) - np.array(list(y_test_dists.dist.interval(0.2)[0])))



    # RMSE

    print("fold{} RMSE score: {:<8.5f}".format(

        fold, np.sqrt(mean_squared_error(y_valid, fold_oof_df["pred_FVC"].values))))

    # Competition Metric

    print("fold{} Metric: {:<8.5f}".format(

        fold, score(fold_oof_df['FVC'].values, fold_oof_df["pred_FVC"].values, fold_oof_df['Confidence'])

    ))

    

    return fold_oof_df, predictions
def run_kfold(

    clf, 

    train,

    valid,

    test,

    features, 

    target, 

    n_fold=FOLDS, 

    categorical=[], 

    my_loss=None

):

          

          

    

    print(f"================================= FOLDS : {n_fold} =================================")

    

    oof_df = pd.DataFrame()

    predictions = np.zeros((len(test), 2))



    for fold_ in range(n_fold):

        

        print("Fold {}".format(fold_))

        fold_oof_df, fold_predictions = run_one_fold(

                clf, 

                train,

                valid, 

                test,

                features, 

                target, 

                fold=fold_

        )

          

        oof_df = pd.concat([oof_df, fold_oof_df], axis=0)

        predictions += fold_predictions        



    # RMSE

    print("CV RMSE score: {:<8.5f}".format(np.sqrt(mean_squared_error(oof_df["FVC"], oof_df["pred_FVC"]))))

    # Metric

    print("CV Metric: {:<8.5f}".format(

        score(oof_df['FVC'].values, oof_df["pred_FVC"].values, oof_df['Confidence'].values)

    ))

    

    predictions = predictions / n_fold



    print(f"=========================================================================================")

    

    return oof_df, predictions
from lightgbm import LGBMRegressor
targets = 'FVC'

features = [col for col in train.columns if col not in col_drop + [target]]





pram = {

    #'Base' : LGBMRegressor(

      #**{

      #      'learning_rate': 0.03585185547472276,

      #      'max_depth': 2,

      #      'n_estimators': 4558,

      #      'num_leaves': 459

      #   }

    #),

    #'col_sample': 1.0, 

    #'minibatch_frac': 0.55, 

    'n_estimators': 1000

}



clf = NGBRegressor(**pram)



oof_df, predictions = run_kfold(

    clf,

    train,

    valid,

    test,

    features,

    targets,

    n_fold=5,

    categorical=None

)
predictions
sample_submission["FVC"] = predictions[:, 0]

sample_submission['Confidence'] = predictions[:, 1]
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)