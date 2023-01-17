!pip install "../input/pytorch-tabnet/pytorch_tabnet-1.2.0-py3-none-any.whl" >> quit
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



import torch

from pytorch_tabnet.tab_model import TabNetRegressor
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
quantiles = (0.1, 0.5, 0.9)

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

    val_idx = valid_df[valid_df.kfold == fold].index

    print(f'len(trn_idx) : {len(trn_idx)}')

    print(f'len(val_idx) : {len(val_idx)}')



    X_train = train_df.iloc[trn_idx][features]

    X_valid = valid_df.iloc[val_idx][features]

    X_test  = test_df[features].values

    



    y_train = train_df.iloc[trn_idx][targets].values#.reshape(-1, 1)

    y_valid = valid_df.iloc[val_idx][targets].values#.reshape(-1, 1)





    cat_idxs = [ i for i, f in enumerate(X_train.columns) if f in categorical_columns]

    cat_dims = [ categorical_dims[f] for i, f in enumerate(X_train.columns) if f in categorical_columns]



    # define your embedding sizes : hear just a random choice

    cat_emb_dim = cat_dims



    

    fold_oof_df = pd.DataFrame()

    predictions = np.zeros((len(test_df), 2))

    

    



    clf = TabNetRegressor(

        cat_dims=cat_dims,

        cat_emb_dim=cat_emb_dim,

        cat_idxs=cat_idxs,

        output_dim=3

    )



    clf.fit(

        X_train=X_train.values, y_train=y_train,

        X_valid=X_valid.values, y_valid=y_valid,

        max_epochs=1000,

        patience=100,

        batch_size=1024, 

        virtual_batch_size=128,

        loss_fn=quantile_loss,

        num_workers=0,

        drop_last=False

    )

    

    fold_oof_df["pred_FVC"] = clf.predict(X_valid.values)[:,1]

    fold_oof_df["Confidence0"] = clf.predict(X_valid.values)[:,0]

    fold_oof_df["Confidence2"] = clf.predict(X_valid.values)[:,2]

    fold_oof_df["Confidence"] = abs(fold_oof_df["Confidence2"] - fold_oof_df["Confidence0"])

    fold_oof_df["fold"] = fold

    fold_oof_df["Patient_Week"] = valid_df.iloc[val_idx]["Patient_Week"].values

    fold_oof_df["FVC"] = y_valid[:,1]

    

    test_predictions = clf.predict(X_test)

    

    predictions[:, 0] = test_predictions[:, 1]

    predictions[:, 1] = np.abs(test_predictions[:, 0] - test_predictions[:, 2])



    # RMSE

    print("fold{} RMSE score: {:<8.5f}".format(

        fold, np.sqrt(mean_squared_error(y_valid[:,1], fold_oof_df["pred_FVC"].values))))

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
targets = ['FVC', 'FVC', 'FVC']

features = [col for col in train.columns if col not in col_drop + [target]]



clf = None



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
sample_submission["FVC"] = predictions[:, 0]

sample_submission['Confidence'] = predictions[:, 1]
sample_submission.head()
sample_submission.to_csv("submission.csv", index=False)