!pip install pytorch-tabnet
# Preliminaries

import numpy as np

import pandas as pd 

import os

import random



#Visuals

import matplotlib.pyplot as plt

import seaborn as sns



#Torch and Tabnet

import torch

from pytorch_tabnet.tab_model import TabNetRegressor



#Sklearn only for splitting

from sklearn.model_selection import KFold
NUM_FOLDS = 7  # you can specify your folds here

seed = 2020   # seed for reproducible results
def seed_everything(seed):

    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
seed_everything(seed)
def metric(y_true, y_pred):

    

    overall_score = 0

    

    weights = [.3, .175, .175, .175, .175]

    

    for i,w in zip(range(y_true.shape[1]),weights):

        ind_score = np.mean(np.sum(np.abs(y_true[:,i] - y_pred[:,i]), axis=0)/np.sum(y_true[:,i], axis=0))

        overall_score += w*ind_score

    

    return overall_score
fnc_df = pd.read_csv("../input/trends-assessment-prediction/fnc.csv")

loading_df = pd.read_csv("../input/trends-assessment-prediction/loading.csv")



fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])

df = fnc_df.merge(loading_df, on="Id")

features = fnc_features + loading_features





labels_df = pd.read_csv("../input/trends-assessment-prediction/train_scores.csv")

target_features = list(labels_df.columns[1:])

labels_df["is_train"] = True





df = df.merge(labels_df, on="Id", how="left")



test_df = df[df["is_train"] != True].copy()

df = df[df["is_train"] == True].copy()



df.shape, test_df.shape
# Creating FOLDS



df = df.dropna().reset_index(drop=True)

df["kfold"] = -1



df = df.sample(frac=1,random_state=2020).reset_index(drop=True)



kf = KFold(n_splits=NUM_FOLDS)



for fold, (trn_, val_) in enumerate(kf.split(X=df, y=df)):

    df.loc[val_, 'kfold'] = fold
# Giving less importance to FNC features since they are easier to overfit due to high dimensionality.

FNC_SCALE = 1/500



df[fnc_features] *= FNC_SCALE

test_df[fnc_features] *= FNC_SCALE
model = TabNetRegressor(n_d=16,

                       n_a=16,

                       n_steps=4,

                       gamma=1.9,

                       n_independent=4,

                       n_shared=5,

                       seed=seed,

                       optimizer_fn = torch.optim.Adam,

                       scheduler_params = {"milestones": [150,250,300,350,400,450],'gamma':0.2},

                       scheduler_fn=torch.optim.lr_scheduler.MultiStepLR)
y_test = np.zeros((test_df.shape[0],len(target_features), NUM_FOLDS))  #A 3D TENSOR FOR STORING RESULTS OF ALL FOLDS
def run(fold):

    df_train = df[df.kfold != fold]

    df_valid = df[df.kfold == fold]

    

    X_train = df_train[features].values

    Y_train = df_train[target_features].values

    

    X_valid = df_valid[features].values

    Y_valid = df_valid[target_features].values

    

    y_oof = np.zeros((df_valid.shape[0],len(target_features)))   # Out of folds validation

    

    print("--------Training Begining for fold {}-------------".format(fold+1))

     

    model.fit(X_train = X_train,

             y_train = Y_train,

             X_valid = X_valid,

             y_valid = Y_valid,

             max_epochs = 1000,

             patience =70)

              

    

    print("--------Validating For fold {}------------".format(fold+1))

    

    y_oof = model.predict(X_valid)

    y_test[:,:,fold] = model.predict(test_df[features].values)

    

    val_score = metric(Y_valid,y_oof)

    

    print("Validation score: {:<8.5f}".format(val_score))

    

    # VISUALIZTION

    plt.figure(figsize=(12,6))

    plt.plot(model.history['train']['loss'])

    plt.plot(model.history['valid']['loss'])

    

    #Plotting Metric

    #plt.plot([-x for x in model.history['train']['metric']])

    #plt.plot([-x for x in model.history['valid']['metric']])
run(fold=0)
run(fold=1)
run(fold=2)
run(fold=3)
run(fold=4)
run(fold=5)
run(fold=6)
y_test = y_test.mean(axis=-1) # Taking mean of all the fold predictions

test_df[target_features] = y_test
test_df = test_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]]
sub_df = pd.melt(test_df, id_vars=["Id"], value_name="Predicted")

sub_df["Id"] = sub_df["Id"].astype("str") + "_" +  sub_df["variable"].astype("str")



sub_df = sub_df.drop("variable", axis=1).sort_values("Id")

assert sub_df.shape[0] == test_df.shape[0]*5

sub_df.head(10)
sub_df.to_csv('submission.csv',index=False)