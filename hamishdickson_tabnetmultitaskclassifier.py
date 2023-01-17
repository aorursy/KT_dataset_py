!pip uninstall -y typing # this should avoid  AttributeError: type object 'Callable' has no attribute '_abc_registry'
import sys

sys.path.insert(0, "../input/tabnetfeatmultitaskclassification/tabnet-feat-MultiTaskClassification")
from pytorch_tabnet.multitask import TabNetMultiTaskClassifier



import torch

from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import roc_auc_score, log_loss



import pandas as pd

import numpy as np

np.random.seed(0)



from tqdm.notebook import tqdm



import os



from matplotlib import pyplot as plt

%matplotlib inline
dataset_name = "lish-moa"

train = pd.read_csv("../input/lish-moa/train_features.csv")

train_targets = pd.read_csv('../input/lish-moa/train_targets_scored.csv')

train_targets.drop(columns=["sig_id"], inplace=True)



test = pd.read_csv('../input/lish-moa/test_features.csv')
np.random.seed(42)

if "Set" not in train.columns:

    train["Set"] = np.random.choice(["train", "valid"], p =[.8, .2], size=(train.shape[0],))



train_indices = train[train.Set=="train"].index

valid_indices = train[train.Set=="valid"].index
# Encoding train set and test set



nunique = train.nunique()

types = train.dtypes



categorical_columns = []

categorical_dims =  {}

for col in tqdm(train.columns):

    if types[col] == 'object' or nunique[col] < 200:

        print(col, train[col].nunique())

        l_enc = LabelEncoder()

        train[col] = train[col].fillna("VV_likely")

        train[col] = l_enc.fit_transform(train[col].values)

        try:

            test[col] = test[col].fillna("VV_likely")

            test[col] = l_enc.transform(test[col].values)

        except:

            print(f"Column {col} does not exist in test set")

        categorical_columns.append(col)

        categorical_dims[col] = len(l_enc.classes_)

    else:

        training_mean = train.loc[train_indices, col].mean()

        train.fillna(training_mean, inplace=True)

        test.fillna(training_mean, inplace=True)
unused_feat = ['Set', 'sig_id'] # Let's not use splitting sets and sig_id



features = [ col for col in train.columns if col not in unused_feat] 



cat_idxs = [ i for i, f in enumerate(features) if f in categorical_columns]



cat_dims = [ categorical_dims[f] for i, f in enumerate(features) if f in categorical_columns]



X_train = train[features].values[train_indices]

y_train = train_targets.values[train_indices]



X_valid = train[features].values[valid_indices]

y_valid = train_targets.values[valid_indices]



X_test = test[features].values



clf = TabNetMultiTaskClassifier(cat_idxs=cat_idxs,

                                cat_dims=cat_dims,

                                cat_emb_dim=1,

                                optimizer_fn=torch.optim.Adam,

                                optimizer_params=dict(lr=2e-2),

                                scheduler_params={"step_size":50, # how to use learning rate scheduler

                                                  "gamma":0.9},

                                scheduler_fn=torch.optim.lr_scheduler.StepLR,

                                mask_type='entmax', # "sparsemax",

                                lambda_sparse=0, # don't penalize for sparser attention

                       

                      )
max_epochs = 1000

clf.fit(

    X_train=X_train, y_train=y_train,

    X_valid=X_valid, y_valid=y_valid,

    max_epochs=max_epochs ,

    patience=50, # please be patient ^^

    batch_size=1024,

    virtual_batch_size=128,

    num_workers=1,

    drop_last=False,

)



# scores displayed here are -average of log loss



# TabNet is not as fast as XGBoost (at least for binary classification and regression problems)

# If you wish to speed things up you could play with batch_size, virtual_batch_size and num_workers (or create a smaller network with less steps)

# Another way to speed things up is to improve the source code : please contribute here https://github.com/dreamquark-ai/tabnet/issues/183
# plot losses (drop first epochs to have a nice plot)

plt.plot(clf.history['train']['loss'][5:])

plt.plot(clf.history['valid']['loss'][5:])
# plot learning rates

plt.plot([x for x in clf.history['train']['lr']][5:])
preds_valid = clf.predict_proba(X_valid) # This is a list of results for each task



# We are here getting rid of tasks where only 0 are available in the validation set

valid_aucs = [roc_auc_score(y_score=task_pred[:,1], y_true=y_valid[:, task_idx])

             for task_idx, (task_pred, n_pos) in enumerate(zip(preds_valid, y_valid.sum(axis=0))) if n_pos > 0]



valid_logloss = [log_loss(y_pred=task_pred[:,1], y_true=y_valid[:, task_idx])

             for task_idx, (task_pred, n_pos) in enumerate(zip(preds_valid, y_valid.sum(axis=0))) if n_pos > 0]



plt.scatter(y_valid.sum(axis=0)[y_valid.sum(axis=0)>0], valid_aucs)
# Valid score should match mean log loss - They don't match exactly because we removed some tasks

print(f"BEST VALID SCORE FOR {dataset_name} : {clf.best_cost}")

print(f"VALIDATION MEAN LOGLOSS SCORES FOR {dataset_name} : {np.mean(valid_logloss)}")

print(f"VALIDATION MEAN AUC SCORES FOR {dataset_name} : {np.mean(valid_aucs)}")
preds = clf.predict_proba(X_test)
# save tabnet model

saving_path_name = "./TabNetMultiTaskClassifier_baseline"

saved_filepath = clf.save_model(saving_path_name)
# define new model with basic parameters and load state dict weights (all parameters will be updated)

loaded_clf = TabNetMultiTaskClassifier()

loaded_clf.load_model(saved_filepath)
loaded_preds = loaded_clf.predict_proba(X_test)



# Make sure that this is working as expected

np.testing.assert_array_equal(preds, loaded_preds)
clf.feature_importances_
explain_matrix, masks = clf.explain(X_test)
fig, axs = plt.subplots(1, 3, figsize=(20,20))



for i in range(3):

    axs[i].imshow(masks[i][:500])

    axs[i].set_title(f"mask {i}")
