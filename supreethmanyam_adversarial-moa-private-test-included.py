import numpy as np

import pandas as pd



from sklearn import model_selection, metrics

import lightgbm as lgb



import matplotlib.pyplot as plt



from pathlib import Path
SEED = 1729

INPUT_PATH = Path("../input/lish-moa/")
train_features = pd.read_csv(INPUT_PATH/"train_features.csv"); print(f"Train features shape: {train_features.shape}")

test_features = pd.read_csv(INPUT_PATH/"test_features.csv"); print(f"Test features shape: {test_features.shape}")



train_targets = pd.read_csv(INPUT_PATH/"train_targets_scored.csv"); print(f"Train targets shape: {train_targets.shape}")
train_features.head()
test_features.head()
# Build a model that can separate train and test based on the provided features.

train_features["is_test"] = 0

test_features["is_test"] = 1



panel = pd.concat([train_features, test_features], sort=False, ignore_index=True)
cp_type_dict = {"trt_cp": 0, "ctl_vehicle": 1}

cp_dose_dict = {"D1": 0, "D2": 1}



panel["cp_type"] = panel["cp_type"].map(cp_type_dict)

panel["cp_dose"] = panel["cp_dose"].map(cp_dose_dict)
columns_for_model = panel.columns[~np.in1d(panel.columns, ["sig_id", "is_test"])]

print(len(columns_for_model))
params = {

    "num_leaves": 128,

    "min_data_in_leaf": 64, 

    "objective": "binary",

    "max_depth": 6,

    "learning_rate": 0.001,

    "min_child_samples": 64,

    "boosting": "gbdt",

    "feature_fraction": 0.9,

    "bagging_freq": 5,

    "bagging_fraction": 0.9 ,

    "bagging_seed": SEED,

    "metric": "auc",

    "lambda_l1": 50.0,

    "lambda_l2": 10.0,

    "verbosity": -1

}

num_rounds = 1000

early_stopping_rounds = 50

verbose_eval = 50
kf = model_selection.KFold(n_splits=5, shuffle=True, random_state=SEED)

cv_scores = []

models = []

for fold_idx, (dev_idx, val_idx) in enumerate(kf.split(panel)):

    print(f"Fold: {fold_idx+1}")

    X_dev, y_dev = panel.loc[dev_idx, columns_for_model], panel.loc[dev_idx, "is_test"].values

    X_val, y_val = panel.loc[val_idx, columns_for_model], panel.loc[val_idx, "is_test"].values

    

    dev_dataset = lgb.Dataset(X_dev, y_dev)

    val_dataset = lgb.Dataset(X_val, y_val)

    

    clf = lgb.train(

        params,

        dev_dataset,

        num_rounds,

        valid_sets=[dev_dataset, val_dataset],

        early_stopping_rounds=early_stopping_rounds,

        verbose_eval=verbose_eval

    )

    

    cv_scores.append(clf.best_score["valid_1"]['auc'])

    models.append(clf)

    print("\n")

    

adversarial_validation_auc = np.mean(cv_scores)

print(f"Mean Adversarial AUC: {adversarial_validation_auc:.4f}")
model = models[-1]

fig, ax = plt.subplots(figsize=(15,15))

lgb.plot_importance(model, max_num_features=50, importance_type="gain", height=0.8, ax=ax)

ax.grid(False)

plt.title("LightGBM - Feature Importance by gain (Adversarial Validation)", fontsize=15)

plt.show()
sample_submission = pd.read_csv(INPUT_PATH/"sample_submission.csv")
if adversarial_validation_auc < 0.55:

    sample_submission.to_csv("submission.csv", index=False)

else:

    pass