from pathlib import Path



import numpy as np

import pandas as pd



from sklearn.model_selection import train_test_split



import lightgbm as lgb_org

import optuna.integration.lightgbm as lgb
base_dir = Path(

    "/kaggle/input/otto-group-product-classification-challenge/"

)
def read_csv(path):

    df = pd.read_csv(path)

    for col in df.columns:

        if col.startswith("feat_"):

            df[col] = df[col].astype("int32")

    return df
df = read_csv(str(base_dir / "train.csv"))
class_to_order = dict()

order_to_class = dict()



for idx, col in enumerate(df.target.unique()):

    order_to_class[idx] = col

    class_to_order[col] = idx



df["target_ord"] = df["target"].map(class_to_order)
feature_columns = [

    col for col in df.columns if col.startswith("feat_")

]

target_column = ["target_ord"]
X_train, X_val, y_train, y_val = train_test_split(

    df[feature_columns], df[target_column],

    test_size=0.3, random_state=42,

    stratify=df[target_column]

)
dtrain = lgb_org.Dataset(X_train, y_train)

dval = lgb_org.Dataset(X_val, y_val)
params = dict(

    objective="multiclass",

    metric="multi_logloss",

    num_class=9,

    seed=42,

)



best_params, tuning_history = dict(), list()

booster = lgb.train(params, dtrain, valid_sets=dval,

                    verbose_eval=0,

                    best_params=best_params,

                    early_stopping_rounds=5,

                    tuning_history=tuning_history)

 

print("Best Params:", best_params)

print("Tuning history:", tuning_history)
df_test = read_csv(str(base_dir / "test.csv"))
pred = booster.predict(df_test[feature_columns])
for idx, col in order_to_class.items():

    df_test[col] = pred[:,idx]
df_test[["id"] + [f"Class_{i}" for i in range(1, 10)]].to_csv('submission.csv', index=False)