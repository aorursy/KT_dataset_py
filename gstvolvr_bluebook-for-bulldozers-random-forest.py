import pandas as pd

import numpy as np

import os



from IPython.display import display

from sklearn.ensemble import ExtraTreesRegressor

from sklearn import metrics

from fastai.tabular import cont_cat_split, add_datepart
df_raw_train = pd.read_csv("../input/train/Train.csv", parse_dates=["saledate"], low_memory=False)

df_raw_test = pd.read_csv("../input/Test.csv", parse_dates=["saledate"], low_memory=False)

df_raw_valid = pd.read_csv("../input/Valid.csv", parse_dates=["saledate"], low_memory=False)
def build_features(frame):

    df = frame.copy()

    df = add_datepart(df, 'saledate') 

    cont_cols, cat_cols = cont_cat_split(df)



    for col in cont_cols:

        if pd.isnull(df[col]).sum(): 

            df.drop(col, axis=1)

        df[col] = df[col].fillna(df[col].median())



    df[cat_cols] = df[cat_cols].astype('category')

    for col in cat_cols:

        df[col] = df[col].cat.codes + 1

        

    return df
df_train = build_features(df_raw_train)

df_test = build_features(df_raw_test)

df_valid = build_features(df_raw_valid)
os.makedirs("tmp", exist_ok=True)

df_train.to_feather("tmp/features")
from sklearn.model_selection import cross_val_score, train_test_split



label = 'SalePrice'



df_train[label] = np.log(df_train[label])



# split based on time

def split_vals(a, n): 

    return a[:n].copy(), a[n:].copy()



n_valid = 12000

n_trn = len(df_train) - n_valid



# cols_to_drop = [c for c in df_train.columns if c[-3:] == "_na"] + [label]

x = df_train.drop(label, axis=1)

y = df_train[label]



feature_cols = x.columns



x_train, x_valid = split_vals(x, n_trn)

y_train, y_valid = split_vals(y, n_trn)



[x.shape for x in [x_train, x_valid]]
def rmse(x, y): 

    return np.sqrt(metrics.mean_squared_error(x, y))



def print_score(m):

    res = [rmse(m.predict(x_train), y_train),

           rmse(m.predict(x_valid), y_valid),

           m.score(x_train, y_train),

           m.score(x_valid, y_valid)]

    

    if hasattr(m, "oob_score_"):

        res.append(m.oob_score_)

    print(res)
rgs = ExtraTreesRegressor(n_estimators=10, n_jobs=-1)

%time rgs.fit(x_train, y_train)



print_score(rgs)
df_test[label] = rgs.predict(df_test[feature_cols])

df_test[['SalesID', label]].to_csv("submission.csv", index=False)