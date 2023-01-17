# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from sklearn.svm import SVR

from sklearn.model_selection import KFold



def metric(y_true, y_pred):

    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))



fnc_df = pd.read_csv("../input/trends-assessment-prediction/fnc.csv")

fnc_df
loading_df = pd.read_csv("../input/trends-assessment-prediction/loading.csv")

fnc_features, loading_features = list(fnc_df.columns[1:]), list(loading_df.columns[1:])

loading_df
df = fnc_df.merge(loading_df, on="Id")

df
labels_df = pd.read_csv("../input/trends-assessment-prediction/train_scores.csv")

labels_df
labels_df["is_train"] = True

df = df.merge(labels_df, on="Id", how="left")

df
test_df = df[df["is_train"] != True].copy()

df = df[df["is_train"] == True].copy()



df.shape, test_df.shape
# Giving less importance to FNC features since they are easier to overfit due to high dimensionality.

FNC_SCALE = 1/500



df[fnc_features] *= FNC_SCALE

test_df[fnc_features] *= FNC_SCALE
%%time

NUM_FOLDS = 7

kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)





features = loading_features + fnc_features



overal_score = 0

for target, c, w in [("age", 100, 0.3), ("domain1_var1", 10, 0.175), ("domain1_var2", 10, 0.175), ("domain2_var1", 10, 0.175), ("domain2_var2", 10, 0.175)]:    

    y_oof = np.zeros(df.shape[0])

    y_test = np.zeros((test_df.shape[0], NUM_FOLDS))

    

    for f, (train_ind, val_ind) in enumerate(kf.split(df, df)):

        train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]

        train_df = train_df[train_df[target].notnull()]



        model = SVR(C=c, cache_size=3000.0)

        model.fit(train_df[features], train_df[target])



        y_oof[val_ind] = model.predict(val_df[features])

        y_test[:, f] = model.predict(test_df[features])

        

    df["pred_{}".format(target)] = y_oof

    test_df[target] = y_test.mean(axis=1)

    score = metric(df[df[target].notnull()][target].values, df[df[target].notnull()]["pred_{}".format(target)].values)

    overal_score += w*score

    print(target, np.round(score, 4))

    print()

    

print("Overal score:", np.round(overal_score, 4))
sub_df = pd.melt(test_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")

sub_df["Id"] = sub_df["Id"].astype("str") + "_" +  sub_df["variable"].astype("str")



sub_df = sub_df.drop("variable", axis=1).sort_values("Id")

assert sub_df.shape[0] == test_df.shape[0]*5

sub_df.head(10)
sub_df.to_csv("submission.csv", index=False)