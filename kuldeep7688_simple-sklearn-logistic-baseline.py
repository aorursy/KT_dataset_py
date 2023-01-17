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
from tqdm import tqdm

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import log_loss

from sklearn.preprocessing import StandardScaler

import random

random.seed(123)
TRAIN_FEATURES_FILE_NAME = "/kaggle/input/lish-moa/train_features.csv"

TRAIN_TARGETS_SCORED = "/kaggle/input/lish-moa/train_targets_scored.csv"
train_features_df = pd.read_csv(TRAIN_FEATURES_FILE_NAME)

print(train_features_df.shape)

train_features_df.head()
train_targets_scored_df = pd.read_csv(TRAIN_TARGETS_SCORED)

print(train_targets_scored_df.shape)

train_targets_scored_df.head()
train_idxs, val_idxs = train_test_split([i for i in range(0, 23814)], test_size=0.2)

len(train_idxs), len(val_idxs)
train_features = train_features_df.iloc[train_idxs, 1:]

val_features = train_features_df.iloc[val_idxs, 1:]



train_labels = train_targets_scored_df.iloc[train_idxs, 1:]

val_labels = train_targets_scored_df.iloc[val_idxs, 1:]



len(train_features), len(val_features), len(train_labels), len(val_labels)
# handling cp_type

cp_type_dict = {

    "trt_cp": 0,

    "ctl_vehicle": 1

}



train_features["cp_type"] = train_features.cp_type.map(cp_type_dict)

val_features["cp_type"] = val_features.cp_type.map(cp_type_dict)
# handling cp_dose

cp_dose_dict = {

    "D1": 1,

    "D2": 2

}



train_features["cp_dose"] = train_features.cp_dose.map(cp_dose_dict)

val_features["cp_dose"] = val_features.cp_dose.map(cp_dose_dict)
# handling cp_time

cp_time_dict = {

    24: 1,

    48: 2,

    72: 3

}



train_features["cp_time"] = train_features.cp_time.map(cp_time_dict)

val_features["cp_time"] = val_features.cp_time.map(cp_time_dict)
# standard scaling continuous columns

continuous_columns = [col for col in list(train_features_df.columns) if col not in ["cp_type", "cp_dose", "cp_time", "sig_id"]]

print("Number of continuous columns are {}".format(len(continuous_columns)))



train_continuous_columns_df = train_features[continuous_columns].copy()



standard_scaler_object = StandardScaler().fit(train_continuous_columns_df.values)



train_continuous_columns_df = standard_scaler_object.transform(train_continuous_columns_df.values)



val_continuous_columns_df = val_features[continuous_columns].copy()

val_continuous_columns_df = standard_scaler_object.transform(val_continuous_columns_df.values)



# assigning scaled values to original data

train_features[continuous_columns] = train_continuous_columns_df

val_features[continuous_columns] = val_continuous_columns_df
all_categories = list(train_labels.columns)

len(all_categories)
model_dict = {}
for category in tqdm(all_categories):

    # Training logistic regression model on train data

    logistic_model = LogisticRegression(max_iter=5000)

    logistic_model.fit(train_features, train_labels[category])

    

    # saving model

    model_dict[category] = logistic_model 
def calculate_score(models_dict, val_features, val_labels, all_categories):

    log_loss_per_category = []

    for category in tqdm(all_categories):

        # predicting using logistic regression model from the models_dict

        logistic_model = models_dict[category]

        category_probabs = logistic_model.predict_proba(val_features)

        

        log_loss_per_category.append(

            log_loss(val_labels[category], category_probabs, labels=[0, 1])

        )

    

    return float(sum(log_loss_per_category)) / len(log_loss_per_category)
val_score = calculate_score(model_dict, val_features, val_labels, all_categories)

print("Validation score on validation set is {}".format(val_score))
test_features_df = pd.read_csv("/kaggle/input/lish-moa/test_features.csv")

print(test_features_df.shape)

test_features_df.head()
test_features_df["cp_type"] = test_features_df.cp_type.map(cp_type_dict)

test_features_df["cp_dose"] = test_features_df.cp_dose.map(cp_dose_dict)

test_features_df["cp_time"] = test_features_df.cp_time.map(cp_time_dict)
test_continuous_columns_df = test_features_df[continuous_columns].copy()

test_continuous_columns_df = standard_scaler_object.transform(test_continuous_columns_df.values)

test_features_df[continuous_columns] = test_continuous_columns_df
predictions_df = pd.DataFrame()

predictions_df["sig_id"] = test_features_df.sig_id

for category in tqdm(all_categories):

    predictions_df[category] = model_dict[category].predict_proba(test_features_df.iloc[:, 1:])[:, 1]
predictions_df = predictions_df.round(1)

predictions_df.head()
predictions_df.shape
all_ctl_test_ids = list(test_features_df[test_features_df.cp_type == 1].sig_id)

print(len(all_ctl_test_ids))

for id_ in tqdm(all_ctl_test_ids):

    predictions_df.loc[predictions_df.sig_id == id_, all_categories] = 0.0
predictions_df.head()
predictions_df.to_csv("submission.csv", index=False)