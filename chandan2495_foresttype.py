# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
train_df = pd.read_csv("/kaggle/input/learn-together/train.csv")

test_df = pd.read_csv("/kaggle/input/learn-together/test.csv")
print(f"Number of columns {train_df.shape[1]}")

print(f"Training data size {train_df.shape[0]}")

print(f"Test data size {test_df.shape[0]}")
train_df.head()
train_df.columns
categorical_cols = train_df.select_dtypes(include="object").columns

print(categorical_cols)
# check constant columns

constant_cols = [col for col in train_df.columns if train_df[col].nunique()==1]

print(constant_cols)
# get columns with a certain threshold

std_threshold = 0.1

quasi_constant_cols = [col for col in train_df.columns if train_df[col].std()<std_threshold] + ["Id"]

print(f"Number of columns with standard deviation less than {std_threshold} are {len(quasi_constant_cols)}")

print(quasi_constant_cols)
# check the Wilderness_Area2

train_df["Wilderness_Area2"].value_counts()
train_df = train_df.sample(frac=1)
# lets drop all the quasi-constant columns and build a random forest classifier

X = train_df.drop(quasi_constant_cols + ["Cover_Type", "Wilderness_Area2"], axis="columns")

y = train_df["Cover_Type"]

test_data = test_df.drop(quasi_constant_cols + ["Wilderness_Area2"], axis="columns")
X.head()
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=42, test_size=0.2)
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=4)

model.fit(X_train, y_train)

print(model.score(X_valid, y_valid))
# from sklearn.ensemble import GradientBoostingClassifier

# model = GradientBoostingClassifier(n_estimators=400, random_state=42)

# model.fit(X_train, y_train)

# print(model.score(X_valid, y_valid))
# from catboost import CatBoostClassifier

# model = CatBoostClassifier(

#     iterations=700,

#     learning_rate=0.1,

#     random_state=42,

#     loss_function='MultiClass'

# )

# model.fit(

#     X_train, y_train,

# #     cat_features=cat_features,

#     eval_set=(X_valid, y_valid),

#     verbose=False,

#     plot=True

# )

# print(model.score(X_valid, y_valid))
# make a submission

y_pred = model.predict(test_data)
import warnings

warnings.filterwarnings("ignore")
# submission_df = pd.DataFrame({"Id": test_df["Id"], "Cover_Type": np.reshape(y_pred, -1).astype(int)})

# submission_df.to_csv("submission.csv", index=False)
submission_df = pd.DataFrame({"Id": test_df["Id"], "Cover_Type": y_pred})

submission_df.to_csv("submission.csv", index=False)
# submission_df.head()
# rfc.feature_importances_
# np.reshape(y_pred, -1).astype(int)