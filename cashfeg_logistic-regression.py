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
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
train_users_game1_df = pd.read_csv("../input/ds2019uec-task2/train_users_game1.csv")

train_users_game2_df = pd.read_csv("../input/ds2019uec-task2/train_users_game2.csv")

test_users_game1_df = pd.read_csv("../input/ds2019uec-task2/test_users_game1.csv")
train_users_game1_df.head()
train_users_game2_df.head()
test_users_game1_df.head()
train_user_encoder = LabelEncoder()

train_user_encoder.fit(np.concatenate([train_users_game1_df["user_id"], train_users_game2_df["user_id"]]))

print(train_user_encoder.classes_.shape)

game1_encoder = LabelEncoder()

game1_encoder.fit(np.concatenate([train_users_game1_df["game_title"], test_users_game1_df["game_title"]]))

print(game1_encoder.classes_.shape)
X = np.zeros((9393, 4155))

user_vec = train_user_encoder.transform(train_users_game1_df["user_id"])

game_vec = game1_encoder.transform(train_users_game1_df["game_title"])



for i, j in zip(user_vec, game_vec):

    X[i, j] = 1
Y_mat = np.zeros((9393, 1000))

user_vec = train_user_encoder.transform(train_users_game2_df["user_id"])

game_vec = train_users_game2_df["predict_game_id"]



for i, j in zip(user_vec, game_vec):

    Y_mat[i, j] = 1
lr_899 = LogisticRegression()

lr_899.fit(X, Y_mat[:, 899])
pred_train_899 = lr_899.predict_proba(X)[:, 1]
pred_train_899.mean()
# （コンペ評価指標の本題とは外れるが）AUCを見てみる

from sklearn.metrics import roc_auc_score

roc_auc_score(Y_mat[:, 899], pred_train_899)
test_users = pd.read_csv("../input/ds2019uec-task2/test_user_ids.csv")["user_id"]
test_user_encoder = LabelEncoder()

test_user_encoder.fit(test_users)

print(test_user_encoder.classes_.shape)
X_test = np.zeros((3000, 4155))

user_vec_test = test_user_encoder.transform(test_users_game1_df["user_id"])

game_vec = game1_encoder.transform(test_users_game1_df["game_title"])



for i, j in zip(user_vec_test, game_vec):

    X_test[i, j] = 1
pred_test_899 = lr_899.predict_proba(X_test)[:, 1]
pred_test_899.mean()
pred_test_899.shape
pred_test_mat = np.zeros((3000, 1000))

pred_test_mat[:, 899] = pred_test_899
res_list = [""] * 3000

for i in range(3000):

    res_list[i] = " ".join([str(i) for i in np.argsort(-pred_test_mat[i, :])[:10]])



sample_submission_df = pd.read_csv("../input/ds2019uec-task2/sample_submission.csv")

sub_df = sample_submission_df.copy()

sub_df["purchased_games"] = res_list
sub_df.head()
sub_df.to_csv("enjoy_your_submission.csv", index=False)