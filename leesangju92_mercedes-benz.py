# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import LabelEncoder



# Plotting 관련 import

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
test = pd.read_csv("../input/test.csv")

train = pd.read_csv("../input/train.csv")

sample_submission = pd.read_csv("../input/sample_submission.csv")
test.head()
train.head()
sample_submission.head()
train.describe(include="all")
print(test.shape, train.shape)
print(train.isnull().values.any(), test.isnull().values.any())
# 카테고리 features를 label-encoding

cat_columns = ["X0", "X1", "X2", "X3", "X4", "X5", "X6", "X8"]



le0 = LabelEncoder()

le0.fit(list(train.X0)+list(test.X0))

train.X0 = le0.transform(train.X0)

test.X0 = le0.transform(test.X0)



le1 = LabelEncoder()

le1.fit(list(train.X1)+list(test.X1))

train.X1 = le1.transform(train.X1)

test.X1 = le1.transform(test.X1)



le2 = LabelEncoder()

le2.fit(list(train.X2)+list(test.X2))

train.X2 = le2.transform(train.X2)

test.X2 = le2.transform(test.X2)



le3 = LabelEncoder()

le3.fit(list(train.X3)+list(test.X3))

train.X3 = le3.transform(train.X3)

test.X3 = le3.transform(test.X3)





le4 = LabelEncoder()

le4.fit(list(train.X4)+list(test.X4))

train.X4 = le4.transform(train.X4)

test.X4 = le4.transform(test.X4)





le5 = LabelEncoder()

le5.fit(list(train.X5)+list(test.X5))

train.X5 = le5.transform(train.X5)

test.X5 = le5.transform(test.X5)



le6 = LabelEncoder()

le6.fit(list(train.X6)+list(test.X6))

train.X6 = le6.transform(train.X6)

test.X6 = le6.transform(test.X6)



le8 = LabelEncoder()

le8.fit(list(train.X8)+list(test.X8))

train.X8 = le8.transform(train.X8)

test.X8 = le8.transform(test.X8)



# X0_list = train.X0.unique().tolist()

# X1_list = train.X1.unique().tolist()

# X2_list = train.X2.unique().tolist()

# X3_list = train.X3.unique().tolist()

# X4_list = train.X4.unique().tolist()

# X5_list = train.X5.unique().tolist()

# X6_list = train.X6.unique().tolist()

# X8_list = train.X8.unique().tolist()
# train에서 값이 constant인 column은 아무런 정보도 없을 수 없다.. 찾아서 제거하기~



constant_columns = []

for i in range(386):

    if i not in [7, 9, 25, 72, 121, 149, 188, 193, 303, 381]:

        column_name = "X" + str(i)

        if train[column_name].mean() in [0, 1]:

            constant_columns.append(column_name)

            print(column_name, train[column_name].mean())



print("------------")



# 이건 그냥~

for i in range(386):

    if i not in [7, 9, 25, 72, 121, 149, 188, 193, 303, 381]:

        column_name = "X" + str(i)

        if test[column_name].mean() in [0, 1]: 

            print(column_name, test[column_name].mean())
constant_columns
train.drop(constant_columns, axis=1, inplace=True)

test.drop(constant_columns, axis=1, inplace=True)
train_y = pd.DataFrame(train["y"])

train_x = train.drop(["y"], axis=1)
train.head()
test.head()
for col_name in cat_columns:

    print(col_name, train[col_name].nunique())
f = plt.figure(figsize=(60, 60))

plt.matshow(train.corr(), fignum=f.number)

plt.xticks(range(train.shape[1]), train.columns, fontsize=10, rotation=45)

plt.yticks(range(train.shape[1]), train.columns, fontsize=10)

cb = plt.colorbar()

cb.ax.tick_params(labelsize=14)

plt.title('Correlation Matrix', fontsize=16);
plt.figure(figsize=(30, 30))

for col_idx in range(len(cat_columns)):

    plt.subplot(4,4,col_idx+1).set_title(cat_columns[col_idx])

    plt.hist(train_x[cat_columns[col_idx]], bins=train_x[cat_columns[col_idx]].nunique())

    plt.tight_layout()

plt.show()
train_x.X4.value_counts() # 이것도 상수 취급?
# deleted_columns = [ 11, 93, 107, 233, 235, 268, 289, 290, 293, 297, 330, 347, 7, 9, 25, 72, 121, 149, 188, 193, 303, 381]
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
XGBR = XGBRegressor()

XGBR.fit(train_x, train_y, verbose=False)

predicts = XGBR.predict(test)

submission = pd.DataFrame({'ID': test.ID, 'y': predicts})

submission.to_csv('submission.csv', index=False)
# predicts

# learning_rate=0.4, max_depth=3, n_estimators=100, max_delta_step=0.9, colsample_bytree=1, subsample=0.75
# submission
# from lightgbm import LGBMClassifier

# model = LGBMClassifier(objective="multiclass", num_class=39, max_bin = 465, max_delta_step = 0.9, learning_rate=0.4, num_leaves = 42, n_estimators=100, feature_fraction=0.75, subsample=0.75)

# model.fit(train_x, train_y, categorical_feature=['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8'])

# preds = model.predict_proba(test)

# submission = pd.DataFrame({'Id': test.Id, 'y': preds})

# submission.to_csv('submission.csv', index_labe=False)