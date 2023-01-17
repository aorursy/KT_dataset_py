

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")

train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")

sample_submission = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv")
sample_submission.head(5)
test.head(5)
train.head(10)
test.describe()
train.describe()
train.shape, test.shape
test.isna().sum()
train.isna().sum()
train.dtypes.sample(20)
test = test.fillna(0)

train = train.fillna(0)
trainP = train.hist(figsize = (20,20))
testP = test.hist(figsize = (20,20))
import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



plt.figure(figsize=(16,6))

features = train.columns.values[1:80]

plt.title("Distribution")

sns.distplot(train[features].mean(axis=1),color="red", hist = True, label='train')

sns.distplot(test[features].mean(axis=1),color="purple", hist = True, label='test')

plt.tight_layout()

plt.legend()

plt.show()
train.columns
test.shape, train.shape
one_hot_encoded_train = pd.get_dummies(train)

one_hot_encoded_test = pd.get_dummies(test)

final_train, final_test = one_hot_encoded_train.align(one_hot_encoded_test, join='left', axis=1)
final_train
final_test
final_train.shape, final_test.shape
corr = final_train.corr()

corr.head()
sns.heatmap(corr)
columns = np.full((corr.shape[0]), True, dtype=bool)

for i in range(corr.shape[0]):

    for j in range(i+1, corr.shape[0]):

        if corr.iloc[i,j] >= 0.7:

            if columns[j]:

                columns[j] = False
selected_columns = final_train.columns[columns]

selected_columns.shape
final_train = final_train[selected_columns]

final_test = final_test[selected_columns]
X = final_train.iloc[:,:]

y = train.iloc[:,-1].values
X
y
X.shape, final_test.shape
from xgboost import XGBRegressor



xg = XGBRegressor()

xg.fit(X, y, verbose=False)
XGBReg_pred = xg.predict(final_test)
sub_df = pd.DataFrame({"Id": test["Id"],

                       "SalePrice": pd.Series(XGBReg_pred)

                      })

sub_df.to_csv("XGBRegressor_submission.csv", index=False)
from sklearn.ensemble import RandomForestRegressor
final_test.isna().sum().sort_values(ascending=False)
final_test = final_test.fillna(0)
rf = RandomForestRegressor()

rf.fit(X, y)
RandomForest_pred = rf.predict(final_test)
sub_df = pd.DataFrame({"Id": test["Id"],

                       "SalePrice": pd.Series(RandomForest_pred)

                      })

sub_df.to_csv("RandomForestRegressor_submission.csv", index=False)
from sklearn.svm import SVR
svrfit = SVR(C=1.0, epsilon=0.2)
svrfit.fit(X,y)
svrpred = svrfit.predict(final_test)
sub_df = pd.DataFrame({"Id": test["Id"],

                       "SalePrice": pd.Series(svrpred)

                      })

sub_df.to_csv("SVR_submission.csv", index=False)
from sklearn.tree import DecisionTreeRegressor 
dtregressor = DecisionTreeRegressor(random_state = 0)
dtregressor.fit(X, y)
dtr_pred = dtregressor.predict(final_test)
sub_df = pd.DataFrame({"Id": test["Id"],

                       "SalePrice": pd.Series(dtr_pred)

                      })

sub_df.to_csv("DecisionTreeRegressor_submission.csv", index=False)