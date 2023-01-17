import numpy as np

import pandas as pd

import matplotlib.pylab as plt

%matplotlib inline
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
col_names = np.array(train.columns)

col_type = [train[col_name].dtype for col_name in col_names]

feature_type = pd.DataFrame({"Column": col_names, "Type": col_type})

feature_type.head()
feature_type.loc[feature_type["Type"]=="int64", "Column"].values
feature_type.loc[feature_type["Type"]=="float64", "Column"].values
feature_type.loc[feature_type["Type"]=="object", "Column"].values
# Seperate x, y

all_data = pd.concat((train, test)).reset_index(drop=True)

ntrain = train.shape[0]

ntest = test.shape[0]



from sklearn.preprocessing import LabelEncoder

for col_name in all_data:

    if all_data[col_name].dtype == np.int64:

        all_data[col_name].fillna(-1, inplace=True)

    elif all_data[col_name].dtype == np.float64:

        all_data[col_name].fillna(-1, inplace=True)        

    else:

        all_data[col_name].fillna("None", inplace=True)

        encoder = LabelEncoder()

        encoder.fit(all_data[col_name].values)

        all_data[col_name] = encoder.transform(all_data[col_name].values)



all_data.head()
train = all_data.loc[:ntrain,:]

train_x = train.drop(["SalePrice", "Id"], axis=1)

train_y = train["SalePrice"]

test = all_data.loc[ntrain:, :]

test_x = test.drop(["Id", "SalePrice"], axis=1)
# Cross validation

import xgboost as xgb

param = {'max_depth':6, 'eta':0.1, 'silent':1, 'objective':'reg:linear'}

num_round = 200

dtrain = xgb.DMatrix(train_x, label=train_y)



cv = xgb.cv(param, dtrain, num_round, nfold=5, early_stopping_rounds=3,

       metrics={'rmse'}, seed=0)

cv.tail()
model = xgb.train(param, dtrain, num_boost_round=70)
dtest = xgb.DMatrix(test_x)

prediction = model.predict(dtest)
submission = pd.DataFrame()

submission["Id"] = test["Id"]

submission["SalePrice"] = prediction
submission.head()
submission.to_csv("submission.csv", index=False)
score_dict = model.get_score()

importance = pd.DataFrame({"Feature": list(score_dict.keys()), "Importance":list(score_dict.values())})

importance.sort_values(by=["Importance"], inplace=True, ascending=False)

importance.plot(x="Feature", y="Importance", kind="bar", figsize=(15,15))
importance