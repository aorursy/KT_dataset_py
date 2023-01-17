import numpy as np 

import pandas as pd 

import lightgbm as lgb

from sklearn.model_selection import GroupKFold

import category_encoders as ce

import matplotlib.pyplot as plt



SEED = 19

np.random.seed(SEED)
train = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/train.csv")

train.columns = [col.lower() for col in train.columns]

train.info()
test = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/test.csv")

test.columns = [col.lower() for col in test.columns]

test.info()
submission = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv")

submission.columns = [col.lower() for col in submission.columns]

submission.info()
train.sort_values(["patient","weeks"], inplace=True)

train["base_week"] = train.groupby("patient")["weeks"].transform(lambda x: x.iloc[0])

train["base_fvc"] = train.groupby("patient")["fvc"].transform(lambda x: x.iloc[0])

train["base_percent"] = train.groupby("patient")["percent"].transform(lambda x: x.iloc[0])

train
train.info()
input_features = ["weeks", "age", "sex", "smokingstatus", "base_week", "base_fvc", "base_percent"]

categorical_features = ["sex","smokingstatus"]



group_col = "patient"

n_folds = 10

target = "fvc"



# left and right quantiles used to estimate confidence

alpha_qleft = 0.2

alpha_qright = 0.8
encoder = ce.OrdinalEncoder(cols=categorical_features, handle_unknown='impute')

encoder.fit(train.loc[:, categorical_features])

train.loc[:, categorical_features] = encoder.transform(train.loc[:, categorical_features])
# model hyperparams

model_params = {

    'objective':'quantile',

    'metric':'quantile',

    'max_bin': 127,

    'num_leaves': 7,

    'min_data_in_leaf': 15,

    'learning_rate': 0.025,

    'feature_fraction':0.8,

    'bagging_fraction':0.8,

    'bagging_freq':1,

    'seed':SEED,

}
# columns where oof predictions will be saved

train["pred_qleft"] = None

train["pred_expected"] = None

train["pred_qright"] = None
model_params["alpha"] = alpha_qleft

gkf = GroupKFold(n_splits=n_folds)

all_models_qleft = list()



for fold,(train_idx, valid_idx) in enumerate(gkf.split(train, train[target], train[group_col])):

    

    train_data = train.loc[train_idx, :]

    valid_data = train.loc[valid_idx, :]

    

    train_df_kwargs = {

        "data":train_data.loc[:, input_features],

        "categorical_feature":categorical_features,

        "free_raw_data":False,

        "label":train_data.loc[:, target].values

    }

    _train_data = lgb.Dataset(**train_df_kwargs)



    valid_df_kwargs = {

        "data":valid_data.loc[:, input_features],

        "categorical_feature":categorical_features,

        "free_raw_data":False,

        "label":valid_data.loc[:, target].values

    }

    _valid_data = lgb.Dataset(**valid_df_kwargs)

    

    training_kwargs = {

        "train_set": _train_data,

        "valid_sets": _valid_data,

        "early_stopping_rounds": 250,

        "num_boost_round": 1000,

        "params": model_params,

        "verbose_eval":50,

    }

    model = lgb.train(**training_kwargs)

    all_models_qleft.append(model)

    

    ## oof predictions

    train.loc[valid_idx,"pred_qleft"] = model.predict(train.loc[valid_idx, input_features])
lgb.plot_importance(all_models_qleft[0], importance_type='gain', figsize=(10,8))

plt.show()



lgb.plot_importance(all_models_qleft[0], importance_type='split', figsize=(10,8))

plt.show()
model_params["alpha"] = 0.5

gkf = GroupKFold(n_splits=n_folds)

all_models_expected = list()



for fold,(train_idx, valid_idx) in enumerate(gkf.split(train, train[target], train[group_col])):

    

    train_data = train.loc[train_idx, :]

    valid_data = train.loc[valid_idx, :]

    

    train_df_kwargs = {

        "data":train_data.loc[:, input_features],

        "categorical_feature":categorical_features,

        "free_raw_data":False,

        "label":train_data.loc[:, target].values

    }

    _train_data = lgb.Dataset(**train_df_kwargs)



    valid_df_kwargs = {

        "data":valid_data.loc[:, input_features],

        "categorical_feature":categorical_features,

        "free_raw_data":False,

        "label":valid_data.loc[:, target].values

    }

    _valid_data = lgb.Dataset(**valid_df_kwargs)

    

    training_kwargs = {

        "train_set": _train_data,

        "valid_sets": _valid_data,

        "early_stopping_rounds": 250,

        "num_boost_round": 1000,

        "params": model_params,

        "verbose_eval":50,

    }

    model = lgb.train(**training_kwargs)

    all_models_expected.append(model)



    ## oof predictions

    train.loc[valid_idx,"pred_expected"] = model.predict(train.loc[valid_idx, input_features])
lgb.plot_importance(all_models_expected[0], importance_type='gain', figsize=(10,8))

plt.show()



lgb.plot_importance(all_models_expected[0], importance_type='split', figsize=(10,8))

plt.show()
model_params["alpha"] = alpha_qright

gkf = GroupKFold(n_splits=n_folds)

all_models_qright = list()



for fold,(train_idx, valid_idx) in enumerate(gkf.split(train, train[target], train[group_col])):

    

    train_data = train.loc[train_idx, :]

    valid_data = train.loc[valid_idx, :]

    

    train_df_kwargs = {

        "data":train_data.loc[:, input_features],

        "categorical_feature":categorical_features,

        "free_raw_data":False,

        "label":train_data.loc[:, target].values

    }

    _train_data = lgb.Dataset(**train_df_kwargs)



    valid_df_kwargs = {

        "data":valid_data.loc[:, input_features],

        "categorical_feature":categorical_features,

        "free_raw_data":False,

        "label":valid_data.loc[:, target].values

    }

    _valid_data = lgb.Dataset(**valid_df_kwargs)

    

    training_kwargs = {

        "train_set": _train_data,

        "valid_sets": _valid_data,

        "early_stopping_rounds": 250,

        "num_boost_round": 1000,

        "params": model_params,

        "verbose_eval":50,

    }

    model = lgb.train(**training_kwargs)

    all_models_qright.append(model)



    ## oof predictions

    train.loc[valid_idx,"pred_qright"] = model.predict(train.loc[valid_idx, input_features])
lgb.plot_importance(all_models_qright[0], importance_type='gain', figsize=(10,8))

plt.show()



lgb.plot_importance(all_models_qright[0], importance_type='split', figsize=(10,8))

plt.show()
def error_metric(ytrue, ypred, confidence):

    sig_clipped = (np.clip(confidence, a_min=70., a_max=None)).astype(float)

    delta = (np.clip(np.abs(ytrue - ypred), a_min=None, a_max=1000)).astype(float)

    metric = -(np.sqrt(2)*delta / sig_clipped) - np.log(np.sqrt(2)*sig_clipped)

    return np.mean(metric)
ytrue = train.fvc.values

ypred = train.pred_expected.values

confidence = train.pred_qright.values - train.pred_qleft.values

print(f"CV error: {error_metric(ytrue, ypred, confidence)}")
test.rename({"weeks":"base_week", "fvc":"base_fvc", "percent":"base_percent"}, axis=1, inplace=True)

test.loc[:, categorical_features] = encoder.transform(test.loc[:, categorical_features])

test
submission["patient"] = submission.patient_week.apply(lambda x: x.split("_")[0])

submission["weeks"] = submission.patient_week.apply(lambda x: int(x.split("_")[1]))

submission
predict_dataframe = (submission

                     .merge(test, how="left", on=["patient"])

                     .loc[:, input_features])

predict_dataframe
predictions_qleft = list()

for model in all_models_qleft:

    _pred = model.predict(predict_dataframe)

    predictions_qleft.append(_pred)

    

pred_qleft = np.mean(predictions_qleft, axis=0)
predictions_expected = list()

for model in all_models_expected:

    _pred = model.predict(predict_dataframe)

    predictions_expected.append(_pred)

    

pred_expected = np.mean(predictions_expected, axis=0)
predictions_qright = list()

for model in all_models_qright:

    _pred = model.predict(predict_dataframe)

    predictions_qright.append(_pred)

    

pred_qright = np.mean(predictions_qright, axis=0)
submission = pd.read_csv("/kaggle/input/osic-pulmonary-fibrosis-progression/sample_submission.csv")

submission.loc[:, "FVC"] = pred_expected

submission.loc[:, "Confidence"] = pred_qright - pred_qleft

submission.to_csv("submission.csv", index=False)
submission.FVC.describe()
submission.Confidence.describe()