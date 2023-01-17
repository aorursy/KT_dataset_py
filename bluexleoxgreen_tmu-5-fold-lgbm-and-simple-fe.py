import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from tqdm import tqdm_notebook as tqdm

pd.set_option('display.max_columns', 50)



import warnings

warnings.filterwarnings('ignore')



import os

for dirname, _, filenames in os.walk('../input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
train_df = pd.read_csv("../input/tmu-inclass-competition/train.csv")

test_df = pd.read_csv("../input/tmu-inclass-competition/test.csv")

sub_df = pd.read_csv("../input/tmu-inclass-competition/sample_submission.csv")
print(f"{len(train_df)} {len(test_df)}")
from sklearn.preprocessing import LabelEncoder



cat_list = ["jurisdiction_names", "country_code", "smart_location", "property_type", "host_id", "host_response_time", "room_type"]



def preprocess(train_df, test_df):

    new_df = pd.concat([train_df, test_df]).reset_index(drop=True)



    d = {}

    for s in new_df["calendar_updated"].value_counts().index:

        if s == "today":

            d[s] = 0

        elif s == "yesterday":

            d[s] = 1

        elif s == "a week ago" or s == "1 week ago":

            d[s] = 7

        elif s == "never":

            d[s] = 9999

        elif s[-len("months ago"):] == "months ago":

            d[s] = int(s[:-len("months ago")]) * 30

        elif s[-len("weeks ago"):] == "weeks ago":

            d[s] = int(s[:-len("weeks ago")]) * 7

        elif s[-len("days ago"):] == "days ago":

            d[s] = int(s[:-len("days ago")])

        else:

            print(s)

            print("Error")

            break



    oldest = min(pd.to_datetime(new_df["host_since"]))

    newest = max(pd.to_datetime(new_df["host_since"]))

    dt = newest - oldest



    processed_train_df = train_df.select_dtypes("number").drop("listing_id", axis=1)

    processed_train_df["host_since"] = pd.to_datetime(train_df["host_since"])

    processed_train_df["host_since"] = processed_train_df["host_since"].apply(lambda x: (x - oldest)/dt * 100)

    processed_train_df["calendar_updated"] = train_df["calendar_updated"].apply(lambda x: d[x])

    processed_train_df["host_response_rate"] = train_df["host_response_rate"].fillna("0%").apply(lambda x: int(x[:-1]))

    processed_train_df = pd.concat([processed_train_df, pd.get_dummies(train_df["bed_type"])], axis=1, join='inner')

    processed_train_df = pd.concat([processed_train_df, pd.get_dummies(train_df["cancellation_policy"])], axis=1, join='inner')

    



    processed_test_df = test_df.select_dtypes("number").drop("listing_id", axis=1)

    processed_test_df["host_since"] = pd.to_datetime(test_df["host_since"])

    processed_test_df["host_since"] = processed_test_df["host_since"].apply(lambda x: (x - oldest)/dt * 100)

    processed_test_df["calendar_updated"] = test_df["calendar_updated"].apply(lambda x: d[x])

    processed_test_df["host_response_rate"] = test_df["host_response_rate"].fillna("0%").apply(lambda x: int(x[:-1]))

    processed_test_df = pd.concat([processed_test_df, pd.get_dummies(test_df["bed_type"])], axis=1, join='inner')

    processed_test_df = pd.concat([processed_test_df, pd.get_dummies(test_df["cancellation_policy"])], axis=1, join='inner')



    # one-hot

    for col in ["host_is_superhost", "host_has_profile_pic", "host_identity_verified", "is_location_exact", "has_availability", "requires_license"\

                , "instant_bookable", "is_business_travel_ready", "require_guest_profile_picture", "require_guest_phone_verification"]:

        processed_train_df[col] = train_df[col].apply(lambda x: 1 if x == "t" else 0)

        processed_test_df[col] = test_df[col].apply(lambda x: 1 if x == "t" else 0)



    train_df["host_verifications"] = train_df["host_verifications"].apply(lambda x: x.replace("]", "").replace("[", "").replace("'", "").replace(",", ""))

    test_df["host_verifications"] = test_df["host_verifications"].apply(lambda x: x.replace("]", "").replace("[", "").replace("'", "").replace(",", ""))



    vers = ["email", "phone", "facebook", "google", "weibo", "sent_id" , "reviews", "kba", "jumio", "government_id", "offline_government_id", "selfie", "identity_manual", "sesame", "sesame_offline", "work_email"]

    for v in vers:

        processed_train_df[v] = train_df["host_verifications"].apply(lambda x: 1 if v in x.split() else 0)

        processed_test_df[v] = test_df["host_verifications"].apply(lambda x: 1 if v in x.split() else 0)



    # label encoding

    for col in cat_list:

        le = LabelEncoder()

        train_df[col] = train_df[col].fillna("NAN")

        test_df[col] = test_df[col].fillna("NAN")

        le = le.fit(pd.concat([train_df[col], test_df[col]]))

        processed_train_df[col] = le.transform(train_df[col])

        processed_test_df[col] = le.transform(test_df[col])



    # feature engineering

    processed_train_df["col1"] = processed_train_df["bedrooms"] / processed_train_df["accommodates"]

    processed_train_df["col2"] = processed_train_df["bathrooms"] / processed_train_df["accommodates"]

    processed_train_df["col3"] = processed_train_df["beds"] / processed_train_df["bedrooms"]

    processed_train_df["col4"] = processed_train_df["bedrooms"] + processed_train_df["bathrooms"]

    processed_train_df["col5"] = processed_train_df["bedrooms"] + processed_train_df["bathrooms"] + processed_train_df["accommodates"] + processed_train_df["beds"]

    processed_train_df["col6"] = processed_train_df["bedrooms"] / processed_train_df.groupby(['host_id'])['bedrooms'].transform('std')

    processed_train_df["col7"] = processed_train_df["bedrooms"] / processed_train_df.groupby(['host_id'])['bedrooms'].transform('mean')

    processed_train_df["col8"] = processed_train_df["bedrooms"] / processed_train_df.groupby(['property_type'])['bedrooms'].transform('std')

    processed_train_df["col9"] = processed_train_df["bedrooms"] / processed_train_df.groupby(['property_type'])['bedrooms'].transform('mean')

    processed_train_df["col10"] = processed_train_df["bedrooms"] / processed_train_df.groupby(['smart_location'])['bedrooms'].transform('std')

    processed_train_df["col11"] = processed_train_df["bedrooms"] / processed_train_df.groupby(['smart_location'])['bedrooms'].transform('mean')

    processed_train_df["col12"] = processed_train_df["bedrooms"] / processed_train_df.groupby(['room_type'])['bedrooms'].transform('std')

    processed_train_df["col13"] = processed_train_df["bedrooms"] / processed_train_df.groupby(['room_type'])['bedrooms'].transform('mean')

    processed_train_df["col14"] = processed_train_df["bathrooms"] / processed_train_df.groupby(['host_id'])['bathrooms'].transform('std')

    processed_train_df["col15"] = processed_train_df["bathrooms"] / processed_train_df.groupby(['host_id'])['bathrooms'].transform('mean')

    processed_train_df["col16"] = processed_train_df["bathrooms"] / processed_train_df.groupby(['property_type'])['bathrooms'].transform('std')

    processed_train_df["col17"] = processed_train_df["bathrooms"] / processed_train_df.groupby(['property_type'])['bathrooms'].transform('mean')

    processed_train_df["col18"] = processed_train_df["bathrooms"] / processed_train_df.groupby(['smart_location'])['bathrooms'].transform('std')

    processed_train_df["col19"] = processed_train_df["bathrooms"] / processed_train_df.groupby(['smart_location'])['bathrooms'].transform('mean')

    processed_train_df["col20"] = processed_train_df["bathrooms"] / processed_train_df.groupby(['room_type'])['bathrooms'].transform('std')

    processed_train_df["col21"] = processed_train_df["bathrooms"] / processed_train_df.groupby(['room_type'])['bathrooms'].transform('mean')

    processed_train_df["col22"] = processed_train_df["accommodates"] / processed_train_df.groupby(['host_id'])['accommodates'].transform('std')

    processed_train_df["col23"] = processed_train_df["accommodates"] / processed_train_df.groupby(['host_id'])['accommodates'].transform('mean')

    processed_train_df["col24"] = processed_train_df["accommodates"] / processed_train_df.groupby(['property_type'])['accommodates'].transform('std')

    processed_train_df["col25"] = processed_train_df["accommodates"] / processed_train_df.groupby(['property_type'])['accommodates'].transform('mean')

    processed_train_df["col26"] = processed_train_df["accommodates"] / processed_train_df.groupby(['smart_location'])['accommodates'].transform('std')

    processed_train_df["col27"] = processed_train_df["accommodates"] / processed_train_df.groupby(['smart_location'])['accommodates'].transform('mean')

    processed_train_df["col28"] = processed_train_df["accommodates"] / processed_train_df.groupby(['room_type'])['accommodates'].transform('std')

    processed_train_df["col29"] = processed_train_df["accommodates"] / processed_train_df.groupby(['room_type'])['accommodates'].transform('mean')

    

    processed_test_df["col1"] = processed_test_df["bedrooms"] / processed_test_df["accommodates"]

    processed_test_df["col2"] = processed_test_df["bathrooms"] / processed_test_df["accommodates"]

    processed_test_df["col3"] = processed_test_df["beds"] / processed_test_df["bedrooms"]

    processed_test_df["col4"] = processed_test_df["bedrooms"] + processed_test_df["bathrooms"]

    processed_test_df["col5"] = processed_test_df["bedrooms"] + processed_test_df["bathrooms"] + processed_test_df["accommodates"] + processed_test_df["beds"]

    processed_test_df["col6"] = processed_test_df["bedrooms"] / processed_test_df.groupby(['host_id'])['bedrooms'].transform('std')

    processed_test_df["col7"] = processed_test_df["bedrooms"] / processed_test_df.groupby(['host_id'])['bedrooms'].transform('mean')

    processed_test_df["col8"] = processed_test_df["bedrooms"] / processed_test_df.groupby(['property_type'])['bedrooms'].transform('std')

    processed_test_df["col9"] = processed_test_df["bedrooms"] / processed_test_df.groupby(['property_type'])['bedrooms'].transform('mean')

    processed_test_df["col10"] = processed_test_df["bedrooms"] / processed_test_df.groupby(['smart_location'])['bedrooms'].transform('std')

    processed_test_df["col11"] = processed_test_df["bedrooms"] / processed_test_df.groupby(['smart_location'])['bedrooms'].transform('mean')

    processed_test_df["col12"] = processed_test_df["bedrooms"] / processed_test_df.groupby(['room_type'])['bedrooms'].transform('std')

    processed_test_df["col13"] = processed_test_df["bedrooms"] / processed_test_df.groupby(['room_type'])['bedrooms'].transform('mean')

    processed_test_df["col14"] = processed_test_df["bathrooms"] / processed_test_df.groupby(['host_id'])['bathrooms'].transform('std')

    processed_test_df["col15"] = processed_test_df["bathrooms"] / processed_test_df.groupby(['host_id'])['bathrooms'].transform('mean')

    processed_test_df["col16"] = processed_test_df["bathrooms"] / processed_test_df.groupby(['property_type'])['bathrooms'].transform('std')

    processed_test_df["col17"] = processed_test_df["bathrooms"] / processed_test_df.groupby(['property_type'])['bathrooms'].transform('mean')

    processed_test_df["col18"] = processed_test_df["bathrooms"] / processed_test_df.groupby(['smart_location'])['bathrooms'].transform('std')

    processed_test_df["col19"] = processed_test_df["bathrooms"] / processed_test_df.groupby(['smart_location'])['bathrooms'].transform('mean')

    processed_test_df["col20"] = processed_test_df["bathrooms"] / processed_test_df.groupby(['room_type'])['bathrooms'].transform('std')

    processed_test_df["col21"] = processed_test_df["bathrooms"] / processed_test_df.groupby(['room_type'])['bathrooms'].transform('mean')

    processed_test_df["col22"] = processed_test_df["accommodates"] / processed_test_df.groupby(['host_id'])['accommodates'].transform('std')

    processed_test_df["col23"] = processed_test_df["accommodates"] / processed_test_df.groupby(['host_id'])['accommodates'].transform('mean')

    processed_test_df["col24"] = processed_test_df["accommodates"] / processed_test_df.groupby(['property_type'])['accommodates'].transform('std')

    processed_test_df["col25"] = processed_test_df["accommodates"] / processed_test_df.groupby(['property_type'])['accommodates'].transform('mean')

    processed_test_df["col26"] = processed_test_df["accommodates"] / processed_test_df.groupby(['smart_location'])['accommodates'].transform('std')

    processed_test_df["col27"] = processed_test_df["accommodates"] / processed_test_df.groupby(['smart_location'])['accommodates'].transform('mean')

    processed_test_df["col28"] = processed_test_df["accommodates"] / processed_test_df.groupby(['room_type'])['accommodates'].transform('std')

    processed_test_df["col29"] = processed_test_df["accommodates"] / processed_test_df.groupby(['room_type'])['accommodates'].transform('mean')



    return processed_train_df, processed_test_df

processed_train_df, processed_test_df = preprocess(train_df, test_df)
processed_train_df.head()
processed_test_df.head()
def rmsle(y_true, y_pred):

    assert len(y_true) == len(y_pred)

    return np.sqrt(np.mean(np.power(np.log1p(y_true + 1) - np.log1p(y_pred + 1), 2)))



def rmsle_lgb(preds, data):

    y_true = np.array(data.get_label())

    result = rmsle(preds, y_true)

    return 'RMSLE', result, False
import lightgbm as lgb

from sklearn.model_selection import KFold



# optimized params using optuna

# https://github.com/pfnet/optuna



params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': 'rmsle',

    'max_depth': 20,

    'max_bin': 200,

    'num_leaves': 97,

    'min_data_in_leaf': 10,

    'learning_rate': 0.0022,

    'feature_fraction': 0.8,

    'bagging_fraction': 0.8,

    'bagging_freq': 10,

    'min_sum_hessian_in_leaf': 10,

    'lambda_l1': 0.01,

    'lambda_l2': 0.01,

    'verbose': 0,

    'metric': 'rmse'

}
y = processed_train_df["price"].values

X = processed_train_df.drop("price", axis=1).values

features = processed_train_df.drop("price", axis=1).columns

X_test = processed_test_df.values



cols = processed_train_df.drop("price", axis=1).columns.values

categorical_cols = cat_list[:]



feature_importance_df = pd.DataFrame()



N = 5

oof = np.zeros(len(X))

test_preds = np.zeros(len(test_df))

kf = KFold(n_splits=N, shuffle=True, random_state=1)

cv_score = []



for fold_, (train_idx, val_idx) in enumerate(kf.split(X), start=1):

    X_train, X_val = X[train_idx], X[val_idx]

    y_train, y_val = y[train_idx], y[val_idx]

    lgb_train = lgb.Dataset(X_train, y_train, feature_name=processed_train_df.drop("price", axis=1).columns.tolist(), categorical_feature=categorical_cols)

    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train, feature_name=processed_train_df.drop("price", axis=1).columns.tolist(), categorical_feature=categorical_cols)

    lgb_reg = lgb.train(params,

            lgb_train,

            num_boost_round=20000,

            valid_sets=lgb_eval,

            early_stopping_rounds=100,

            verbose_eval=500,

            feval = rmsle_lgb)

    y_pred = lgb_reg.predict(X_val, num_iteration=lgb_reg.best_iteration)

    oof[val_idx] = lgb_reg.predict(X_val, num_iteration=lgb_reg.best_iteration)

    test_preds += lgb_reg.predict(X_test, num_iteration=lgb_reg.best_iteration) / N

    cv_score.append(rmsle(y_val, y_pred))



    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = features

    fold_importance_df["importance"] = lgb_reg.feature_importance(importance_type='gain')

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)



    print(f"fold{fold_}: {cv_score[-1]}\n\n")

    sns.residplot(y_pred, y_val, lowess=True, color='g')

    plt.show()

    

print(f"CV RMSLE Score: {sum(cv_score)/len(cv_score)}")
cols = (feature_importance_df[["feature", "importance"]]

        .groupby("feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:40].index)

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]



plt.figure(figsize=(10,10), dpi=200)

sns.barplot(x="importance",

            y="feature",

            data=best_features.sort_values(by="importance",ascending=False))

plt.title('LightGBM Features (avg over folds)')

plt.tight_layout()
sub_df["price"] = test_preds

sub_df.to_csv(f"submission{sum(cv_score)/len(cv_score)}.csv", index=False)
rmsle(y, oof)
# drop outlier index

l = []

for idx, (true, pred) in enumerate(zip(y, oof)):

    l.append([np.power(np.log1p(true + 1) - np.log1p(pred + 1), 2), idx])

l.sort(reverse=True)

l_idx = [x[1] for x in l[:len(l)//20]]

l_idx.sort()

idx = []

j = 0

for i in range(len(l)):

    if i == l_idx[j]:

        if j < len(l_idx) - 1:

            j += 1

    else:

        idx.append(i)

y = processed_train_df["price"].values

X = processed_train_df.drop("price", axis=1).values

X = X[idx]

y = y[idx]
X
N = 5

oof = np.zeros(len(X))

test_preds = np.zeros(len(test_df))

kf = KFold(n_splits=N, shuffle=True, random_state=1)

cv_score = []



for fold_, (train_idx, val_idx) in enumerate(kf.split(X), start=1):

    X_train, X_val = X[train_idx], X[val_idx]

    y_train, y_val = y[train_idx], y[val_idx]

    lgb_train = lgb.Dataset(X_train, y_train, feature_name=processed_train_df.drop("price", axis=1).columns.tolist(), categorical_feature=categorical_cols)

    lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train, feature_name=processed_train_df.drop("price", axis=1).columns.tolist(), categorical_feature=categorical_cols)

    lgb_reg = lgb.train(params,

            lgb_train,

            num_boost_round=20000,

            valid_sets=lgb_eval,

            early_stopping_rounds=100,

            verbose_eval=500,

            feval = rmsle_lgb)

    y_pred = lgb_reg.predict(X_val, num_iteration=lgb_reg.best_iteration)

    oof[val_idx] = lgb_reg.predict(X_val, num_iteration=lgb_reg.best_iteration)

    test_preds += lgb_reg.predict(X_test, num_iteration=lgb_reg.best_iteration) / N

    cv_score.append(rmsle(y_val, y_pred))



    fold_importance_df = pd.DataFrame()

    fold_importance_df["feature"] = features

    fold_importance_df["importance"] = lgb_reg.feature_importance(importance_type='gain')

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)



    print(f"fold{fold_}: {cv_score[-1]}\n\n")

    sns.residplot(y_pred, y_val, lowess=True, color='g')

    plt.show()

    

print(f"CV RMSLE Score: {sum(cv_score)/len(cv_score)}")
sub_df["price"] = test_preds

sub_df.to_csv(f"v2_submission{sum(cv_score)/len(cv_score)}.csv", index=False)
# #pseudo labeling

# X_test = processed_test_df.values

# processed_test_df["price"] = test_preds

# processed_train_df = pd.concat([processed_train_df, processed_test_df]).reset_index(drop=True)

# y = processed_train_df["price"].values

# X = processed_train_df.drop("price", axis=1).values

# features = processed_train_df.drop("price", axis=1).columns





# cols = processed_train_df.drop("price", axis=1).columns.values

# categorical_cols = cat_list[:]



# feature_importance_df = pd.DataFrame()



# N = 5

# test_preds = np.zeros(len(test_df))

# kf = KFold(n_splits=N, shuffle=True, random_state=1)

# cv_score = []



# for fold_, (train_idx, val_idx) in enumerate(kf.split(X), start=1):

#     X_train, X_val = X[train_idx], X[val_idx]

#     y_train, y_val = y[train_idx], y[val_idx]

#     lgb_train = lgb.Dataset(X_train, y_train, feature_name=processed_train_df.drop("price", axis=1).columns.tolist(), categorical_feature=categorical_cols)

#     lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train, feature_name=processed_train_df.drop("price", axis=1).columns.tolist(), categorical_feature=categorical_cols)

#     lgb_reg = lgb.train(params,

#             lgb_train,

#             num_boost_round=10000,

#             valid_sets=lgb_eval,

#             early_stopping_rounds=100,

#             verbose_eval=100,

#             feval = rmsle_lgb)

#     y_pred = lgb_reg.predict(X_val, num_iteration=lgb_reg.best_iteration)

#     test_preds += lgb_reg.predict(X_test, num_iteration=lgb_reg.best_iteration) / N

#     cv_score.append(rmsle(y_val, y_pred))



#     fold_importance_df = pd.DataFrame()

#     fold_importance_df["feature"] = features

#     fold_importance_df["importance"] = lgb_reg.feature_importance(importance_type='gain')

#     fold_importance_df["fold"] = fold_ + 1

#     feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)



#     sns.residplot(y_pred, y_val, lowess=True, color='g')

#     plt.show()

#     print(f"fold{fold_}: {cv_score[-1]}\n\n")

# print(f"CV RMSLE Score: {sum(cv_score)/len(cv_score)}")