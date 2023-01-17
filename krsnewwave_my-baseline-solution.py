import numpy as np
import pandas as pd
import os
import json
from pandas.io.json import json_normalize
import seaborn as sns
from matplotlib import pyplot as plt

print(os.listdir("../input"))
# shared from this kernel
# https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields
# def load_df(csv_path='../input/train.csv', nrows=None):
#     JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
#     df = pd.read_csv(csv_path, dtype={'fullVisitorId': 'str'}, nrows=nrows)
#     for column in JSON_COLUMNS:
#         df = df.join(pd.DataFrame(df.pop(column).apply(pd.io.json.loads).values.tolist(), index=df.index))

#     return df

def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df
%%time
df_train = load_df(nrows=50000)
df_test = load_df(nrows=50000)
import missingno as msno
msno.matrix(df_train);
msno.dendrogram(df_train);
# remove constant cols
constant_cols = []
all_cols = df_train.columns
for col in all_cols:
    if df_train[col].nunique() == 1:
        constant_cols.append(col)
        
df_train.drop(constant_cols, axis=1, inplace=True)
# convert numericals
def convert_to_numerical(df, cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

total_cols = df_train.filter(like="totals")
df_train = convert_to_numerical(df_train, total_cols)
df_train = convert_to_numerical(df_train, total_cols)

other_numerical_cols = ["trafficSource.adwordsClickInfo.page", ]
df_train = convert_to_numerical(df_train, other_numerical_cols)
df_test = convert_to_numerical(df_test, other_numerical_cols)
# format dates
df_train["date"] = pd.to_datetime(df_train["date"], errors='coerce', format="%Y%m%d")
df_train['visitStartTime'] = pd.to_datetime(df_train['visitStartTime'],unit='s')
# These values are considered NONE
DEFAULT_VALUE = ["not available in demo dataset", "(none)"]

# then all categoricals default values will be "NA"
FILLNA_VALUE_CAT = "NA"
for col in df_train.select_dtypes(np.object).columns:
    df_train[col].replace(DEFAULT_VALUE, value=None, inplace=True)
    df_train[col].fillna(FILLNA_VALUE_CAT, inplace=True)
FULL_ID = "fullVisitorId"
list_feature_dfs = []
# basic stats
numerical_cols = ["totals.hits", "totals.pageviews"]
df_feat_num = df_train.groupby(FULL_ID)[numerical_cols].agg(["mean", "std", "sem"])
df_feat_num.columns = [' '.join(col).strip() for col in df_feat_num.columns.values]

# number of visits
df_num_visits = df_train.groupby(FULL_ID).size().to_frame('num_visits')

# number of sessions
df_num_sessions = df_train.groupby(FULL_ID)["sessionId"].nunique().to_frame("num_sessions")

list_feature_dfs.append(df_feat_num)
list_feature_dfs.append(df_num_visits)
list_feature_dfs.append(df_num_sessions)
# consider as categorical all columns that have unique values <= 100
# this is a feature engineering parameter
NUM_UNIQUE_VALS_TO_BE_CAT = 100
EXEMPT_COLS = [FULL_ID, "sessionId", "trafficSource.adwordsClickInfo.gclId"]
categorical_cols = []
text_cols = []
# all categorical-like columns
for col in df_train.select_dtypes(np.object).drop(EXEMPT_COLS, axis=1).columns:
    if len(df_train[col].value_counts()) <= NUM_UNIQUE_VALS_TO_BE_CAT:
        categorical_cols.append(col)
    else:
        text_cols.append(col)
categorical_cols
text_cols
# this can be considered as a regularization parameter. 
# All values outside of the top THRESH_CAT are considered "others"
THRESH_CAT = 20
OTHERS = "OTHERS"

def group_id_categorical(df, col, threshold_categorical = THRESH_CAT, others_value = OTHERS):    
    vc = df[col].value_counts()

    list_top = vc[:threshold_categorical].index
    
    col_series = np.where(df[col].isin(list_top), df[col], others_value)
    id_series = df[FULL_ID]

    df_group_col = pd.DataFrame({FULL_ID : id_series, col : col_series})
    df_group_col = df_group_col.groupby([FULL_ID, col]).size().unstack()
    
    # naming: origin_variable.value
    df_group_col.columns = col + "=" + df_group_col.columns
    
    return df_group_col.fillna(0)
for col in categorical_cols:
    df_group_col = group_id_categorical(df_train, col)
    list_feature_dfs.append(df_group_col)
df_train["date-month"] = "month-" + df_train["date"].dt.month.apply(str)
df_train["date-dayofweek"] = "dayofweek-" + df_train["date"].dt.day_name()
df_train['date-weekofmonth'] = "weekofmonth-" + df_train["date"].apply(lambda d: str((d.day-1) // 7 + 1))

df_date_month = df_train.groupby([FULL_ID, "date-month"]).size().unstack().fillna(0)
df_date_dayofweek = df_train.groupby([FULL_ID, "date-dayofweek"]).size().unstack().fillna(0)
df_date_weekofmonth = df_train.groupby([FULL_ID, "date-weekofmonth"]).size().unstack().fillna(0)

list_feature_dfs.append(df_date_month)
list_feature_dfs.append(df_date_dayofweek)
list_feature_dfs.append(df_date_weekofmonth)
df_train["visit-month"] = "visit-month-" + df_train["visitStartTime"].dt.month.apply(str)
df_train["visit-dayofweek"] = "visit-dayofweek-" + df_train["visitStartTime"].dt.day_name()
df_train['visit-weekofmonth'] = "visit- weekofmonth-" + df_train["visitStartTime"].apply(
    lambda d: str((d.day-1) // 7 + 1))
df_train["time-hour"] = "hour-" + df_train["visitStartTime"].dt.hour.apply(str)

df_visit_month = df_train.groupby([FULL_ID, "visit-month"]).size().unstack().fillna(0)
df_visit_dayofweek = df_train.groupby([FULL_ID, "visit-dayofweek"]).size().unstack().fillna(0)
df_visit_weekofmonth = df_train.groupby([FULL_ID, "visit-weekofmonth"]).size().unstack().fillna(0)
df_time_hour = df_train.groupby([FULL_ID, "time-hour"]).size().unstack().fillna(0)

list_feature_dfs.append(df_visit_month)
list_feature_dfs.append(df_visit_dayofweek)
list_feature_dfs.append(df_visit_weekofmonth)
list_feature_dfs.append(df_time_hour)

from sklearn.feature_extraction.text import CountVectorizer
col = "trafficSource.source"
df = df_train

def group_id_text(df, col, min_df=3):
    df_text = df.groupby(FULL_ID)[col].apply(lambda x: "%s" % ' '.join(x))
    df_text_index = df_text.index

    count_vec = CountVectorizer(binary=True, min_df=min_df)
    df_text = pd.DataFrame(count_vec.fit_transform(df_text).toarray(), index=df_text_index, 
                 columns=count_vec.get_feature_names())
    df_text.columns = col + "=" + df_text.columns
    return df_text
for col in text_cols:
    list_feature_dfs.append(group_id_text(df_train, col))
from functools import reduce
def join_dfs(ldf, rdf):
    return ldf.join(rdf, how='left')

df_features = reduce(join_dfs, list_feature_dfs)

df_features = df_features.fillna(0)
df_features.head()
target_data = df_train.groupby(FULL_ID)["totals.transactionRevenue"].sum()

# log my target data
target_data = np.log(1+ target_data)

# reindex target data so features are aligned
target_data = target_data.reindex(df_features.index)
df_features.shape
from lightgbm.sklearn import LGBMRegressor

model=LGBMRegressor(boosting_type='gbdt', max_depth=5, n_estimators=1000, 
                    max_bin=255,subsample_for_bin=50000,
                    min_child_weight=3,min_child_samples=10,subsample=0.6,subsample_freq=1,colsample_bytree=0.6,
                    seed=23,silent=False,nthread=-1,n_jobs=-1)
# from sklearn.ensemble import RandomForestRegressor
# params = {"n_estimators" : 100, "max_depth" : 7, "min_samples_split" : 0.001, "max_features" : "sqrt"}
# rfr = RandomForestRegressor(verbose=1, n_jobs=4)
# rfr.set_params(**params)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error, mean_squared_error

random_seed = 1234
X_train, X_test, y_train, y_test = train_test_split(df_features, target_data, test_size=0.2, 
                                                    random_state = random_seed)
PATIENCE = 3
# evaluation set
X_train1, X_val, y_train1, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state = random_seed)

eval_params = {"eval_set": (X_val, y_val), "eval_metric":'l2_loss', "early_stopping_rounds":PATIENCE}
model.fit(X_train, y_train, **eval_params)

y_preds = model.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_preds))
list_test_feature_dfs = []
df_test = convert_to_numerical(df_test, total_cols)
df_test = convert_to_numerical(df_test, other_numerical_cols)
for col in df_test.select_dtypes(np.object).columns:
    df_test[col].replace(DEFAULT_VALUE, value=None, inplace=True)
    df_test[col].fillna(FILLNA_VALUE_CAT, inplace=True)
# format dates
df_test["date"] = pd.to_datetime(df_test["date"], errors='coerce', format="%Y%m%d")
df_test['visitStartTime'] = pd.to_datetime(df_test['visitStartTime'],unit='s')
# basic stats
numerical_cols = ["totals.hits", "totals.pageviews"]
df_feat_num = df_test.groupby(FULL_ID)[numerical_cols].agg(["mean", "std", "sem"])
df_feat_num.columns = [' '.join(col).strip() for col in df_feat_num.columns.values]

# number of visits
df_num_visits = df_test.groupby(FULL_ID).size().to_frame('num_visits')

# number of sessions
df_num_sessions = df_train.groupby(FULL_ID)["sessionId"].nunique().to_frame("num_sessions")

list_test_feature_dfs.append(df_feat_num)
list_test_feature_dfs.append(df_num_visits)
list_test_feature_dfs.append(df_num_sessions)
for col in categorical_cols:
    df_group_col = group_id_categorical(df_test, col)
    list_test_feature_dfs.append(df_group_col)
    
for col in text_cols:
    list_test_feature_dfs.append(group_id_text(df_test, col))
df_test["date-month"] = "month-" + df_test["date"].dt.month.apply(str)
df_test["date-dayofweek"] = "dayofweek-" + df_test["date"].dt.day_name()
df_test['date-weekofmonth'] = "weekofmonth-" + df_test["date"].apply(lambda d: str((d.day-1) // 7 + 1))

df_date_month = df_test.groupby([FULL_ID, "date-month"]).size().unstack().fillna(0)
df_date_dayofweek = df_test.groupby([FULL_ID, "date-dayofweek"]).size().unstack().fillna(0)
df_date_weekofmonth = df_test.groupby([FULL_ID, "date-weekofmonth"]).size().unstack().fillna(0)

list_test_feature_dfs.append(df_date_month)
list_test_feature_dfs.append(df_date_dayofweek)
list_test_feature_dfs.append(df_date_weekofmonth)
df_test["visit-month"] = "visit-month-" + df_test["visitStartTime"].dt.month.apply(str)
df_test["visit-dayofweek"] = "visit-dayofweek-" + df_test["visitStartTime"].dt.day_name()
df_test['visit-weekofmonth'] = "visit- weekofmonth-" + df_test["visitStartTime"].apply(
    lambda d: str((d.day-1) // 7 + 1))
df_test["time-hour"] = "hour-" + df_test["visitStartTime"].dt.hour.apply(str)

df_visit_month = df_test.groupby([FULL_ID, "visit-month"]).size().unstack().fillna(0)
df_visit_dayofweek = df_test.groupby([FULL_ID, "visit-dayofweek"]).size().unstack().fillna(0)
df_visit_weekofmonth = df_test.groupby([FULL_ID, "visit-weekofmonth"]).size().unstack().fillna(0)
df_time_hour = df_test.groupby([FULL_ID, "time-hour"]).size().unstack().fillna(0)

list_test_feature_dfs.append(df_visit_month)
list_test_feature_dfs.append(df_visit_dayofweek)
list_test_feature_dfs.append(df_visit_weekofmonth)
list_test_feature_dfs.append(df_time_hour)
df_test_features = reduce(join_dfs, list_test_feature_dfs)

# add columns that are not in the test set
missing_columns = df_features.columns.difference(df_test_features.columns)
df_test_features = df_test_features.reindex(columns = df_test_features.columns.tolist() + missing_columns.tolist())

# fill with 0
df_test_features = df_test_features.fillna(0)

# reorder columns
df_test_features = df_test_features[df_features.columns.tolist()]

df_test_features.head()
y_test_preds = model.predict(df_test_features)
df_submission = pd.DataFrame({"PredictedLogRevenue" : 
                             y_test_preds}, index=df_test_features.index)
df_sample_submission = pd.read_csv("../input/sample_submission.csv", index_col=0)
df_sample_submission.update(df_submission)
df_sample_submission.reset_index(inplace=True)
df_sample_submission.to_csv('submission.csv', index=False)

