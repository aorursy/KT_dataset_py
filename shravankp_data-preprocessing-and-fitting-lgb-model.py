import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb
def load_df(csv_path='./train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, #json.loads takes in a string and converts to dict or list object.
                     dtype={'fullVisitorId': 'str'}, #convert id to string
                     nrows=nrows)
    

    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])  #converts semi-structured json to flat table.
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Shape: {df.shape}")
    return df
train_df = load_df("../input/train.csv")
test_df = load_df("../input/test.csv")
#train_df.to_csv("train_sep_cols")
#test_df.to_csv("test_sep_cols")
#train_df = load_df("./train_sep_cols")
#test_df = load_df("./test_sep_cols")
train_df.head()
test_df.head()
numeric_features = train_df.select_dtypes(include=[np.number])
print(numeric_features.columns)

categorical_features = train_df.select_dtypes(include=[np.object])
print(categorical_features.columns)
train_df["totals.transactionRevenue"] = train_df["totals.transactionRevenue"].astype('float')
grouped_revenue = train_df.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()

non_zero_customers = (grouped_revenue["totals.transactionRevenue"]>0).sum()
print("Number of unique customers with non-zero revenue : ", non_zero_customers, "and the ratio is : ", non_zero_customers / grouped_revenue.shape[0])
# convert the 'date' column values to datetime object
import datetime
train_df['date'] = train_df['date'].apply(lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))
test_df['date'] = test_df['date'].apply(lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))
consts = [c for c in train_df.columns if train_df[c].nunique(dropna=False)==1 ] #lets include nan values in the count(nunique())
consts
set(train_df.columns).difference(set(test_df.columns))
cols_to_drop = consts + ["sessionId"] + ["trafficSource.campaignCode"]
train_df = train_df.drop(cols_to_drop, axis=1)
test_df = test_df.drop(cols_to_drop[:-1], axis=1)
#train_df.head()
#train_df.info()
#train_df.describe()

train_df["totals.transactionRevenue"].fillna(0, inplace=True)
train_y = train_df["totals.transactionRevenue"].values

#identify categorical variables and label encode them.
categorical_cols = ["channelGrouping", "device.browser", 
            "device.deviceCategory", "device.operatingSystem", 
            "geoNetwork.city", "geoNetwork.continent", 
            "geoNetwork.country", "geoNetwork.metro",
            "geoNetwork.networkDomain", "geoNetwork.region", 
            "geoNetwork.subContinent", "trafficSource.adContent", 
            "trafficSource.adwordsClickInfo.adNetworkType", 
            "trafficSource.adwordsClickInfo.gclId", 
            "trafficSource.adwordsClickInfo.page", 
            "trafficSource.adwordsClickInfo.slot", "trafficSource.campaign",
            "trafficSource.keyword", "trafficSource.medium", 
            "trafficSource.referralPath", "trafficSource.source",
            'trafficSource.adwordsClickInfo.isVideoAd', 'trafficSource.isTrueDirect']

for col in categorical_cols:
    print(col)
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))


numeric_cols = ["totals.hits", "totals.pageviews", "visitNumber", "visitStartTime", 'totals.bounces',  'totals.newVisits']    
for col in numeric_cols:
    train_df[col] = train_df[col].astype(float)
    test_df[col] = test_df[col].astype(float)

# Split the train dataset into development and valid based on time 
dev_df = train_df[train_df['date']<=datetime.date(2017,5,31)]
val_df = train_df[train_df['date']>datetime.date(2017,5,31)]

dev_y = np.log1p(dev_df["totals.transactionRevenue"].values)
val_y = np.log1p(val_df["totals.transactionRevenue"].values)

dev_X = dev_df[categorical_cols + numeric_cols] 
val_X = val_df[categorical_cols + numeric_cols] 
test_X = test_df[categorical_cols + numeric_cols] 
def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {"objective" : "regression","metric" : "rmse", 
            "subsample" : 0.9,"colsample_bytree" : 0.9,
            "num_leaves" : 31,"min_child_samples" : 100,
            "learning_rate" : 0.03,"bagging_fraction" : 0.7,
            "feature_fraction" : 0.5,"bagging_frequency" : 5,
            "bagging_seed" : 2018,"verbosity" : -1}
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(params, lgtrain, 1000,early_stopping_rounds=100, valid_sets=[lgval],  verbose_eval=100)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    pred_val_y = model.predict(val_X, num_iteration=model.best_iteration)
    return pred_test_y, model, pred_val_y

# Training the model #
pred_test, model, pred_val = run_lgb(dev_X, dev_y, val_X, val_y, test_X)
from sklearn import metrics

pred_val[pred_val<0] = 0
val_pred_df = pd.DataFrame({"fullVisitorId":val_df["fullVisitorId"].values})
val_pred_df["transactionRevenue"] = val_df["totals.transactionRevenue"].values
val_pred_df["PredictedRevenue"] = np.expm1(pred_val) #exp(x) -1 can also be used but expm1 gives greater precision when converting log

val_pred_df = val_pred_df.groupby("fullVisitorId")["transactionRevenue", "PredictedRevenue"].sum().reset_index()
val_pred_df["transactionRevenue"] = np.log1p(val_pred_df["transactionRevenue"].values)
val_pred_df["PredictedRevenue"] =  np.log1p(val_pred_df["PredictedRevenue"].values)

#Now apply rms to find out error
print(np.sqrt(metrics.mean_squared_error(val_pred_df["transactionRevenue"].values, val_pred_df["PredictedRevenue"].values)))
train_id = train_df["fullVisitorId"].values
test_id = test_df["fullVisitorId"].values   
submit_df = pd.DataFrame({"fullVisitorId":test_id})

#Repeat same steps as we did for cross-validation
pred_test[pred_test<0] = 0
submit_df["PredictedLogRevenue"] = np.expm1(pred_test)
submit_df = submit_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
submit_df["PredictedLogRevenue"] = np.log1p(submit_df["PredictedLogRevenue"])

submit_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
print(submit_df.head())
submit_df.to_csv("submission.csv", index=False)
