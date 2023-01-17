import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
color = sns.color_palette()
%matplotlib inline

from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go

from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb

pd.options.mode.chained_assignment = None
pd.options.display.max_columns = 999
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
#show traindata
train_df = load_df()
test_df = load_df("../input/test.csv")


#train_df["totals.transactionRevenue"] = train_df["totals.transactionRevenue"].fillna(0,inplace=True)
y_data = ["date","totals.transactionRevenue"]
train_y = train_df[y_data].astype("float")
#train_y = train_y.groupby("fullVisitorId").sum()

#conbain train and test
# Align the training and testing data, keep only columns present in both dataframes
train_df, test_df = train_df.align(test_df, join = 'inner', axis = 1)

#check column which have only one category
one_category = [c for c in train_df.columns if train_df[c].nunique()==1]

train_df = train_df.drop(one_category,axis=1)
test_df = test_df.drop(one_category,axis=1)
# Impute 0 for missing target values
#train_y.fillna(0, inplace=True).value
#train_y = train_df["totals.transactionRevenue"].values
train_id = train_df["fullVisitorId"].values
test_id = test_df["fullVisitorId"].values

# label encode the categorical variables and convert the numerical variables to float
cat_cols = ["channelGrouping", "device.browser", "device.deviceCategory", "device.operatingSystem", 
            "geoNetwork.city", "geoNetwork.continent", "geoNetwork.country", "geoNetwork.metro",
            "geoNetwork.networkDomain", "geoNetwork.region", "geoNetwork.subContinent", "trafficSource.adContent", 
            "trafficSource.adwordsClickInfo.adNetworkType", "trafficSource.adwordsClickInfo.gclId", 
            "trafficSource.adwordsClickInfo.page", "trafficSource.adwordsClickInfo.slot", "trafficSource.campaign",
            "trafficSource.keyword", "trafficSource.medium", "trafficSource.referralPath", "trafficSource.source"]
num_cols = ["totals.hits", "totals.pageviews"]
for col in cat_cols:
    print(col)
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))
for col in num_cols:
    train_df[col] = train_df[col].astype(float)
    test_df[col] = test_df[col].astype(float)
train_df['date'] = train_df['date'].apply(lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))
test_df['date'] = test_df['date'].apply(lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))

dev_bool = train_df["date"] <= datetime.date(2017,5,31)
val_bool = train_df["date"] > datetime.date(2017,5,31)

# Split the train dataset into development and valid based on time 
dev_df = train_df[dev_bool]
val_df = train_df[val_bool]
dev_y = train_y[dev_bool]
val_y = train_y[val_bool]


dev_X = dev_df[cat_cols + num_cols]
val_X = val_df[cat_cols + num_cols]
test_X = test_df[cat_cols + num_cols]

dev_y["totals.transactionRevenue"].sum()
val_y["totals.transactionRevenue"].sum()
dev_y["totals.transactionRevenue"].fillna(0,inplace=True)
val_y["totals.transactionRevenue"].fillna(0,inplace=True)
dev_y = np.log1p(dev_y["totals.transactionRevenue"].values)
val_y = np.log1p(val_y["totals.transactionRevenue"].values)

param_list = [0.01,0.03,0.04,0.05,0.1,0.2,0.25,0.3]
#0.03,0.2が良い
def run_lgb(train_X, train_y, val_X, val_y, test_X,learning_rate=0.1):
    params = {
        "objective" : "regression",
        "metric" : "rmse",
        "num_leaves" : 30,
        "learning_rate" : learning_rate,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1
        }

    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    return pred_test_y, model
# Training the model #
#bestmodel = []
bestscore = 1000
#best_learning_rate = []
#bestpred_test = []
for learning_rate in param_list:
    print("\n"+"learning_rate ="+str(learning_rate))
    pred_test, model = run_lgb(dev_X, dev_y, val_X, val_y, test_X,learning_rate)
    if bestscore > model.best_score["valid_0"]["rmse"]:
        bestmodel = model
        bestscore = model.best_score["valid_0"]["rmse"]
        best_learning_rate = learning_rate
        bestpred_test = pred_test
print("bestscore="+str(model.best_score["valid_0"]["rmse"])+" best_learning_rate"+str(best_learning_rate))        
 
def make_csv(pred,id=test_id,name="baseline_lgb.csv"):
    sub_df = pd.DataFrame({"fullVisitorId":id})
    pred[pred<0] = 0
    sub_df["PredictedLogRevenue"] = np.expm1(pred)
    sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
    sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
    sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
    sub_df.to_csv(name, index=False)
sub_df = pd.DataFrame({"fullVisitorId":test_id})
bestpred_test[bestpred_test<0] = 0
sub_df["PredictedLogRevenue"] = np.expm1(bestpred_test)
sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
sub_df.to_csv("baseline_lgb.csv", index=False)

fig, ax = plt.subplots(figsize=(12,18))
lgb.plot_importance(bestmodel, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()
train_df["totals.pageviews"].fillna(0,inplace=True)
test_df["totals.pageviews"].fillna(0,inplace=True)
FeatureImportance = pd.DataFrame({"name":bestmodel.feature_name(),
                                  "importance":bestmodel.feature_importance()})
FeatureImportance.sort_values("importance")
PF_list = FeatureImportance.sort_values("importance",ascending=False)
PF_list = PF_list[:5]
PF_list
# Make a new dataframe for polynomial features
poly_features_train = train_df[PF_list["name"]]
poly_features_test = test_df[PF_list["name"]]

# imputer for handling missing values
#from sklearn.preprocessing import Imputer
#imputer = Imputer(strategy = 'median')

#poly_target = poly_features['TARGET']

#poly_features = poly_features.drop(columns = ['TARGET'])

# Need to impute missing values
#poly_features = imputer.fit_transform(poly_features)
#poly_features_test = imputer.transform(poly_features_test)

from sklearn.preprocessing import PolynomialFeatures
                                  
# Create the polynomial object with specified degree
poly_transformer = PolynomialFeatures(degree = 3)
# Train the polynomial features
poly_transformer.fit(poly_features_train)

# Transform the features
poly_features_train = poly_transformer.transform(poly_features_train)
poly_features_test = poly_transformer.transform(poly_features_test)
print('Polynomial Features shape: ', poly_features_train.shape)
list(PF_list["name"])
poly_transformer.get_feature_names(input_features = list(PF_list["name"]))
poly_features_train = pd.DataFrame(poly_features_train,
                                   columns = poly_transformer.get_feature_names(
                                       list(PF_list["name"])))
poly_features_test = pd.DataFrame(poly_features_test,columns = poly_transformer.get_feature_names(list(PF_list["name"])))
train_df_poly = pd.concat([train_df,poly_features_train],axis=1)
train_df_poly.shape
test_df_poly = pd.concat([test_df,poly_features_test],axis=1)
test_df_poly.shape
PF_namelist = list(poly_features_train.columns)
PF_namelist.pop(0)


PF_namelist = list(set(PF_namelist+cat_cols+num_cols))
dev_poly_df = train_df_poly[dev_bool]
val_poly_df = train_df_poly[val_bool]
dev_poly_y = train_y[dev_bool]
val_poly_y = train_y[val_bool]


dev_poly_X = dev_poly_df[PF_namelist]
val_poly_X = val_poly_df[PF_namelist]
test_poly_X = test_df_poly[PF_namelist]
dev_poly_y["totals.transactionRevenue"].fillna(0,inplace=True)
val_poly_y["totals.transactionRevenue"].fillna(0,inplace=True)
dev_poly_y = np.log1p(dev_poly_y["totals.transactionRevenue"].values)
val_poly_y = np.log1p(val_poly_y["totals.transactionRevenue"].values)


param_list = [0.01,0.03,0.04,0.05,0.1,0.2,0.25,0.3]

bestscore_poly = 1000
#best_learning_rate = []
#bestpred_test = []
for learning_rate in param_list:
    print("\n"+"learning_rate ="+str(learning_rate))
    pred_test, model = run_lgb(dev_poly_X, dev_poly_y, val_poly_X, val_poly_y, test_poly_X,learning_rate)
    if bestscore_poly > model.best_score["valid_0"]["rmse"]:
        bestmodel_poly = model
        bestscore_poly = model.best_score["valid_0"]["rmse"]
        best_learning_rate_poly = learning_rate
        bestpred_test_poly = pred_test
print("bestscore="+str(model.best_score["valid_0"]["rmse"])+" best_learning_rate"+str(best_learning_rate))        

make_csv(pred=bestpred_test_poly,name="polynomical_lgb")
fig, ax = plt.subplots(figsize=(12,18))
lgb.plot_importance(bestmodel_poly, max_num_features=50, height=0.8, ax=ax)
ax.grid(False)
plt.title("LightGBM - Feature Importance", fontsize=15)
plt.show()
FeatureImportance = pd.DataFrame({"name":bestmodel_poly.feature_name(),
                                  "importance":bestmodel_poly.feature_importance()})
FeatureImportance.sort_values("importance")
FeatureImportance2 = FeatureImportance[FeatureImportance["importance"] > 50]
PF_namelist2 = list(FeatureImportance2["name"])
PF_namelist2
len(PF_namelist2)
PF_namelist

for i in range(len(PF_namelist2)):
    PF_namelist2[i] = PF_namelist2[i].replace("_"," ")
PF_namelist2
PF_namelist
dev_poly_df = train_df_poly[dev_bool]
val_poly_df = train_df_poly[val_bool]
dev_poly_y = train_y[dev_bool]
val_poly_y = train_y[val_bool]

dev_poly2_X = dev_poly_df[PF_namelist2]
val_poly2_X = val_poly_df[PF_namelist2]
test_poly2_X = test_df_poly[PF_namelist2]
dev_poly_y["totals.transactionRevenue"].fillna(0,inplace=True)
val_poly_y["totals.transactionRevenue"].fillna(0,inplace=True)
dev_poly_y = np.log1p(dev_poly_y["totals.transactionRevenue"].values)
val_poly_y = np.log1p(val_poly_y["totals.transactionRevenue"].values)


bestscore_poly2 = 1000
#best_learning_rate = []
#bestpred_test = []
for learning_rate in param_list:
    print("\n"+"learning_rate ="+str(learning_rate))
    pred_test, model = run_lgb(dev_poly2_X, dev_poly_y, val_poly2_X, val_poly_y, test_poly2_X,learning_rate)
    if bestscore_poly2 > model.best_score["valid_0"]["rmse"]:
        bestmodel_poly2 = model
        bestscore_poly2 = model.best_score["valid_0"]["rmse"]
        best_learning_rate_poly2 = learning_rate
        bestpred_test_poly2 = pred_test
print("bestscore="+str(model.best_score["valid_0"]["rmse"])+" best_learning_rate"+str(best_learning_rate))        

make_csv(pred=bestpred_test_poly2,name="polynomical2_lgb.csv")


