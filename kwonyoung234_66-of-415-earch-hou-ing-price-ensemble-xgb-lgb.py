import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import warnings

warnings.filterwarnings('ignore')



import os

print(os.listdir("../input"))
IS_LOCAL = False



if IS_LOCAL:

    PATH="drive/colab/Kakr_2nd/data/"

else:

    PATH="../input/"
train = pd.read_csv(PATH+"train.csv")

test = pd.read_csv(PATH+"test.csv")
print("The Dataset's shape for train is {}, for test is {}".format(train.shape,test.shape))
train.head()
train.describe()
train.info()
def draw_kdeplot(column):



    plt.figure(figsize=[8,6])

    

    sns.kdeplot(train[column],bw=0.5,label='train')

    sns.kdeplot(test[column],bw=0.5,label='test')

    

    plt.xlabel(column,fontsize=12)

    plt.title(f"Distribution of {column}",fontsize=20)

    plt.show()
def make_count_df(df,column):

    dummy = df.copy()

    result_df = dummy[column].value_counts().sort_index().to_frame().reset_index().rename(columns={"index":column,column:"counts"})

    return result_df
def compare_categorical_ratio(count_train,count_test,column,adjust_x_annotate=5,fontsize=14):

    fig, ax = plt.subplots(1,2,figsize=[12,6])

    

    ax1 = plt.subplot(1,2,1)

    sns.barplot(x=column,y='counts',data=count_train,label='train')



    for p in ax1.patches:

        ax1.annotate('{:.2f}%'.format(p.get_height()/count_train["counts"].sum()) , (p.get_x()+p.get_width()/adjust_x_annotate, p.get_height()),fontsize=fontsize)



    ax2 = plt.subplot(1,2,2)

    sns.barplot(x=column,y='counts',data=count_test,label='test')



    for p in ax2.patches:

        ax2.annotate('{:.2f}%'.format(p.get_height()/count_test["counts"].sum()) , (p.get_x()+p.get_width()/adjust_x_annotate, p.get_height()),fontsize=fontsize)



    plt.suptitle(f"Comparing btw train and test about {column}")

    plt.show()
train.date.head(10)
train.date.apply(lambda x:str(x)[-7:]).value_counts()
print("Minimum value of Price is {}, Maximum value of Price is {}".format(train.price.min(),train.price.max()))
plt.figure(figsize=[8,4])

sns.distplot(train.price,hist=False,label='train',color='blue')

plt.xticks(rotation=60)

plt.title("Distribution of Price value")
train.bedrooms.value_counts().sort_index()
bedroom_train = make_count_df(train,"bedrooms")

bedroom_test = make_count_df(test,"bedrooms")
plt.figure(figsize=[8,6])



# ax = train.bedrooms.value_counts().sort_index().to_frame().plot(kind='bar',linewidth=2,figsize=[8,6])

# for p in ax.patches:

#     ax.annotate(p.get_height(), (p.get_x()-0.05, p.get_height()))



sns.barplot(x='bedrooms',y='counts',data=bedroom_train,label='train',color='red')

sns.barplot(x='bedrooms',y='counts',data=bedroom_test,label='test',color='blue')

plt.legend()

plt.ylabel("# of Bedrooms",fontsize=12)

plt.xlabel("Bedrooms",fontsize=12)



plt.title("Number of Bedrooms",fontsize=20)
train.bathrooms.value_counts().head()
plt.figure(figsize=[8,6])



sns.kdeplot(train.bathrooms,bw=0.5,label='train')

sns.kdeplot(test.bathrooms,bw=0.5,label='test')



plt.xlabel("Bathrooms(# of Bathrooms / # of Bedrooms)",fontsize=12)

plt.title("Distribution of Bathrooms(# of Bathrooms / # of Bedrooms)",fontsize=20)
train.bathrooms.mul(train.bedrooms).head()
print("The min number of real bathroom is {}, max number of real bathroom is {}".format(train.bathrooms.mul(train.bedrooms).min(),train.bathrooms.mul(train.bedrooms).max()))
fig,ax = plt.subplots(1,2,figsize=[12,6])



ax1 = plt.subplot(1,2,1)

sns.kdeplot(train.sqft_living,bw=0.5,label="train")

sns.kdeplot(test.sqft_living,bw=0.5,label='test')

ax1.set_xlabel("sqft_living",fontsize=12)



ax2 = plt.subplot(1,2,2)

sns.kdeplot(train.sqft_lot,bw=0.5,label="train")

sns.kdeplot(test.sqft_lot,bw=0.5,label='test')

ax2.set_xlabel("sqft_lot",fontsize=12)



plt.suptitle("Distribution of sqft_living and sqft_lot")
train.floors.value_counts()
draw_kdeplot("floors")
waterfront_train = make_count_df(train,"waterfront")

waterfront_test = make_count_df(test,"waterfront")
compare_categorical_ratio(waterfront_train,waterfront_test,"waterfront",3)
train.view.value_counts().sort_index()
view_train= make_count_df(train,"view")

view_test = make_count_df(test,"view")
compare_categorical_ratio(view_train,view_test,"view",10)
condition_train = make_count_df(train,"condition")

condition_test = make_count_df(test,"condition")
compare_categorical_ratio(condition_train,condition_test,"condition",8)
grade_train = make_count_df(train,"grade") 

grade_test = make_count_df(test,"grade")
compare_categorical_ratio(grade_train,grade_test,"grade",adjust_x_annotate=20,fontsize=10)
fig,ax = plt.subplots(1,2,figsize=[12,6])



ax1 = plt.subplot(1,2,1)

sns.kdeplot(train.sqft_above,bw=0.5,label="train")

sns.kdeplot(test.sqft_above,bw=0.5,label='test')

ax1.set_xlabel("sqft_above",fontsize=12)



ax2 = plt.subplot(1,2,2)

sns.kdeplot(train.sqft_basement,bw=0.5,label="train")

sns.kdeplot(test.sqft_basement,bw=0.5,label='test')

ax2.set_xlabel("sqft_basement",fontsize=12)



plt.suptitle("Distribution of sqft_above and sqft_basement")
print("Ratio of 0 in sqft_basement of train_set {:.2f}% among {}".format(sum(train.sqft_basement==0)/len(train)*100,len(train)))

print("Ratio of 0 in sqft_basement of test_set {:.2f}% among {}".format(sum(test.sqft_basement==0)/len(test)*100,len(test)))
fig,ax = plt.subplots(1,2,figsize=[12,6])



ax1 = plt.subplot(1,2,1)

sns.kdeplot(train.yr_built,bw=0.5,label="train")

sns.kdeplot(test.yr_built,bw=0.5,label='test')

ax1.set_xlabel("yr_built",fontsize=12)



ax2 = plt.subplot(1,2,2)

sns.kdeplot(train.yr_renovated,bw=0.5,label="train")

sns.kdeplot(test.yr_renovated,bw=0.5,label='test')

ax2.set_xlabel("yr_renovated",fontsize=12)



plt.suptitle("Distribution of yr_built and yr_renovated")
plt.figure(figsize=[6,6])



sns.kdeplot(train.loc[train["yr_renovated"]!= 0,"yr_renovated"],bw=0.5,label="train")

sns.kdeplot(test.loc[test["yr_renovated"]!= 0,"yr_renovated"],bw=0.5,label="test")

plt.xlabel("yr_renovated")

plt.title("yr_renovated except for 0")
print("Ratio of 0 in yr_renovated of train_set {:.2f}% among {}".format(sum(train.yr_renovated==0)/len(train)*100,len(train)))

print("Ratio of 0 in yr_renovated of test_set {:.2f}% among {}".format(sum(test.yr_renovated==0)/len(test)*100,len(test)))
train.zipcode.head()
str(train.zipcode[0])
import re



re1='(\\d{5})'

rg = re.compile(re1)



dummy_train = train.zipcode.apply(lambda x :rg.search(str(x)))

dummy_test = test.zipcode.apply(lambda x :rg.search(str(x)))
print("The number of unexpected form about zipcode of train_set {}".format(sum(dummy_train == 0)))

print("The number of unexpected form about zipcode of test_set {}".format(sum(dummy_test == 0)))
plt.scatter(x=train.long,y=train.lat,color='red',label='train',alpha=0.7)

plt.scatter(x=test.long,y=test.lat,color='blue',label='test',alpha=0.7)

plt.legend()

plt.xlabel("longitude",fontsize=14)

plt.ylabel("latitude",fontsize=14)

plt.title("Distribution of lat and long about train and test set")
sns.jointplot(x='long',y='lat',data=train,kind="hex")

plt.suptitle("Longitude and Latitude Distribution of train_set")
sns.jointplot(x='long',y='lat',data=test,kind="hex")

plt.suptitle("Longitude and Latitude Distribution of test_set")
def decomposition_date(df):

    dummy = df.copy()

    

    dummy["date"] = pd.to_datetime(dummy["date"])

    

    dummy["year"] = dummy.date.apply(lambda x: str(x).split("-")[0]).astype('int')

    dummy["month"] = dummy.date.apply(lambda x:str(x).split("-")[1]).astype('int')

    dummy["day"] = dummy.date.apply(lambda x:str(x).split("-")[-1]).apply(lambda x:x.split(" ")[0]).astype('int')

    

    return dummy
decom_train = decomposition_date(train)

decom_test = decomposition_date(test)
decom_train.groupby('year')['price'].agg(['mean','median'])
decom_train.groupby('year')['price'].agg(['mean','median']).plot(kind='bar',linewidth=2)

plt.title("Mean and Median by Year")



decom_train.groupby('month')['price'].agg(['mean','median']).plot(kind='bar',linewidth=1,figsize=[8,6])

plt.title("Mean and Median by Month")



decom_train.groupby('day')['price'].agg(['mean','median']).plot(kind='bar',linewidth=1,figsize=[20,6])

plt.title("Mean and Median by Day")
ax = decom_train.groupby('bedrooms')['price'].agg(['mean','median']).plot(kind='bar',linewidth=2,figsize=[10,6])



for i,p in enumerate(ax.patches):

    if i < 11:

        ax.annotate(decom_train.bedrooms.value_counts().sort_index()[i],(p.get_x()+p.get_width()*0.5, p.get_y()+p.get_height()*1.01),fontsize=15,rotation=45)



plt.title("Mean, Median value by Bedrooms",fontsize=20)

plt.xlabel("Bedrooms",fontsize=12)

plt.ylabel("Price",fontsize=12)
plt.figure(figsize=[12,6])

sns.barplot(x='bathrooms',y='price',data=decom_train)
def float_with_price(xlabel,df):

    fig,ax = plt.subplots(1,2,figsize=[14,6])



    ax1 = plt.subplot(1,2,1)

    sns.scatterplot(x=xlabel,y='price',data=df,ci=0.95)

    ax1.set_title(f"Scatterplot about {xlabel} with price",fontsize=14)

    ax2 = plt.subplot(1,2,2)

    sns.regplot(x=xlabel,y='price',data=df,ci=0.95)

    ax2.set_title(f"Regplot about {xlabel} with price",fontsize=14)

    plt.xticks(rotation=60)

    plt.suptitle(f"Relationship about {xlabel} with price",fontsize=20)
float_with_price("sqft_living",train)
float_with_price("sqft_lot",train)
def ordinal_with_price(xlabel,df,rotation=0):

    

    fig,ax = plt.subplots(1,2,figsize=[14,6])



    ax1 = plt.subplot(1,2,1)

    sns.barplot(x=xlabel,y="price",data=train)

    ax1.set_xlabel(xlabel,fontsize=12)

    ax1.set_ylabel("price",fontsize=12)

    ax1.set_title(f"Barplot about {xlabel} with price",fontsize=18)

    

    for i,p in enumerate(ax1.patches):



        ax1.annotate(s=train[xlabel].value_counts().sort_index().values[i],xy= (p.get_x()+p.get_width()/len(train[xlabel].value_counts()), p.get_height()*1.05),fontsize=15,rotation=rotation)

        

    ax2 = plt.subplot(1,2,2)

    sns.boxplot(x=xlabel,y='price',data=train)

    ax2.set_xlabel(xlabel,fontsize=12)

    ax2.set_ylabel("price",fontsize=12)

    ax2.set_title(f"Boxplot about {xlabel} with price",fontsize=18)

    

    plt.suptitle(f"Relationship about {xlabel} with price",fontsize=20)
ordinal_with_price("floors",train)
ordinal_with_price("waterfront",train)
ordinal_with_price("view",train)
ordinal_with_price("condition",train)
ordinal_with_price("grade",train,rotation=60)
float_with_price("sqft_above",train)
float_with_price("sqft_basement",train)
float_with_price("yr_built",train)
float_with_price("yr_renovated",train)
float_with_price("sqft_living15",train)
float_with_price("sqft_lot15",train)
skewed_cols = ["bedrooms","sqft_living","sqft_lot","sqft_above","sqft_basement","sqft_living15","sqft_lot15"]
def to_logarithm(df,cols):

    

    result_df = df.copy()

    

    for col in cols:

        result_df[col] = np.log1p(result_df[col])

        

    return result_df
log_train = to_logarithm(train,skewed_cols)

log_test = to_logarithm(test,skewed_cols)
def making_additional_cols(df):

    

    result_df = df.copy()

    

    result_df['date'] = result_df['date'].apply(lambda e: e.split('T')[0])

    result_df['yr_renovated'] = result_df['yr_renovated'].apply(lambda x: np.nan if x == 0 else x)

    result_df['renovated'] = result_df['yr_renovated'].apply(lambda x: np.nan if x == 0 else x)

    result_df['yr_renovated'] = result_df['yr_renovated'].fillna(result_df['yr_built'])

    result_df['renovated'] = result_df['renovated'].fillna(0)

    result_df['yr_renovated'] = result_df['yr_renovated'].astype('int')

    

    result_df.loc[result_df.renovated > 0,'renovated']= 1.0

    

    result_df['total_rooms'] = result_df['bedrooms'] + result_df['bathrooms']

    result_df['sqft_ratio'] = result_df['sqft_living'] / result_df['sqft_lot']

    result_df['sqft_total_size'] = result_df['sqft_above'] + result_df['sqft_basement']

    result_df['sqft_ratio_1'] = result_df['sqft_living'] / result_df['sqft_total_size']

    result_df['sqft_ratio15'] = result_df['sqft_living15'] / result_df['sqft_lot15']

    

    result_df["year"] = pd.to_numeric(result_df.date.apply(lambda x:x[:4]))

    result_df["month"] = pd.to_numeric(result_df.date.apply(lambda x:x[4:6]))

    result_df.drop("date",axis=1,inplace=True)

    

    return result_df
adj_log_train = making_additional_cols(log_train)

adj_log_test = making_additional_cols(log_test)
adj_log_train.head()
adj_log_train.head()
adj_log_train["price"] = np.log1p(adj_log_train["price"])
X_adj_log_train = adj_log_train[adj_log_train.columns.values[2:]]

y_adj_log_train = adj_log_train["price"]

X_adj_log_test = adj_log_test[adj_log_test.columns.values[1:]]
predictions_dict = dict()

scores_dict = dict()
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold,KFold

from sklearn.metrics import mean_squared_error



# price를 로그화 하기 전에 사용했던 xgboost평가 함수

def xgb_rmse(preds,dtrain):

    

    labels = dtrain.get_label()

    

    score = np.sqrt(mean_squared_error(labels,preds))

  

    return "xgb_rmse",score



# 로그화된 price를 사용하는 xgboost평가 함수

def log_xgb_rmse(preds,dtrain):

    

    labels = dtrain.get_label()

    

    score = np.sqrt(mean_squared_error(np.expm1(labels),np.expm1(preds)))

  

    return "xgb_rmse",score



# xgboost 실행 함수

def model_xgb(X_train,y_train,X_test,nfolds=5,hyperparameters=None,logarithm=False):

    

    feature_names = X_train.columns.values

    

    valid_scores = np.zeros(len(X_train))

    predictions = np.zeros(len(X_test))

    

    valid_scores_list = []

    

    importances = np.zeros(len(feature_names))

    

    feature_importance_df = pd.DataFrame()

    feature_importance_df["features"] = feature_names

    

    if hyperparameters:

        params = hyperparameters

      

    else:

        params = {

            'base_score': 0.5,

             'booster': 'gbtree',

             'colsample_bylevel': 1,

             'colsample_bytree': 1,

             'gamma': 0,

             'importance_type': 'gain',

             'learning_rate': 0.1,

             'max_delta_step': 0,

             'max_depth': 3,

             'min_child_weight': 1,

             'missing': None,

             'n_estimators': 100,

             'n_jobs': 1,

             'nthread': None,

             'objective': 'reg:linear',

             'random_state': 101,

             'reg_alpha': 0,

             'reg_lambda': 1,

             'scale_pos_weight': 1,

             'seed': None,

             'silent': True,

             'subsample': 1}

    

    xgbr = xgb.XGBRegressor(**params)

    

    if logarithm:

        fold = KFold(n_splits=nfolds,shuffle=True,random_state=12)

        e_metric = log_xgb_rmse

    else:

        fold = StratifiedKFold(n_splits=nfolds,shuffle=True,random_state=12)

        e_metric = xgb_rmse

    

    print(params)

    

    for i,(train_indices,valid_indices) in enumerate(fold.split(X_train.values,y_train.values)):

        

        X = X_train.loc[train_indices]

        y = y_train.loc[train_indices]

        X_valid = X_train.loc[valid_indices]

        y_valid = y_train.loc[valid_indices]

        

        print("{} fold processing".format(i+1),"#"*20)

        

        xgbr.fit(X,y,eval_metric=log_xgb_rmse,verbose=500,early_stopping_rounds=250,eval_set=[(X,y),(X_valid,y_valid)])

  

        valid_scores_list.append(xgbr.get_booster().best_score)

        

        feature_importance_df[f"{i+1}"] = xgbr.feature_importances_

        

        if logarithm:

            valid_score = np.expm1(xgbr.predict(X_valid))

            prediction = np.expm1(xgbr.predict(X_test))

        

        else:

            valid_score = xgbr.predict(X_valid)

            prediction = xgbr.predict(X_test)

        

        valid_scores[valid_indices] += valid_score

        predictions += prediction / nfolds

    

    valid_mean_score = np.mean(valid_scores_list)

    

    print(f"mean_valid_score is {valid_mean_score} at {nfolds}")

    

    feature_importance_df["mean"] = feature_importance_df[feature_importance_df.columns.values[1:]].mean(axis=1)

    feature_importance_df["std"] = feature_importance_df[feature_importance_df.columns.values[1:]].mean(axis=1)

    

    fi_sorted = feature_importance_df.sort_values("mean",ascending=False)

    

    plt.figure(figsize=[6,40])

    sns.barplot(x="mean",y="features",data=fi_sorted,xerr=fi_sorted["std"])

    plt.title("Feature Importances of xgboost",fontsize=12)

    plt.show()

        

        

    return valid_mean_score, predictions

tuned_params = {

    'alpha': 0.23381888633529596,

    'booster': 'gbtree',

    'colsample_bytree': 0.5833187080443007,

    'gamma': 0.11877149186475625,

    'lambda': 0.7815712086648032,

    'learning_rate': 0.060873580474025094,

    'max_depth': 7,

    'min_child_weight': 5,

    'n_estimators': 3000, #1046

    'objective': 'reg:linear',

    'random_state': 101,

    'subsample': 0.6307967933325185}
valid_score,predictions = model_xgb(X_train=X_adj_log_train,X_test=X_adj_log_test,y_train=y_adj_log_train,hyperparameters=tuned_params,logarithm=True)
target = "xgb_uni_5"

scores_dict[target] = valid_score

predictions_dict[target] = predictions
submission = pd.read_csv(PATH+"sample_submission.csv")

submission["price"] = predictions

submission.to_csv("xgb_uni_layer_5.csv",index=False)
valid_score, predictions = model_xgb(X_train=X_adj_log_train,X_test=X_adj_log_test,y_train=y_adj_log_train,hyperparameters=tuned_params,logarithm=True,nfolds=10)
target = "xgb_uni_10"

scores_dict[target] = valid_score

predictions_dict[target] = predictions
submission = pd.read_csv(PATH+"sample_submission.csv")

submission["price"] = predictions

submission.to_csv("xgb_uni_layer_10.csv",index=False)
import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold,KFold

from sklearn.metrics import mean_squared_error



# price를 로그화 하기 전에 사용했던 lighgbm평가 함수

def lgb_rmse(y,preds):

    

    score = np.sqrt(mean_squared_error(y,preds))

  

    return "lgb_rmse",score,False



# price를 로그화 한 후에 사용하는 lightgbm평가 함수

def log_lgb_rmse(y,preds):

    

    score = np.sqrt(mean_squared_error(np.expm1(y),np.expm1(preds)))

  

    return "lgb_rmse",score,False



# lightgbm 실행함수

def model_lgb(X_train,y_train,X_test,nfolds=5,hyperparameters=None,logarithm=False):

    

    feature_names = X_train.columns.values

    

    valid_scores = np.zeros(len(X_train))

    predictions = np.zeros(len(X_test))

    

    valid_scores_list = []

    

    importances = np.zeros(len(feature_names))

    

    feature_importance_df = pd.DataFrame()

    feature_importance_df["features"] = feature_names

    

    if hyperparameters:

        params = hyperparameters

      

    else:

        params = {'boosting_type': 'gbdt',

                 'class_weight': None,

                 'colsample_bytree': 1.0,

                 'importance_type': 'split',

                 'learning_rate': 0.1,

                 'max_depth': -1,

                 'min_child_samples': 20,

                 'min_child_weight': 0.001,

                 'min_split_gain': 0.0,

                 'n_estimators': 1000,

                 'n_jobs': -1,

                 'num_leaves': 31,

                 'objective': "rmse",

                 'random_state': 101,

                 'reg_alpha': 0.0,

                 'reg_lambda': 0.0,

                 'silent': True,

                 'subsample': 1.0,

                 'subsample_for_bin': 200000,

                 'subsample_freq': 0}

    

    lgbr = lgb.LGBMRegressor(**params)

    

    if logarithm:

        fold = KFold(n_splits=nfolds,shuffle=True,random_state=12)

        e_metric = log_lgb_rmse

    else:

        fold = StratifiedKFold(n_splits=nfolds,shuffle=True,random_state=12)

        e_metric = lgb_rmse

    

    print(params)

    

    for i,(train_indices,valid_indices) in enumerate(fold.split(X_train.values,y_train.values)):

        

        X = X_train.values[train_indices]

        y = y_train.values[train_indices]

        X_valid = X_train.values[valid_indices]

        y_valid = y_train.values[valid_indices]

        

        print("{} fold processing".format(i+1),"#"*20)

        

        lgbr.fit(X,y,eval_set=[(X,y),(X_valid,y_valid)],eval_names=["train","valid"],eval_metric=e_metric,verbose=500,early_stopping_rounds=250)

        

        valid_scores_list.append(lgbr.best_score_["valid"]["lgb_rmse"])



        feature_importance_df[f"{i+1}"] = lgbr.feature_importances_

        

        if logarithm:

            valid_score = np.expm1(lgbr.predict(X_valid))

            prediction = np.expm1(lgbr.predict(X_test))

        

        else:

            valid_score = lgbr.predict(X_valid)

            prediction = lgbr.predict(X_test)

        

        valid_scores[valid_indices] += valid_score

        predictions += prediction / nfolds

        

    valid_mean_score = np.mean(valid_scores_list)

    

    print(f"mean_valid_score is {valid_mean_score} at {nfolds}")

    

    feature_importance_df["mean"] = feature_importance_df[feature_importance_df.columns.values[1:]].mean(axis=1)

    feature_importance_df["std"] = feature_importance_df[feature_importance_df.columns.values[1:]].mean(axis=1)

    

    fi_sorted = feature_importance_df.sort_values("mean",ascending=False)

    

    plt.figure(figsize=[6,40])

    sns.barplot(x="mean",y="features",data=fi_sorted,xerr=fi_sorted["std"])

    plt.title("Feature Importances of lightgbm",fontsize=12)

    plt.show()

        

    return valid_mean_score, predictions

tuned_params = {

    'boosting_type': 'gbrt',

    'class_weight': None,

    'colsample_bytree': 0.8424667117862588,

    'learning_rate': 0.11657835160316778,

    'max_depth': 21,

    'min_child_samples': 30,

    'min_child_weight': 2.7488547729054593,

    'n_estimators': 2160, #1080

    'num_leaves': 9,

    'objective': 'regression',

    'random_state': 101,

    'reg_alpha': 0.6915673059398951,

    'reg_lambda': 0.6362045095817355,

    'subsample': 0.8931910384738333,

    'subsample_for_bin': 100000,

    'subsample_freq': 4}
valid_score, predictions = model_lgb(X_train=X_adj_log_train,X_test=X_adj_log_test,y_train=y_adj_log_train,hyperparameters=tuned_params,logarithm=True)
target = "lgb_uni_5"

scores_dict[target] = valid_score

predictions_dict[target] = predictions
submission = pd.read_csv(PATH+"sample_submission.csv")

submission["price"] = predictions

submission.to_csv("lgb_uni_layer_5.csv",index=False)
valid_score, predictions = model_lgb(X_train=X_adj_log_train,X_test=X_adj_log_test,y_train=y_adj_log_train,hyperparameters=tuned_params,logarithm=True,nfolds=10)
target = "lgb_uni_10"

scores_dict[target] = valid_score

predictions_dict[target] = predictions
submission = pd.read_csv(PATH+"sample_submission.csv")

submission["price"] = predictions

submission.to_csv("lgb_uni_layer_10.csv",index=False)
def multiple_model_lgb(X_train,y_train,X_test,nfolds=5,hyperparameters=None,logarithm=False,seeds=[101]):

    

    #훈련에사용하는 컬럼들

    feature_names = X_train.columns.values

    

    #seed별로 모델의 feature_importance 저장을 위한 딕셔너리

    lgb_fi_dict = dict()

    #stack된 예측값을 저장하기 위한 변수

    stacked_predictions = np.zeros(len(X_test))

    #stack된 모델들의 점수를 모아두기 위한 배열

    total_best_valid_scores = []

    

    #시드별로 모델을 생성하며 모델에 대한 feature_importance, 예측값 그리고 valid_score를 위의 변수들에 기록함.  

    for k,seed in enumerate(seeds):



        valid_scores = np.zeros(len(X_train))

        predictions = np.zeros(len(X_test))



        best_valid_scores = []



        importances = np.zeros(len(feature_names))



        feature_importance_df = pd.DataFrame()

        feature_importance_df["features"] = feature_names



        if hyperparameters:

            params = hyperparameters



        else:

            params = {'boosting_type': 'gbdt',

                     'class_weight': None,

                     'colsample_bytree': 1.0,

                     'importance_type': 'split',

                     'learning_rate': 0.1,

                     'max_depth': -1,

                     'min_child_samples': 20,

                     'min_child_weight': 0.001,

                     'min_split_gain': 0.0,

                     'n_estimators': 1000,

                     'n_jobs': -1,

                     'num_leaves': 31,

                     'objective': "rmse",

                     'random_state': 101,

                     'reg_alpha': 0.0,

                     'reg_lambda': 0.0,

                     'silent': True,

                     'subsample': 1.0,

                     'subsample_for_bin': 200000,

                     'subsample_freq': 0}



        params["random_state"] = seed

        

        lgbr = lgb.LGBMRegressor(**params)



        if logarithm:

            fold = KFold(n_splits=nfolds,shuffle=True,random_state=12)

            e_metric = log_lgb_rmse

        else:

            fold = StratifiedKFold(n_splits=nfolds,shuffle=True,random_state=12)

            e_metric = lgb_rmse



        print(f"Current seed is {seed} of {k+1} elements","#"*15)

        print(params)



        for i,(train_indices,valid_indices) in enumerate(fold.split(X_train.values,y_train.values)):



            X = X_train.values[train_indices]

            y = y_train.values[train_indices]

            X_valid = X_train.values[valid_indices]

            y_valid = y_train.values[valid_indices]



            print("{} fold processing".format(i+1),"#"*20)



            lgbr.fit(X,y,eval_set=[(X,y),(X_valid,y_valid)],eval_names=["train","valid"],eval_metric=e_metric,verbose=500,early_stopping_rounds=250)



            best_valid_scores.append(lgbr.best_score_["valid"]["lgb_rmse"])



            feature_importance_df[f"{i+1}"] = lgbr.feature_importances_



            if logarithm:

                valid_score = np.expm1(lgbr.predict(X_valid))

                prediction = np.expm1(lgbr.predict(X_test))



            else:

                valid_score = lgbr.predict(X_valid)

                prediction = lgbr.predict(X_test)



            valid_scores[valid_indices] += valid_score

            predictions += prediction / nfolds



        valid_mean_score = np.mean(best_valid_scores)    

        print(f"mean_valid_score is {valid_mean_score} at {nfolds}")

        total_best_valid_scores.append(valid_mean_score)

    

        feature_importance_df["mean"] = feature_importance_df[feature_importance_df.columns.values[1:]].mean(axis=1)

        feature_importance_df["std"] = feature_importance_df[feature_importance_df.columns.values[1:]].mean(axis=1)

    

        lgb_fi_dict[seed] = feature_importance_df

        stacked_predictions += predictions / len(seeds)

    

    #stacking된 결과 값들을 저장하고 plot을 그리거나 return해주기 위한 과정.

    

    stacked_importances = pd.DataFrame()

    stacked_importances["features"] = feature_names

    stacked_importances["mean"] = 0

    stacked_importances["std"] = 0

    

    for key,values in lgb_fi_dict.items():

        stacked_importances["mean"] += values["mean"]/len(seeds)

        stacked_importances["std"] += values["std"]/len(seeds)

    

    fi_sorted= stacked_importances.sort_values(by="mean",ascending=False)

    

    stacked_valid_mean_score = np.mean(total_best_valid_scores)

    

    print(f"Stacked valid_score of {seeds} is {stacked_valid_mean_score} at {nfolds}")

    

    plt.figure(figsize=[6,40])

    sns.barplot(x="mean",y="features",data=fi_sorted,xerr=fi_sorted["std"])

    plt.title("Feature Importances of lightgbm",fontsize=12)

    plt.show()

        

    return stacked_valid_mean_score,stacked_predictions

valid_score, predictions = multiple_model_lgb(X_train=X_adj_log_train,X_test=X_adj_log_test,y_train=y_adj_log_train,hyperparameters=tuned_params,logarithm=True,seeds=[101,200,27,1085,567])
target = "lgb_multiple_5"

scores_dict[target] = valid_score

predictions_dict[target] = predictions
submission = pd.read_csv(PATH+"sample_submission.csv")

submission["price"] = predictions

submission.to_csv("lgb_multiple_layer_5.csv",index=False)
valid_score, predictions = multiple_model_lgb(X_train=X_adj_log_train,X_test=X_adj_log_test,y_train=y_adj_log_train,nfolds=10,hyperparameters=tuned_params,logarithm=True,seeds=[101,200,27,1085,567])
target = "lgb_multiple_10"

scores_dict[target] = valid_score

predictions_dict[target] = predictions
submission = pd.read_csv(PATH+"sample_submission.csv")

submission["price"] = predictions

submission.to_csv("lgb_multiple_layer_10.csv",index=False)
result_df = np.transpose(pd.DataFrame(scores_dict,index=range(1))).rename(columns={0:"rmse_score"}).sort_values(by="rmse_score")

result_df = result_df.reset_index().rename(columns={"index":"model"})

sns.barplot(x="rmse_score",y="model",data=result_df)
target1 = "xgb_uni_10"

target2 = "lgb_multiple_10"



ensembled_predictions = predictions_dict[target1] * 0.5 + predictions_dict[target2] * 0.5

ensembled_score = scores_dict[target1] * 0.5 + scores_dict[target2] * 0.5



predictions_dict["ensemble1"] = ensembled_predictions

scores_dict["ensemble1"] = ensembled_score



submission = pd.read_csv(PATH+"sample_submission.csv")

submission["price"] = ensembled_predictions

submission.to_csv("xgb_lgb_ensemble.csv",index=False)
result_df = np.transpose(pd.DataFrame(scores_dict,index=range(1))).rename(columns={0:"rmse_score"}).sort_values(by="rmse_score")

result_df = result_df.reset_index().rename(columns={"index":"model"})

sns.barplot(x="rmse_score",y="model",data=result_df)