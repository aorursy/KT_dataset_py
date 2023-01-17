import pandas as pd,numpy as np,matplotlib.pyplot as plt,seaborn as sns
import os
import warnings
import scipy
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns",200)
import datetime
import gc
import xgboost as xgb
# Reading training data
train = pd.read_csv("/kaggle/input/train.csv")
# Making sure there is no duplicate card id
train.shape[0] == train["card_id"].nunique()
# Checking for Null values
train.isnull().sum()
# break down first active month into year and month
train["active_year"],train["active_month"] = list(zip(*train["first_active_month"].apply(lambda x:x.split("-"))))
# unique values of features
train.nunique()
# histogram of target
plt.figure(figsize=(14,14))
sns.distplot(train["target"])
# box plot of target based on features present
feature = ["active_year","active_month","feature_1","feature_2","feature_3"]
f,ax=plt.subplots(1,5)
f.set_figheight(8)
f.set_figwidth(20)
for k,i in enumerate(feature):
    sns.boxplot(x= i,y="target",data = train,ax=ax[k])
# correlation heat map of features
train[["active_year","active_month"]] = train[["active_year","active_month"]].astype(int)
corr = train.drop(["card_id","first_active_month"],axis=1).corr()
plt.figure(figsize =(10,6))
sns.heatmap(corr,annot=True)
# Reading test dataset
test = pd.read_csv("/kaggle/input/test.csv")
test.head()
# checking for null values
test.isnull().sum()
# replacing one null with mode of first active month of train set
impute = train["first_active_month"].value_counts()[:1].index[0]
test = test.fillna(impute)

test["active_year"],test["active_month"] = list(zip(*test["first_active_month"].apply(lambda x:x.split("-"))))
test[["active_year","active_month"]] = test[["active_year","active_month"]].astype(int)

# created elapsed features using the following kernel
#https://www.kaggle.com/tunguz/eloda-with-feature-engineering-and-stacking
train["first_active_month"] = pd.to_datetime(train["first_active_month"])
train['elapsed_time'] = (datetime.date(2018, 2, 1) - train['first_active_month'].dt.date).dt.days
test["first_active_month"] = pd.to_datetime(test["first_active_month"])
test['elapsed_time'] = (datetime.date(2018, 2, 1) - test['first_active_month'].dt.date).dt.days

# Reading historical transaction
historical_trans =pd.read_csv("/kaggle/input/historical_transactions.csv")
historical_trans.head()
historical_trans.isnull().sum()
print(historical_trans["installments"].unique())
# assuming -1 as missing data and 999 as wrong entry, replacing both with 0, i.e mode of remaining data
historical_trans["installments"] = historical_trans["installments"].replace([-1,999],[0,0],inplace=False)

# Created binary feature from installment 
historical_trans["has_installments"] = historical_trans["installments"].apply(lambda x: "No" if x==0 else "Yes")
new_merchant_transactions = pd.read_csv("/kaggle/input/new_merchant_transactions.csv")
new_merchant_transactions.head()
# assuming -1 as missing data and 999 as wrong entry, replacing both with 0, i.e mode of remaining data
new_merchant_transactions["installments"] = new_merchant_transactions["installments"].replace([-1,999],[0,0],inplace=False)
new_merchant_transactions["has_installments"] = new_merchant_transactions["installments"].apply(lambda x: "No" if x==0 else "Yes")
# Imputing missing values 
def impute(df):
    df["merchant_id"] = df["merchant_id"].fillna("Missing_id")
    features = df.columns[df.isna().any()].tolist()
    for i in features:
        if df[i].dtype =="object" or i == "category_2":
            mode = df[i].value_counts()[:1].index[0]
            df[i].fillna(mode,inplace=True)
        else:
            df[i].fillna(df[i].mean(),inplace=True)
    return df

new_merchant_transactions = impute(new_merchant_transactions)
historical_trans = impute(historical_trans)
# mapping of binomial feature to (0,1)
for i in ["authorized_flag","category_1"]:
    new_merchant_transactions[i] = new_merchant_transactions[i].map({"Y":1,"N":0})
    historical_trans[i] = historical_trans[i].map({"Y":1,"N":0})

historical_trans.loc[:, 'purchase_date'] = pd.DatetimeIndex(historical_trans['purchase_date']).astype(np.int64) * 1e-9
new_merchant_transactions.loc[:, 'purchase_date'] = pd.DatetimeIndex(new_merchant_transactions['purchase_date']).astype(np.int64) * 1e-9

# aggregation for numerical features
aggs = {'month_lag':["min","max","median",'sum'],
       'purchase_amount':['mean','var'],
       'category_2':['min','max','median'],
    'installments':['min','max','median','sum'],
        "authorized_flag":['sum','mean'],
        "category_1":['sum','mean'],
        'merchant_id':["nunique"],'merchant_category_id':["nunique"]
        }

new_columns_hist = [k + '_hist_' + agg for k in aggs.keys() for agg in aggs[k]]
new_columns_new = [k + '_new_' + agg for k in aggs.keys() for agg in aggs[k]]

# new transaction data
aggs_merge2 = new_merchant_transactions.groupby("card_id").aggregate(aggs)
aggs_merge2 = aggs_merge2.fillna(0) # card id with one entry will have nan variance.
aggs_merge2.columns = new_columns_new
aggs_merge2["new_purchase_date"] = new_merchant_transactions.groupby("card_id").aggregate({'purchase_date':[np.ptp]}).values
del new_merchant_transactions
gc.collect()

# past data
aggs_merge_1 = historical_trans.groupby("card_id").aggregate(aggs)
aggs_merge_1 = aggs_merge_1.fillna(0) # card id with one entry will have nan variance.
aggs_merge_1.columns = new_columns_hist
aggs_merge_1["history_purchase_date"] = historical_trans.groupby("card_id").aggregate({'purchase_date':[np.ptp]}).values
del historical_trans
gc.collect()
final_train = train.merge(aggs_merge_1,on="card_id",how="left")
final_train = final_train.merge(aggs_merge2,on="card_id",how="left")

final_test = test.merge(aggs_merge_1,on="card_id",how="left")
final_test = final_test.merge(aggs_merge2,on="card_id",how="left")
del train,test
gc.collect()
# # imputation in merge dataset
imputed_columns = final_train.columns[final_train.isna().any()].tolist()
for i in imputed_columns:
    final_train[i].fillna(final_train[i].mean(),inplace=True)
    final_test[i].fillna(final_train[i].mean(),inplace=True)
# finding features with only one unique value and dropping them
one_value_features =[]
for i in final_train.columns:
    if final_train[i].nunique() ==1:
        one_value_features.append(i)
        
final_train = final_train.drop(one_value_features,axis=1)
final_test = final_test.drop(one_value_features,axis=1)
# Correlation Analysis of features
# top 10 features correlated with target
correlation = final_train.drop(["first_active_month","card_id"],axis=1).corr()
correlation["target"].abs().sort_values(ascending=False)[:10]

# calculating correlation of all features and defining them "important" based on their P-value
pairwise_corr=[]
value=[]
continuous_columns=[]
for i in final_train.drop(["target","card_id","first_active_month"],axis=1).columns:
    c, p = scipy.stats.pearsonr(final_train[i],final_train["target"])
    continuous_columns.append(i)
    value.append(p)
    pairwise_corr.append(c)
df=pd.DataFrame({"column":continuous_columns,"corr_value":pairwise_corr,"p-value":value})
df["importance"]=df["p-value"].apply(lambda x:"important" if x<0.05 else "not important")
plt.figure(figsize=(20,10))
plt.title("p value of predictor with target variable")
plt.xticks(rotation="vertical")
sns.stripplot(x="column",y="p-value",hue="importance",data=df,size=10)
y_train = final_train["target"]
final_train = final_train.sample(frac=1, random_state = 7)
x_train = final_train.drop(["target","card_id","first_active_month"],axis=1)
x_test = final_test.drop(["card_id","first_active_month"],axis=1)
from sklearn.model_selection import train_test_split as tts
Trn_x,val_x,Trn_y,val_y = tts(x_train,y_train,test_size =0.1,random_state = 7)
trn_x , test_x, trn_y, test_y = tts(Trn_x , Trn_y, test_size =0.1, random_state = 7)
# converting into xgb DMatrix
Train = xgb.DMatrix(trn_x,label = trn_y)
Validation = xgb.DMatrix(val_x, label = val_y)
Test = xgb.DMatrix(test_x)

params = {"booster":"gbtree","eta":0.1,'min_split_loss':0,'max_depth':6,
         'min_child_weight':1, 'max_delta_step':0,'subsample':1,'colsample_bytree':1,
         'colsample_bylevel':1,'reg_lambda':1,'reg_alpha':0,
         'grow_policy':'depthwise','max_leaves':0,'objective':'reg:linear','eval_metric':'rmse',
         'seed':7}
history ={}  # This will record rmse score of training and test set
eval_list =[(Train,"Training"),(Validation,"Validation")]

clf = xgb.train(params, Train, num_boost_round=119, evals=eval_list, obj=None, feval=None, maximize=False, 
          early_stopping_rounds=40, evals_result=history)
# dataframe of progress
f,ax=plt.subplots(1,1)
f.set_figheight(10)
f.set_figwidth(20)
df_performance=pd.DataFrame({"train":history["Training"]["rmse"],"test":history["Validation"]["rmse"]}).reset_index(drop=False)
sns.pointplot(ax=ax,y="train",x="index",data=df_performance,color="r")
sns.pointplot(ax=ax,y="test",x="index",data=df_performance,color="g")
ax.legend(handles=ax.lines[::len(df_performance)+1], labels=["Train","Test"])
plt.xlabel('iterations'); plt.ylabel('logloss value'); plt.title('learning curve')
score=clf.get_score(importance_type="gain")
df=pd.DataFrame({"feature":list(score.keys()),"score":list(score.values())})
df=df.sort_values(by="score",ascending=False)
plt.figure(figsize=(20,20))
plt.xticks(rotation="vertical")
sns.barplot(x="feature",y="score",data=df,orient="v")

# Checking rmse on test set (kept during data splitting)
from sklearn.metrics import mean_squared_error as mse
pred_test = clf.predict(Test)
score = mse(test_y , pred_test)
print(np.sqrt(score))
prediction = clf.predict(xgb.DMatrix(x_test))
df_sub=pd.DataFrame()
df_sub["card_id"] = final_test["card_id"].values
df_sub["target"] = np.ravel(prediction)
df_sub[["card_id","target"]].to_csv("new_submission.csv",index=False)