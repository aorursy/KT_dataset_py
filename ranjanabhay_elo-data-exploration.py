# ELO 
import pandas as pd
import numpy as np
import scipy as sc
from matplotlib import pyplot as plt
#Reading Sample Training Data
elo_train_data = pd.read_csv("../input/train.csv",parse_dates=['first_active_month'])
elo_train_data.head(n=5)
#Reading Ssmple Test Data
elo_test_data = pd.read_csv("../input/test.csv",parse_dates=['first_active_month'])
elo_test_data.head(n=5)
# Reading Elo Historical Transactions Data
elo_historical_transactions_data = pd.read_csv("../input/historical_transactions.csv")
elo_historical_transactions_data.head(n=5)
# Reading ELO merchants data

elo_merchants_data = pd.read_csv("../input/merchants.csv")

elo_merchants_data.head(n=5)
# Reading ELO new merchants transaction datas

elo_merchant_transactions_data = pd.read_csv("../input/new_merchant_transactions.csv")
elo_merchant_transactions_data.head(n=5)
# Shape of training dataset.
print(elo_train_data.shape)
# Shape of Test Data
print(elo_test_data.shape)
import seaborn as sns
plt.figure(figsize=(24,32))
target_col="target"
sns.distplot(elo_train_data[target_col],bins=40,kde=False,color="Green")
plt.title("Histogram of Loyalty Score")
plt.xlabel("Loyalty Score")
plt.show()
# First Active Month Analysis
cnt_srs = elo_train_data['first_active_month'].dt.date.value_counts()
#cnt_srs.head(n=5)
cnt_srs.sort_index(inplace=True)
plt.figure(figsize=(14,6))
print(cnt_srs.index)
print(cnt_srs.values)
sns.barplot(cnt_srs.index,cnt_srs.values,alpha=0.8,color='green')
plt.xticks(rotation='vertical')
plt.xlabel('First Active Month',fontsize=12)
plt.ylabel("First Active Month Count in Train Set")
plt.show()
plt.figure(figsize=(8,4))
sns.violinplot(x='feature_1',y=target_col,data=elo_train_data)
plt.xticks(rotation='vertical')
plt.xlabel("Feature-1")
plt.ylabel("Loyalty Score")
plt.title("Feature 1 Distribution")
plt.show()
elo_cumulative_historical_transact_data= elo_historical_transactions_data.groupby(['card_id'])
elo_cumulative_historical_transact_data = elo_cumulative_historical_transact_data['purchase_amount'].size().reset_index()
elo_cumulative_historical_transact_data.columns=['card_id','num_historical_transactions']
elo_cumulative_historical_transact_data.head(n=5)
elo_train_data = pd.merge(elo_train_data,elo_cumulative_historical_transact_data,on='card_id',how="left")
elo_train_data.head(n=5)
elo_test_data = pd.merge(elo_test_data,elo_cumulative_historical_transact_data,on='card_id',how='left')
elo_test_data.head(n=5)
cnt_srs = elo_train_data.groupby("num_historical_transactions")[target_col].mean()
#print(cnt_srs.head(n=5))
sns_srs = cnt_srs.sort_index()
#print(cnt_srs.head(n=5))
cnt_srs = cnt_srs[:-50]
print(cnt_srs.head(n=50))
from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
%matplotlib inline
def scatter_plot(cnt_srs,color):
    trace = go.Scatter(
        x=cnt_srs.index[::-1],
        y=cnt_srs.values[::-1],
        showlegend=False,
        marker=dict(
            color = color,
        ),
    )
    return trace
        
trace = scatter_plot(cnt_srs,"orange")
layout = dict(title="Loyalty Score by Number of Historical Transactions",)
data = [trace]
fig = go.Figure(data=data,layout=layout)
py.iplot(fig,filename="Histtransact")
bins = [0,10,20,30,40,50,75,100,150,200,500,10000]
elo_train_data['binned_number_hist_transactions'] = pd.cut(elo_train_data['num_historical_transactions'],bins)
cnt_srs = elo_train_data.groupby("binned_number_hist_transactions")[target_col].mean()
plt.figure(figsize=(12,8))
sns.boxplot(x="binned_number_hist_transactions",y=target_col,data=elo_train_data,showfliers=False)
plt.xticks(rotation='vertical')
plt.xlabel("binned_number_hist_transactions",fontsize=12)
plt.ylabel('Loyalty Score',fontsize=12)
plt.title("binned_number_hist_transactions_distribution")
plt.show()
elo_cumulative_historical_transact_data =elo_historical_transactions_data.groupby("card_id")
elo_cumulative_historical_transact_data = elo_cumulative_historical_transact_data["purchase_amount"].agg(['sum','min','std','mean','max']).reset_index()
elo_cumulative_historical_transact_data.columns = ['card_id','sum_historical_transactions','min_historical_transactions','std_historical_transactions','mean_historical_transactions','max_historical_transactions']
elo_train_data = pd.merge(elo_train_data,elo_cumulative_historical_transact_data,on="card_id",how="left")
elo_test_data = pd.merge(elo_test_data,elo_cumulative_historical_transact_data,on="card_id",how="left")
elo_train_data.head(n=5)
bins = np.percentile(elo_train_data['sum_historical_transactions'],range(0,101,10))
elo_train_data['binned_sum_historical_transactions'] = pd.cut(elo_train_data['sum_historical_transactions'],bins)
plt.figure(figsize=(12,8))
sns.boxplot(x="binned_sum_historical_transactions",y=target_col,data=elo_train_data,showfliers=False)
plt.xticks(rotation='vertical')
plt.xlabel("binned_sum_historical_transactions",fontsize=12)
plt.ylabel('Loyalty Score',fontsize=12)
plt.title("Sum of Historical Transaction value(Binned) distribution")
plt.show()
# Merchant Transactions
elo_merchant_transactions_data.head(n=5)
elo_cumulative_merchant_transactions_data = elo_merchant_transactions_data.groupby("card_id")
elo_cumulative_merchant_transactions_data = elo_cumulative_merchant_transactions_data["purchase_amount"].size().reset_index()
#elo_cumulative_merchant_transactions_data.head(n=5)
elo_cumulative_merchant_transactions_data.columns=['card_id','num_merchant_transactions']
elo_train_data = pd.merge(elo_train_data,elo_cumulative_merchant_transactions_data,on="card_id",how="left")
elo_test_data = pd.merge(elo_test_data,elo_cumulative_merchant_transactions_data,on="card_id",how="left")
elo_train_data.head(n=5)
bins = [0,10,20,30,40,50,75,1000]
elo_train_data["binned_number_merchant_transactions"] = pd.cut(elo_train_data['num_merchant_transactions'],bins)
cnt_srs = elo_train_data.groupby("binned_number_merchant_transactions")[target_col].mean()
plt.figure(figsize=(12,8))
sns.boxplot(x="binned_number_merchant_transactions",y=target_col,data=elo_train_data,showfliers=False)
plt.xticks(rotation="vertical")
plt.xlabel("binned_number_merchant_transactions",fontsize=12)
plt.ylabel("Loyalty Score",fontsize=12)
plt.title("Number Of New Merchants Transaction Distribution(binned)")
plt.show()
elo_cumulative_merchant_transactions_purchase_data = elo_merchant_transactions_data.groupby("card_id")
elo_cumulative_merchant_transactions_purchase_data = elo_cumulative_merchant_transactions_purchase_data["purchase_amount"].agg(['sum','mean','std','min','max']).reset_index()
elo_cumulative_merchant_transactions_purchase_data.columns = ["card_id","sum_merchant_transactions","mean_merchant_transactions","std_merchant_transactions","min_merchant_transactions","max_merchant_transactions"]
elo_train_data = pd.merge(elo_train_data,elo_cumulative_merchant_transactions_purchase_data,on="card_id",how="left")
elo_test_data = pd.merge(elo_test_data,elo_cumulative_merchant_transactions_purchase_data,on="card_id",how="left")
elo_train_data.head(n=5)
bins = np.nanpercentile(elo_train_data["sum_merchant_transactions"],range(0,101,10))
elo_train_data["binned_sum_merchant_transactions"] = pd.cut(elo_train_data["sum_merchant_transactions"],bins)
plt.figure(figsize=(12,8))
sns.boxplot(x="binned_sum_merchant_transactions",y=target_col,data=elo_train_data,showfliers=False)
plt.xticks(rotation="vertical")
plt.xlabel("Binned Sum New Merchant Transactions",fontsize=12)
plt.ylabel("Loyalty Score")
plt.title("SUm Of New Merchant Transactions Value Distribution")
plt.show()
#BaselineModel
elo_train_data["year"] = elo_train_data["first_active_month"].dt.year
elo_test_data["year"] = elo_test_data["first_active_month"].dt.year
elo_train_data["month"] = elo_train_data["first_active_month"].dt.month
elo_test_data["month"] = elo_test_data["first_active_month"].dt.month
elo_train_data.columns
columns_to_use = ["feature_1","feature_2","feature_3","year","month","num_historical_transactions",
                 "sum_historical_transactions","mean_historical_transactions","std_historical_transactions",
                 "min_historical_transactions","max_historical_transactions","num_merchant_transactions",
                 "sum_merchant_transactions","mean_merchant_transactions","std_merchant_transactions",
                  "min_merchant_transactions","max_merchant_transactions"]
import lightgbm as lgb
from sklearn import model_selection,preprocessing,metrics

def run_light_gradient_boosting(train_X,train_Y,val_X,val_Y,test_X):
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_leaves":30,
        "min_child_weight":50,
        "learning_rate":0.05,
        "bagging_fraction":0.7,
        "feature_fraction":0.7,
        "bagging_seed":2018,
        "verbosity":-1
    }
    
    lgtrain = lgb.Dataset(train_X,label=train_Y)
    lgval = lgb.Dataset(val_X,label=val_Y)
    evals_result ={}
    model = lgb.train(params,lgtrain,1000,valid_sets=lgval,early_stopping_rounds=200,verbose_eval=100,evals_result=evals_result)
    pred_test_Y = model.predict(test_X,num_iteration=model.best_iteration)
    return pred_test_Y,model,evals_result
train_X = elo_train_data[columns_to_use]
test_X = elo_test_data[columns_to_use]
train_Y = elo_train_data[target_col].values

pred_test = 0
kf = model_selection.KFold(n_splits=5,random_state=2018,shuffle=True)
for dev_index,val_index in kf.split(elo_train_data):
    dev_X,val_X = train_X.loc[dev_index,:] ,train_X.loc[val_index,:]
    dev_Y,val_Y = train_Y[dev_index] ,train_Y[val_index]
    pred_test_tmp, model,evals_result = run_light_gradient_boosting(dev_X,dev_Y,val_X,val_Y,test_X)
    pred_test += pred_test_tmp
    
pred_test /=5
fig,ax = plt.subplots(figsize=(12,10))
lgb.plot_importance(model,max_num_features=50,height=0.8,ax=ax)
ax.grid(False)
plt.title("Light GBM- Feature Importance",fontsize=15)
plt.show()
sub_df = pd.DataFrame({"card_id":elo_test_data["card_id"].values})
sub_df["target"] = pred_test
sub_df.to_csv("lgb_output.csv",index=False)