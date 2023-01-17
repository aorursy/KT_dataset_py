# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
import warnings
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split
import datetime
import lightgbm as lgb
sns.set(style='darkgrid', palette='deep')
warnings.filterwarnings('ignore')
%matplotlib inline
# Load train and test data
train = pd.read_csv("../input/train.csv", parse_dates=["first_active_month"])
test = pd.read_csv("../input/test.csv", parse_dates=["first_active_month"])
# Load additional data
merchants = pd.read_csv("../input/merchants.csv")
new_trans = pd.read_csv("../input/new_merchant_transactions.csv", 
                        parse_dates=['purchase_date'])
hist_trans = pd.read_csv("../input/historical_transactions.csv", 
                         parse_dates=['purchase_date'])
temp_df = hist_trans.groupby("card_id")
temp_df = temp_df["purchase_amount"].size().reset_index()
temp_df.columns = ["card_id", "num_hist_transactions"]
train_temp = pd.merge(train, temp_df, on="card_id", how="left")
test_temp = pd.merge(test, temp_df, on="card_id", how="left")
temp_df.to_csv("temp_hist_eda.csv", index=False)
temp_df.head().sort_values(by='num_hist_transactions', ascending=False)
temp_df = hist_trans.groupby("card_id")
temp_df = temp_df["purchase_amount"].agg(['sum', 'mean', 'std', 'min', 'max']).reset_index()
temp_df.columns = ["card_id", "sum_hist_trans", "mean_hist_trans", "std_hist_trans", "min_hist_trans", "max_hist_trans"]
train_temp = pd.merge(train, temp_df, on="card_id", how="left")
test_temp = pd.merge(test, temp_df, on="card_id", how="left")
bins = np.percentile(train_temp["sum_hist_trans"], range(0,101,10))
train_temp['binned_sum_hist_trans'] = pd.cut(train_temp['sum_hist_trans'], bins)


plt.figure(figsize=(12,8))
sns.boxplot(x="binned_sum_hist_trans", y="target", data=train_temp, showfliers=False)
plt.xticks(rotation='vertical')
plt.xlabel('binned_sum_hist_trans', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Sum of historical transaction value (Binned) distribution")
plt.show()
del temp_df
del train_temp
del test_temp
df_temp = new_trans.groupby("card_id")
df_temp = df_temp["purchase_amount"].size().reset_index()
df_temp.columns = ["card_id", "num_merch_transactions"]
train_temp = pd.merge(train, df_temp, on="card_id", how="left")
test_temp = pd.merge(test, df_temp, on="card_id", how="left")
df_temp.to_csv("temp_new_eda.csv", index=False)
df_temp.head().sort_values(by='num_merch_transactions', ascending=False)
bins = [0, 10, 20, 30, 40, 50, 75, 10000]
train_temp['binned_num_merch_transactions'] = pd.cut(train_temp['num_merch_transactions'], bins)

plt.figure(figsize=(12,8))
sns.boxplot(x="binned_num_merch_transactions", y="target", data=train_temp, showfliers=False)
plt.xticks(rotation='vertical')
plt.xlabel('binned_num_merch_transactions', fontsize=12)
plt.ylabel('Loyalty score', fontsize=12)
plt.title("Number of new merchants transaction (Binned) distribution")
plt.show()
del df_temp
del train_temp
del test_temp