from IPython.display import Image
from IPython.core.display import HTML 
Image(url= "https://storage.googleapis.com/kaggle-competitions/kaggle/10445/logos/thumb76_76.png?t=2018-10-24-17-14-05")
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (20,10)
import seaborn as sns
from scipy.stats import norm
import warnings
warnings.filterwarnings('ignore')
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
merchants = pd.read_csv('../input/merchants.csv')
hist_tran = pd.read_csv('../input/historical_transactions.csv')
new_merc_tran = pd.read_csv('../input/new_merchant_transactions.csv')

data_dict_train=pd.read_excel('../input/Data_Dictionary.xlsx',sheet_name='train')
data_dict_hist_tran=pd.read_excel('../input/Data_Dictionary.xlsx',sheet_name='history')
data_dict_new_merc_tran=pd.read_excel('../input/Data_Dictionary.xlsx',sheet_name='new_merchant_period')
data_dict_merchants=pd.read_excel('../input/Data_Dictionary.xlsx',sheet_name='merchant')
train.head()
plt.figure(figsize = (12,7))
sns.distplot(train['target'], fit = norm);
plt.xlabel('Loyalty Score',fontsize = 14);
print('Skewness of Target is :',train.target.skew())
print('Kurtosis of Traget is :',train.target.kurt())
plt.figure(figsize = (10,7))
sns.heatmap(train.corr(),annot = True,linewidths = 0.5,cmap='cubehelix_r');
train.info()
data_dict_train
plt.figure(figsize = (20,7));
plt.subplot(121)
sns.violinplot(train.feature_1,train.target);
plt.xlabel('feature_1',fontsize = 14);
plt.ylabel('target',fontsize = 14);
plt.subplot(122)
train['feature_1'].value_counts().plot(kind='bar');
plt.xlabel('feature_1',fontsize = 14);
plt.ylabel('target',fontsize = 14);
plt.figure(figsize = (20,7));
plt.subplot(121)
sns.violinplot(train.feature_2,train.target);
plt.xlabel('feature_2',fontsize = 14);
plt.ylabel('target',fontsize = 14);
plt.subplot(122)
train['feature_2'].value_counts().plot(kind='bar');
plt.xlabel('feature_2',fontsize = 14);
plt.ylabel('target',fontsize = 14);
plt.figure(figsize = (20,7));
plt.subplot(121)
sns.violinplot(train.feature_3,train.target);
plt.xlabel('feature_3',fontsize = 14);
plt.ylabel('target',fontsize = 14);
plt.subplot(122)
train['feature_3'].value_counts().plot(kind='bar');
plt.xlabel('feature_3',fontsize = 14);
plt.ylabel('target',fontsize = 14);
plt.figure(figsize = (20,25))
plt.subplot(211)
train['first_active_month'].value_counts().sort_index().plot(kind = 'bar');
plt.xlabel('First_active_month',fontsize = 14);
plt.ylabel('Count',fontsize = 14);
plt.title('First Active Month in Training Data',fontsize = 18);
plt.subplot(212)
test['first_active_month'].value_counts().sort_index().plot(kind = 'bar');
plt.xlabel('First_active_month',fontsize = 14);
plt.ylabel('Count',fontsize = 14);
plt.title('First Active Month in Test Data',fontsize = 18);
train_lesser_m20 = train[train['target']<-20]
train_lesser_m20['first_active_month'].value_counts().sort_index().plot(kind = 'bar');
plt.xlabel('First active month', fontsize=15);
plt.ylabel('Number of cards', fontsize=15);
plt.title("First active month count in target less than -20",fontsize=18);
train_lesser_m20
train.isna().sum()
test.isna().sum()
ax = sns.FacetGrid(train, hue="feature_3", col="feature_2", margin_titles=True,
                  palette={1:"red", 0:"green"} )
ax.map(plt.scatter, "first_active_month", "target",edgecolor="w").add_legend();


hist_tran.head()
new_merc_tran.head()
data_dict_hist_tran
data_dict_new_merc_tran
print('Authorized Flag Y if approved, N if denied \n',hist_tran['authorized_flag'].value_counts(),
     '\n Authorized Flag Y if approved, N if denied \n',new_merc_tran['authorized_flag'].value_counts())
plt.subplot(121)
hist_tran['authorized_flag'].value_counts().plot(kind = 'bar');
plt.xlabel('Authorization',fontsize = 15);
plt.ylabel('Values',fontsize = 15);
plt.title('Authorized Flag Y if approved, N if denied \n Historical Data',fontsize = 20);
plt.subplot(122)
new_merc_tran['authorized_flag'].value_counts().plot(kind = 'bar');
plt.xlabel('Authorization',fontsize = 15);
plt.ylabel('Values',fontsize = 15);
plt.title('Authorized Flag Y if approved, N if denied \n New Data',fontsize = 20);
print('Historical Category 3 \n',hist_tran['category_3'].value_counts(),
     '\n New Category 3 \n',new_merc_tran['category_3'].value_counts())
plt.subplot(121)
hist_tran['category_3'].value_counts().plot(kind = 'bar');
plt.xlabel('Category 3',fontsize = 15);
plt.ylabel('Values',fontsize = 15);
plt.title('Historical Data',fontsize = 20);
plt.subplot(122)
new_merc_tran['category_3'].value_counts().plot(kind = 'bar');
plt.xlabel('Category 3',fontsize = 15);
plt.ylabel('Values',fontsize = 15);
plt.title('New Data',fontsize = 20);
print('Historical Category 2 \n',hist_tran['category_2'].value_counts(),
     '\n New Category 2 \n',new_merc_tran['category_2'].value_counts())
plt.subplot(121)
hist_tran['category_2'].value_counts().plot(kind = 'bar');
plt.xlabel('Category 2',fontsize = 15);
plt.ylabel('Values',fontsize = 15);
plt.title('Historical Data',fontsize = 20);
plt.subplot(122)
new_merc_tran['category_2'].value_counts().plot(kind = 'bar');
plt.xlabel('Category 2',fontsize = 15);
plt.ylabel('Values',fontsize = 15);
plt.title('New Data',fontsize = 20);
print('Historical Category 1 \n',hist_tran['category_1'].value_counts(),
     '\n New Category 1 \n',new_merc_tran['category_1'].value_counts())
plt.subplot(121)
hist_tran['category_1'].value_counts().plot(kind = 'bar');
plt.xlabel('Category 1',fontsize = 15);
plt.ylabel('Values',fontsize = 15);
plt.title('Historical Data',fontsize = 20);
plt.subplot(122)
new_merc_tran['category_1'].value_counts().plot(kind = 'bar');
plt.xlabel('Category 1',fontsize = 15);
plt.ylabel('Values',fontsize = 15);
plt.title('New Data',fontsize = 20);
print('Historical Month Lag \n''\n',hist_tran['month_lag'].value_counts(),
     '\n New Month Lag \n''\n',new_merc_tran['month_lag'].value_counts())
plt.subplot(121)
hist_tran['month_lag'].value_counts().plot(kind = 'bar');
plt.xlabel('Month Lag',fontsize = 15);
plt.ylabel('Values',fontsize = 15);
plt.title('Month lag to reference date \n Historical Data',fontsize = 15);
plt.subplot(122)
new_merc_tran['month_lag'].value_counts().plot(kind = 'bar');
plt.xlabel('Month Lag',fontsize = 15);
plt.ylabel('Values',fontsize = 15);
plt.title('Month lag to reference date \n New Data',fontsize = 15);
print('Historical Installments \n''\n',hist_tran['installments'].value_counts(),
     '\n New Installments \n''\n',new_merc_tran['installments'].value_counts())
plt.subplot(121)
hist_tran['installments'].value_counts().plot(kind = 'bar');
plt.title('Number of Installments of Purchase \n Historical Data',fontsize = 20);
plt.subplot(122)
new_merc_tran['installments'].value_counts().plot(kind = 'bar');
plt.title('Number of Installments of Purchase \n New Data',fontsize = 20);

data_dict_merchants
sns.heatmap(merchants.corr(),annot = True);
print('Quantity of active months within Last 3 months \n',merchants['active_months_lag3'].value_counts(),
      '\n Quantity of active months within Last 6 months \n',merchants['active_months_lag6'].value_counts(),
      '\n Quantity of active months within Last 12 months \n',merchants['active_months_lag12'].value_counts())
plt.subplot(131)
merchants['active_months_lag3'].value_counts().plot(kind = 'bar');
plt.xlabel('active_months_lag3',fontsize = 14);
plt.title('Quantity of active months within Last 3 months',fontsize = 15);
plt.subplot(132)
merchants['active_months_lag6'].value_counts().plot(kind = 'bar');
plt.xlabel('active_months_lag6',fontsize = 14);
plt.title('Quantity of active months within Last 6 months',fontsize = 15);
plt.subplot(133)
merchants['active_months_lag12'].value_counts().plot(kind = 'bar');
plt.xlabel('active_months_lag12',fontsize = 14);
plt.title('Quantity of active months within Last 12 months',fontsize = 15);
plt.figure(figsize = (20,7))
plt.subplot(131)
plt.scatter(merchants['avg_sales_lag3'],merchants['avg_sales_lag6'],color = 'red');
plt.title('Average Sales in Lag 3 vs Lag 6',fontsize = 15);
plt.subplot(132)
plt.scatter(merchants['avg_sales_lag3'],merchants['avg_sales_lag12'],color = 'green');
plt.title('Average Sales in Lag 3 vs Lag 12',fontsize=15);
plt.subplot(133)
plt.scatter(merchants['avg_sales_lag6'],merchants['avg_sales_lag12'],color = 'green');
plt.title('Average Sales in Lag 6 vs Lag 12',fontsize=15);
plt.figure(figsize = (20,5))
plt.subplot(131)
sns.distplot(merchants['avg_sales_lag3'].value_counts(),fit = norm);
plt.title('avg_sales_lag3',fontsize = 15);
plt.subplot(132)
sns.distplot(merchants['avg_sales_lag6'].value_counts(),fit = norm);
plt.title('avg_sales_lag6',fontsize = 15);
plt.subplot(133)
sns.distplot(merchants['avg_sales_lag12'].value_counts(),fit = norm);
plt.title('avg_sales_lag12',fontsize = 15);
hist = hist_tran.groupby("card_id").size().reset_index().rename({0:'transactions'},axis=1)
new = new_merc_tran.groupby("card_id").size().reset_index().rename({0:'transactions'},axis=1)
print('Historical Transactions:  \n',hist.describe()," \n New Transactions  \n",new.describe())
plt.subplot(221)
sns.violinplot(hist['transactions']);
plt.xlabel('Historic Transactions',fontsize = 15);
plt.subplot(222)
sns.violinplot(new['transactions'],color = 'red');
plt.xlabel('New Transactions',fontsize = 15);
plt.subplot(223)
sns.distplot(hist['transactions'],fit = norm);
plt.xlabel('Historic Transactions',fontsize = 15);
plt.subplot(224)
sns.distplot(new['transactions'],fit = norm, color = 'red');
plt.xlabel('New Transactions',fontsize = 15);
total_trans = hist_tran.append(new_merc_tran)
total_trans.head()
