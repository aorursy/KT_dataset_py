# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print('List of files:')
print(os.listdir("../input"))
data_folder = '../input'

# Any results you write to the current directory are saved as output.
train = pd.read_csv(os.path.join(data_folder, 'train.csv'))
test = pd.read_csv(os.path.join(data_folder, 'test.csv'))

plt.figure(figsize=[4,3])
plt.bar([0, 1], [train.shape[0], test.shape[0]], edgecolor=[0.2]*3, color=(1,0,0,0.5))
plt.xticks([0,1], ['train rows', 'test rows'], fontsize=13)
plt.title('Number of rows in train.csv and test.csv', fontsize=15)
plt.tight_layout()
plt.show()
%matplotlib inline

plt.figure(figsize=[15,5])
plt.suptitle('Feature distributions in train.csv and test.csv', fontsize=20, y=1.1)
for num, col in enumerate(['feature_1', 'feature_2', 'feature_3', 'target']):
    plt.subplot(2, 4, num+1)
    if col is not 'target':
        v_c = train[col].value_counts() / train.shape[0]
        plt.bar(v_c.index, v_c, label=('train'), align='edge', width=-0.3, edgecolor=[0.2]*3)
        v_c = test[col].value_counts() / test.shape[0]
        plt.bar(v_c.index, v_c, label=('test'), align='edge', width=0.3, edgecolor=[0.2]*3)
        plt.title(col)
        plt.legend()
    else:
        plt.hist(train[col], bins = 100)
        plt.title(col)
    plt.tight_layout()
plt.tight_layout()
plt.show()
outliers = train.loc[train['target'] < -30]
non_outliers = train.loc[train['target'] >= -30]
print('{:d} outliers found (target < -30)'.format(outliers.shape[0]))

plt.figure(figsize=[10,5])
plt.suptitle('Outlier vs. non-outlier feature distributions', fontsize=20, y=1.1)

for num, col in enumerate(['feature_1', 'feature_2', 'feature_3', 'target']):
    if col is not 'target':
        plt.subplot(2, 3, num+1)
        v_c = non_outliers[col].value_counts() / non_outliers.shape[0]
        plt.bar(v_c.index, v_c, label=('non-outliers'), align='edge', width=-0.3, edgecolor=[0.2]*3)
        v_c = outliers[col].value_counts() / outliers.shape[0]
        plt.bar(v_c.index, v_c, label=('outliers'), align='edge', width=0.3, edgecolor=[0.2]*3)
        plt.title(col)
        plt.legend()

plt.tight_layout()
plt.show()
corrs = np.abs(train.corr())
np.fill_diagonal(corrs.values, 0)
plt.figure(figsize=[5,5])
plt.imshow(corrs, cmap='plasma', vmin=0, vmax=1)
plt.colorbar(shrink=0.7)
plt.xticks(range(corrs.shape[0]), list(corrs.columns))
plt.yticks(range(corrs.shape[0]), list(corrs.columns))
plt.title('Correlations between target and user\'s features', fontsize=17)
plt.show()
from pandas.plotting import scatter_matrix
select_cols = ['feature_1', 'feature_2', 'feature_3', 'target']
scatter_matrix(train[select_cols], figsize=[10,10])
plt.suptitle('Pair-wise scatter plots for columns in train.csv', fontsize=15)
plt.show()
merchants = pd.read_csv(os.path.join(data_folder, 'merchants.csv'))
# replacing inf values with nan
merchants.replace([-np.inf, np.inf], np.nan, inplace=True)
clean_merchants = merchants.loc[(merchants['numerical_1'] < 0.1) &
                               (merchants['numerical_2'] < 0.1) &
                               (merchants['avg_sales_lag3'] < 5) &
                               (merchants['avg_purchases_lag3'] < 5) &
                               (merchants['avg_sales_lag6'] < 10) &
                               (merchants['avg_purchases_lag6'] < 10) &
                               (merchants['avg_sales_lag12'] < 10) &
                               (merchants['avg_purchases_lag12'] < 10)]
cat_cols = ['active_months_lag6','active_months_lag3','most_recent_sales_range', 'most_recent_purchases_range','category_1','active_months_lag12','category_4', 'category_2']
num_cols = ['numerical_1', 'numerical_2','merchant_group_id','merchant_category_id','avg_sales_lag3', 'avg_purchases_lag3', 'subsector_id', 'avg_sales_lag6', 'avg_purchases_lag6', 'avg_sales_lag12', 'avg_purchases_lag12']

plt.figure(figsize=[15, 15])
plt.suptitle('Merchants table histograms', y=1.02, fontsize=20)
ncols = 4
nrows = int(np.ceil((len(cat_cols) + len(num_cols))/4))
last_ind = 0
for col in sorted(list(clean_merchants.columns)):
    #print('processing column ' + col)
    if col in cat_cols:
        last_ind += 1
        plt.subplot(nrows, ncols, last_ind)
        vc = clean_merchants[col].value_counts()
        x = np.array(vc.index)
        y = vc.values
        inds = np.argsort(x)
        x = x[inds].astype(str)
        y = y[inds]
        plt.bar(x, y, color=(0, 0, 0, 0.7))
        plt.title(col, fontsize=15)
    if col in num_cols:
        last_ind += 1
        plt.subplot(nrows, ncols, last_ind)
        clean_merchants[col].hist(bins = 50, color=(0, 0, 0, 0.7))
        plt.title(col, fontsize=15)
    plt.tight_layout()
# converting category names to numbers, so these columns
# can participate in the correlation coefficient heat map
clean_merchants['most_recent_purchases_range'].replace({'A':4, 'B':3, 'C':2, 'D':1, 'E':0}, inplace=True)
clean_merchants['most_recent_sales_range'].replace({'A':4, 'B':3, 'C':2, 'D':1, 'E':0}, inplace=True)
clean_merchants['category_1'].replace({'N':0, 'Y':1}, inplace=True)
corrs = np.abs(clean_merchants.corr())
ordered_cols = (corrs).sum().sort_values().index
np.fill_diagonal(corrs.values, 0)
plt.figure(figsize=[10,10])
plt.imshow(corrs.loc[ordered_cols, ordered_cols], cmap='plasma', vmin=0, vmax=1)
plt.colorbar(shrink=0.7)
plt.xticks(range(corrs.shape[0]), list(ordered_cols), rotation=90)
plt.yticks(range(corrs.shape[0]), list(ordered_cols))
plt.title('Heat map of coefficients of correlation between merchant\'s features', fontsize=17)
plt.show()
scatter_matrix(clean_merchants[ordered_cols[-6:]], figsize=[15,15])
plt.show()
x = np.array([12, 6, 3]).astype(str)
sales_rates = clean_merchants[['avg_sales_lag3', 'avg_sales_lag6', 'avg_sales_lag12']].mean().values
purchase_rates = clean_merchants[['avg_purchases_lag3', 'avg_purchases_lag6', 'avg_purchases_lag12']].mean().values
plt.bar(x, sales_rates, width=0.3, align='edge', label='average sales', edgecolor=[0.2]*3)
plt.bar(x, purchase_rates, width=-0.3, align='edge', label='average purchases', edgecolor=[0.2]*3)
plt.legend()
plt.title('Avergage sales and number of purchases\nover the last 12, 6, and 3 months', fontsize=17)
plt.show()
scatter_matrix(clean_merchants[ordered_cols[-14:-8]], figsize=[15,15])
plt.tight_layout()
plt.show()
scatter_matrix(merchants[ordered_cols[0:6]], figsize=[10,10])
plt.show()
new_merch = pd.read_csv(os.path.join(data_folder, 'new_merchant_transactions.csv'))
new_merch.info(verbose=True, null_counts=True)
# converting purchase time string into datetime
from datetime import datetime
new_merch['purchase_date'] = new_merch['purchase_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
# drawing histograms for each column
filtered_new_merch = new_merch.loc[new_merch['purchase_amount'] < 1]
cat_cols = ['authorized_flag', 'category_1', 'installments','category_3', 'month_lag','category_2', 'subsector_id']
num_cols = ['purchase_amount', 'purchase_date', 'merchant_category_id', 'subsector_id']

plt.figure(figsize=[15, 10])
plt.suptitle('New merchant transaction info', y=1.02, fontsize=20)
ncols = 4
nrows = int(np.ceil((len(cat_cols) + len(num_cols))/4))
last_ind = 0
for col in sorted(list(filtered_new_merch.columns)):
    #print('processing column ' + col)
    if col in cat_cols:
        last_ind += 1
        plt.subplot(nrows, ncols, last_ind)
        vc = filtered_new_merch[col].value_counts()
        x = np.array(vc.index)
        y = vc.values
        inds = np.argsort(x)
        x = x[inds].astype(str)
        y = y[inds]
        plt.bar(x, y, color=(0, 0, 0, 0.7))
        plt.title(col, fontsize=15)
        plt.xticks(rotation=90)
    if col in num_cols:
        last_ind += 1
        plt.subplot(nrows, ncols, last_ind)
        filtered_new_merch[col].hist(bins = 50, color=(0, 0, 0, 0.7))
        plt.title(col, fontsize=15)
        plt.xticks(rotation=90)
    plt.tight_layout()
# converting category_1 and category_3 values to numeric ones, so we can use then in scatter plots and correlation coefficients
filtered_new_merch['category_3'].replace({'A':0, 'B':1, 'C':2}, inplace=True)
filtered_new_merch['category_1'].replace({'N':0, 'Y':1}, inplace=True)
trns_history = pd.read_csv(os.path.join(data_folder, 'historical_transactions.csv'))
trns_history.info(verbose=True, null_counts=True)
trns_history['purchase_date'] = trns_history['purchase_date'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
filtered_trns_history = trns_history.loc[trns_history['purchase_amount'] < 1]
cat_cols = ['authorized_flag', 'category_1', 'installments','category_3', 'month_lag','category_2', 'subsector_id']
num_cols = ['purchase_amount', 'purchase_date', 'merchant_category_id', 'subsector_id']

plt.figure(figsize=[15, 10])
plt.suptitle('Transaction history info', y=1.02, fontsize=20)
ncols = 4
nrows = int(np.ceil((len(cat_cols) + len(num_cols))/4))
last_ind = 0
for col in sorted(list(filtered_trns_history.columns)):
    #print('processing column ' + col)
    if col in cat_cols:
        last_ind += 1
        plt.subplot(nrows, ncols, last_ind)
        vc = filtered_trns_history[col].value_counts()
        x = np.array(vc.index)
        y = vc.values
        inds = np.argsort(x)
        x = x[inds].astype(str)
        y = y[inds]
        plt.bar(x, y, color=(0, 0, 0, 0.7))
        plt.title(col, fontsize=15)
        plt.xticks(rotation=90)
    if col in num_cols:
        last_ind += 1
        plt.subplot(nrows, ncols, last_ind)
        filtered_trns_history[col].hist(bins = 50, color=(0, 0, 0, 0.7))
        plt.title(col, fontsize=15)
        plt.xticks(rotation=90)
    plt.tight_layout()