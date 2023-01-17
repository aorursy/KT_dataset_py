# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

main_csv = '/kaggle/input/summer-products-and-sales-in-ecommerce-wish/summer-products-with-rating-and-performance_2020-08.csv'
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv(main_csv)

from collections import Counter

Counter(df['badge_fast_shipping'])
df.describe()
%matplotlib inline
fig, axs = plt.subplots(3, 1, figsize=(10,10))

fig.suptitle("KDE and histogram of the several variables", fontsize=20)
plt.xlabel('Bins of values', color='white', fontsize=14)
plt.ylabel('Density', color='white', fontsize=14)

sns.set_style('whitegrid')

hist_columns = ['price', 'retail_price', 'shipping_option_price']

N = 100

for i in range(len(hist_columns)):
    name = hist_columns[i]
    arr = df[name]
    
    ax = axs[i]
    
    ax.set_title(name)
    sns.distplot(arr, kde=True, color='g', ax=ax).set(xlim=(-3,25))
    plt.plot()
    
    ax.set_xlim([0,30])
    
fig.tight_layout(pad = 3.0)
sns.distplot(df['units_sold'], kde=True, color='g')
plt.show()
g = sns.PairGrid(data=df, vars = ['price', 'retail_price'], hue='units_sold', height=4)
g.map(plt.scatter)
g.add_legend()
plt.show()
sns.jointplot(x='price', y='retail_price', data=df, kind='kde', xlim=(0,20), ylim=(-5,35), height=10)
plt.show()
df['price_drop'] = df['retail_price'] - df['price']
plt.scatter(df['price_drop'], df['units_sold'], alpha=.3)
plt.xlabel('price drop')
plt.ylabel('units sold')
##### ONLY NUMERIC VALUES:
numeric_cols = df.describe().columns
df_numeric = df[numeric_cols]
df_numeric
n = 100
df_sorted_units = df_numeric.sort_values(by='units_sold', ascending=False)
top_n = df_sorted_units.head(n)
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_new = scaler.fit_transform(df_sorted_units)
numeric_cols = df.describe().columns
df_sorted_units = pd.DataFrame(df_new, columns=numeric_cols)
n_quantiles = [top_n]
for i in range(1, len(df)//n):
    n_quan = df_sorted_units.iloc[ (len(df)- n*(i+1)) : -n*i]
    n_quantiles.append(n_quan)
qms, qss = [],[]
qms_r, qss_r = [], []
for quan in n_quantiles:
    quan_mean = quan.describe().loc['mean'][['price', 'retail_price']]
    quan_std = quan.describe().loc['std'][['price', 'retail_price']]
    
    qms.append(quan_mean['price'])
    qss.append(quan_std['price'])
    
    qms_r.append(quan_mean['retail_price'])
    qss_r.append(quan_mean['retail_price'])
print('Mean of mean discounted price across 15 quantiles:', np.mean(qms), '\n Standard deviation of the mean discounted price:', np.std(qms))
print('\n')
print('Mean of mean retailed price across 15 quantiles:', np.mean(qms_r), '\n Standard deviation of the mean retailed price:', np.std(qms_r))
numeric_cols = df.describe().columns
import warnings
warnings.filterwarnings('ignore')

 #take only numeric cols

def feature_search(df, search_feature, n, with_qr=True):
    '''
    Search for features in the given DataFrame that differentiates quantiles based on some value
    '''
    features = df.describe().columns
    print(features)
    sorted_arr = df.sort_values(by=search_feature)[features].to_numpy()
    # Turn this into numpy
    # Consider making them into arrays, then make column as features and the row as the quantile means
    # We're interested in the stdev of the quartile means
    # 1) Find feature mean for each quartile
    
    fq_means = np.array([None]*len(features)) #placeholder for feature means on each quartile
    
    while sorted_arr.shape[0] > n:
        quan = sorted_arr[:n, :] #take the first n
        sorted_arr = np.delete(sorted_arr, slice(n), 0) #remove the bottom n from the sorted arr (free up some mems)
        quan_range = np.max(quan, axis=0) - np.min(quan, axis=0)
        if with_qr:
            quan_feature_means = np.mean(quan, axis=0)*quan_range #yields 1D array of features' means in this quantile MULTIPLIED by the range of the quantile
        else:
            quan_feature_means = np.mean(quan, axis=0)
        fq_means = np.vstack((fq_means, quan_feature_means)) # stack it on previous
        
    fq_means = np.delete(fq_means, 0, 0).astype('float32') # delete the first one, and convert to float32 for numpy ops
    
    # 2) Get np.nanstd(, 0) (by column)
    #features_stdevs = np.std(fq_means, axis=0)
    features_std = np.nanstd(fq_means, axis=0)
    features_mean = np.nanmean(fq_means, axis=0)
    std_to_mean = features_std / features_mean
    return std_to_mean
feature_search(df_sorted_units, 'units_sold', 10, with_qr=False)
### WITHOUT QUAN RANGE
std_mean_ratio = feature_search(df_sorted_units, 'units_sold', 10, with_qr=False)
print(std_mean_ratio[24])
fv_units_sold = {numeric_cols[i]: std_mean_ratio[i] for i in range(len(numeric_cols))}
print('Below is the measure of sensitivity (stdev to mean) for numeric features for quantiles created for the units_sold column: \n', fv_units_sold)
#### WITH QUAN RANGE
std_mean_ratio_quan = feature_search(df_sorted_units, 'units_sold', 10, with_qr=True)
fv_units_sold_quan = {features[i]: std_mean_ratio_quan[i] for i in range(len(features))}
print('Below is the measure of sensitivity (stdev to mean) for numeric features for quantiles created for the units_sold column: \n', fv_units_sold_quan)
df_sorted_units
top_5 = sorted(fv_units_sold_quan, key=fv_units_sold_quan.get, reverse=True)[:10]

for feature in top_5:
    print(feature, fv_units_sold_quan[feature])
# Ratio between the twos (to see how much the range mattered for each):

def range_matter_func(fv_units_sold_quan, fv_units_sold):
    range_matter = {}

    for name in fv_units_sold:
        range_matter[name] = fv_units_sold_quan[name] / fv_units_sold[name]

    del range_matter['has_urgency_banner']

    sort_by_rm = sorted(range_matter, key=range_matter.get, reverse=True)
    top10_by_rm = sort_by_rm[:10]
    top10_by_rm

    for feature in top10_by_rm:
        print(feature, range_matter[feature], '{:.2f}'.format(np.std(df_sorted_units[feature])))
range_matter_func(fv_units_sold_quan, fv_units_sold)
for feature in numeric_cols:
    plt.scatter(df_sorted_units['units_sold'], df_sorted_units[feature], alpha=0.03)
    plt.ylabel(feature)
    plt.xlabel('units_sold')
    plt.show()