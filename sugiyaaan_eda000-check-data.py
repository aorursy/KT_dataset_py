# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling as pdp
pd.set_option('display.max_rows',200)
# Dataset
INPUT_DIR = r'/kaggle/input/house-prices-advanced-regression-techniques'
train_df = pd.read_csv(INPUT_DIR + '/' + 'train.csv')
test_df = pd.read_csv(INPUT_DIR + '/' + 'test.csv')
ss = pd.read_csv(INPUT_DIR + '/'+ 'sample_submission.csv')
print('train columns: ',train_df.columns)
print('test columns: ',test_df.columns)

print(train_df.info())
train_df.head()
print(test_df.info())
test_df.head()
ss.head()
# checking missing values in traing data
# credits https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard/notebook

train_data_na = (train_df.isnull().sum() / len(train_df)) * 100
train_data_na = train_data_na.drop(train_data_na[train_data_na == 0].index).sort_values(ascending=False)[:30]
train_missing_data = pd.DataFrame({'Missing Ratio' :train_data_na})
train_missing_data.head(20)
# checking missing values in test data
# credits https://www.kaggle.com/serigne/stacked-regressions-top-4-on-leaderboard/notebook

test_data_na = (test_df.isnull().sum() / len(test_df)) * 100
test_data_na = test_data_na.drop(test_data_na[test_data_na == 0].index).sort_values(ascending=False)[:30]
test_missing_data = pd.DataFrame({'Missing Ratio' :test_data_na})
test_missing_data.head(20)
# check target variable
p = sns.distplot(train_df['SalePrice'])
p.figure.set_size_inches((10,7))
p.grid()
p.set_title('Distribution of SalesPrice', size = 15, weight='bold')
plt.show()
p = sns.distplot(np.log(train_df['SalePrice']))
p.figure.set_size_inches((10,7))
p.grid()
p.set_title('Distribution of SalesPrice on Log Scale', size = 15, weight='bold')
plt.show()
corr = train_df.corr()
print(corr.shape)
corr['SalePrice'].sort_values(ascending =False)
plt.figure(figsize=(26, 10))
sns.heatmap(corr, annot=True, linewidths=2,cmap="YlGnBu")
plt.show()
plt.close()
numeric_data_list=list(corr['SalePrice'].sort_values(ascending =False).index)
# 
for f in numeric_data_list:
    fig, axes = plt.subplots(1,2, figsize=(13,6))
    ax1= sns.scatterplot(x=f, y='SalePrice', data= train_df,ax=axes[0])
    ax2= sns.boxplot(x=f, data= train_df,ax=axes[1])
    plt.show()
    plt.close()
categ = pd.concat([train_df[train_df.columns[~train_df.columns.isin(numeric_data_list)]], train_df["SalePrice"]],axis=1)
categ.head()
plt.figure(figsize=(10, 5))
for f in categ.columns:
    chart = sns.catplot(x=f, y="SalePrice", kind="swarm",
            data=categ);
    chart.set_xticklabels(rotation=45)
    plt.show()
    plt.close()

