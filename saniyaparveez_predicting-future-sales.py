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
train_df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv', parse_dates = ['date'], infer_datetime_format = True, dayfirst = True)

test_df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')

shops_df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

items_df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

item_categories_df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
train_df.head()
print("Shape of train:", train_df.shape)
print("Shape of test:", test_df.shape)
print("Shape of shops:", shops_df.shape)
print("Shape of items:", items_df.shape)
print("Shape of item_categories:", item_categories_df.shape)
test_df.head()
shops_df.head()
shops_df.dtypes
items_df.head()
train_df.describe()
train_df.info()
print("No. of Null values in the train set :", train_df.isnull().sum().sum())
print("No. of Null values in the test set :", test_df.isnull().sum().sum())
print("No. of Null values in the item set :", items_df.isnull().sum().sum())
print("No. of Null values in the shops set :", shops_df.isnull().sum().sum())
print("No. of Null values in the item_categories set :", item_categories_df.isnull().sum().sum())
import matplotlib.pyplot as plt
import seaborn as sns

plt.rcParams['figure.figsize'] = (19, 9)
sns.barplot(items_df['item_category_id'], items_df['item_id'], palette = 'colorblind')
plt.title('Count for Different Items Categories', fontsize = 30)
plt.xlabel('Item Categories', fontsize = 15)
plt.ylabel('Items in each Categories', fontsize = 15)
plt.show()
plt.rcParams['figure.figsize'] = (19, 9)
sns.countplot(train_df['date_block_num'])
plt.title('Date blocks according to months', fontsize = 30)
plt.xlabel('Different blocks of months', fontsize = 15)
plt.ylabel('No. of Purchases', fontsize = 15)
plt.show()
plt.rcParams['figure.figsize'] = (13, 7)
sns.distplot(train_df['item_price'], color = 'red')
plt.title('Distribution of the price of Items', fontsize = 30)
plt.xlabel('Range of price of items', fontsize = 15)
plt.ylabel('Distrbution of prices over items', fontsize = 15)
plt.show()
x = train_df['item_id'].nunique()
print("The No. of Unique Items Present in the stores available: ", x)
x = item_categories_df['item_category_id'].nunique()
print("The No. of Unique categories for Items Present in the stores available: ", x)
x = train_df['shop_id'].nunique()
print("No. of Unique Shops are :", x)
from wordcloud import WordCloud
from wordcloud import STOPWORDS

plt.rcParams['figure.figsize'] = (15, 12)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'lightblue',
                      max_words = 200, 
                      stopwords = stopwords,
                     width = 1200,
                     height = 800,
                     random_state = 42).generate(str(item_categories_df['item_category_name']))


plt.title('Wordcloud for Item Category Names', fontsize = 30)
plt.axis('off')
plt.imshow(wordcloud, interpolation = 'bilinear')
from wordcloud import WordCloud
from wordcloud import STOPWORDS

plt.rcParams['figure.figsize'] = (15, 12)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'pink',
                      max_words = 200, 
                      stopwords = stopwords,
                     width = 1200,
                     height = 800,
                     random_state = 42).generate(str(items_df['item_name']))


plt.title('Wordcloud for Item Names', fontsize = 30)
plt.axis('off')
plt.imshow(wordcloud, interpolation = 'bilinear')
from wordcloud import WordCloud
from wordcloud import STOPWORDS

plt.rcParams['figure.figsize'] = (15, 12)
stopwords = set(STOPWORDS)
wordcloud = WordCloud(background_color = 'gray',
                      max_words = 200, 
                      stopwords = stopwords,
                     width = 1200,
                     height = 800,
                     random_state = 42).generate(str(shops_df['shop_name']))


plt.title('Wordcloud for Shop Names', fontsize = 30)
plt.axis('off')
plt.imshow(wordcloud, interpolation = 'bilinear')
# making a new column day
train_df['day'] = train_df['date'].dt.day

# making a new column month
train_df['month'] = train_df['date'].dt.month

# making a new column year
train_df['year'] = train_df['date'].dt.year

# making a new column week
train_df['week'] = train_df['date'].dt.week

# checking the new columns
train_df.columns
plt.rcParams['figure.figsize'] = (15, 7)
sns.countplot(train_df['day'])
plt.title('The most busiest days for the shops', fontsize = 30)
plt.xlabel('Days', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)

plt.show()
plt.rcParams['figure.figsize'] = (15, 7)
sns.countplot(train_df['month'], palette = 'dark')
plt.title('The most busiest months for the shops', fontsize = 30)
plt.xlabel('Months', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)

plt.show()
plt.rcParams['figure.figsize'] = (15, 7)
sns.countplot(train_df['year'], palette = 'colorblind')
plt.title('The most busiest years for the shops', fontsize = 30)
plt.xlabel('Year', fontsize = 15)
plt.ylabel('Frequency', fontsize = 15)

plt.show()
train_df.columns
# feature engineering

train_df['revenue'] = train_df['item_price'] * train_df['item_cnt_day']

sns.distplot(train_df['revenue'], color = 'blue')
plt.title('Distribution of Revenue', fontsize = 30)
plt.xlabel('Range of Revenue', fontsize = 15)
plt.ylabel
train.dtypes
plt.rcParams['figure.figsize'] = (15, 7)
sns.violinplot(x = train_df['day'], y = train_df['revenue'])
plt.title('Box Plot for Days v/s Revenue', fontsize = 30)
plt.xlabel('Days', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)
plt.show()
plt.rcParams['figure.figsize'] = (15, 7)
sns.boxplot(x = train_df['month'], y = train_df['revenue'])
plt.title('Box Plot for Days v/s Revenue', fontsize = 30)
plt.xlabel('Months', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)
plt.show()
plt.rcParams['figure.figsize'] = (15, 7)
sns.boxplot(x = train_df['year'], y = train_df['revenue'])
plt.title('Box Plot for Days v/s Revenue', fontsize = 30)
plt.xlabel('Years', fontsize = 15)
plt.ylabel('Revenue', fontsize = 15)
plt.show()
# converting the data into monthly sales data

# making a dataset with only monthly sales data
data = train_df.groupby([train_df['date'].apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).sum().reset_index()

# specifying the important attributes which we want to add to the data
data = data[['date','item_id','shop_id','item_cnt_day']]

# at last we can select the specific attributes from the dataset which are important 
data = data.pivot_table(index=['item_id','shop_id'], columns = 'date', values = 'item_cnt_day', fill_value = 0).reset_index()

# looking at the newly prepared datset
data.shape
test_df = pd.merge(test_df, data, on = ['item_id', 'shop_id'], how = 'left')

# filling the empty values found in the dataset
test_df.fillna(0, inplace = True)

# checking the dataset
test_df.head()
x_train = test_df.drop(['2015-10', 'item_id', 'shop_id'], axis = 1)
y_train = test_df['2015-10']

# deleting the first column so that it can predict the future sales data
x_test = test_df.drop(['2013-01', 'item_id', 'shop_id'], axis = 1)

# checking the shapes of the datasets
print("Shape of x_train :", x_train.shape)
print("Shape of x_test :", x_test.shape)
print("Shape of y_test :", y_train.shape)
x_train.head()
x_test.head()
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)

# checking the shapes
print("Shape of x_train :", x_train.shape)
print("Shape of x_valid :", x_valid.shape)
print("Shape of y_train :", y_train.shape)
print("Shape of y_valid :", y_valid.shape)
from lightgbm import LGBMRegressor

model_lgb = LGBMRegressor( n_estimators=200,
                           learning_rate=0.03,
                           num_leaves=32,
                           colsample_bytree=0.9497036,
                           subsample=0.8715623,
                           max_depth=8,
                           reg_alpha=0.04,
                           reg_lambda=0.073,
                           min_split_gain=0.0222415,
                           min_child_weight=40)
model_lgb.fit(x_train, y_train)

y_pred_lgb = model_lgb.predict(x_test)
y_pred_lgb = model_lgb.predict(x_test).clip(0., 20.)


preds = pd.DataFrame(y_pred_lgb, columns=['item_cnt_month'])
preds
preds.to_csv('submission.csv',index_label='ID')