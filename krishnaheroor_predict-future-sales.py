import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns
# importing the libraries



train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv', parse_dates = ['date'], infer_datetime_format = True, dayfirst = True)

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

item_categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')



# checking the shapes of these datasets

print("Shape of train:", train.shape)

print("Shape of test:", test.shape)

print("Shape of shops:", shops.shape)

print("Shape of items:", items.shape)

print("Shape of item_categories:", item_categories.shape)
# head of training dataset



train.head()
# head of test data set



test.head()
# sample of shops data set



shops.sample(10)
shops.dtypes
# head of items dataset



items.head()
# checking the head of item_categories dataset



item_categories.head()
# describing the training set



train.describe()
# getting the information about the data



train.info()
# checking if there is any Null data inside the given data



print("No. of Null values in the train set :", train.isnull().sum().sum())

print("No. of Null values in the test set :", test.isnull().sum().sum())

print("No. of Null values in the item set :", items.isnull().sum().sum())

print("No. of Null values in the shops set :", shops.isnull().sum().sum())

print("No. of Null values in the item_categories set :", item_categories.isnull().sum().sum())
# looking at the number of different categories



plt.rcParams['figure.figsize'] = (19, 9)

sns.barplot(items['item_category_id'], items['item_id'], palette = 'colorblind')

plt.title('Count for Different Items Categories', fontsize = 30)

plt.xlabel('Item Categories', fontsize = 15)

plt.ylabel('Items in each Categories', fontsize = 15)

plt.show()
# having a look at the distribution of item sold per day



plt.rcParams['figure.figsize'] = (19, 9)

sns.countplot(train['date_block_num'])

plt.title('Date blocks according to months', fontsize = 30)

plt.xlabel('Different blocks of months', fontsize = 15)

plt.ylabel('No. of Purchases', fontsize = 15)

plt.show()
# having a look at the distribution of item price



plt.rcParams['figure.figsize'] = (13, 7)

sns.distplot(train['item_price'], color = 'red')

plt.title('Distribution of the price of Items', fontsize = 30)

plt.xlabel('Range of price of items', fontsize = 15)

plt.ylabel('Distrbution of prices over items', fontsize = 15)

plt.show()
# checking the no. of unique item present in the stores



x = train['item_id'].nunique()

print("The No. of Unique Items Present in the stores available: ", x)
# checking the no. of unique item present in the stores



x = item_categories['item_category_id'].nunique()

print("The No. of Unique categories for Items Present in the stores available: ", x)
# checking the no. of unique shops given in the dataset



x = train['shop_id'].nunique()

print("No. of Unique Shops are :", x)
# making a word cloud for item categories name



from wordcloud import WordCloud

from wordcloud import STOPWORDS



plt.rcParams['figure.figsize'] = (15, 12)

stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'lightblue',

                      max_words = 200, 

                      stopwords = stopwords,

                     width = 1200,

                     height = 800,

                     random_state = 42).generate(str(item_categories['item_category_name']))





plt.title('Wordcloud for Item Category Names', fontsize = 30)

plt.axis('off')

plt.imshow(wordcloud, interpolation = 'bilinear')
# making a word cloud for item name



from wordcloud import WordCloud

from wordcloud import STOPWORDS



plt.rcParams['figure.figsize'] = (15, 12)

stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'pink',

                      max_words = 200, 

                      stopwords = stopwords,

                     width = 1200,

                     height = 800,

                     random_state = 42).generate(str(items['item_name']))





plt.title('Wordcloud for Item Names', fontsize = 30)

plt.axis('off')

plt.imshow(wordcloud, interpolation = 'bilinear')
# making a word cloud for shop name



from wordcloud import WordCloud

from wordcloud import STOPWORDS



plt.rcParams['figure.figsize'] = (15, 12)

stopwords = set(STOPWORDS)

wordcloud = WordCloud(background_color = 'gray',

                      max_words = 200, 

                      stopwords = stopwords,

                     width = 1200,

                     height = 800,

                     random_state = 42).generate(str(shops['shop_name']))





plt.title('Wordcloud for Shop Names', fontsize = 30)

plt.axis('off')

plt.imshow(wordcloud, interpolation = 'bilinear')
# making a new column day

train['day'] = train['date'].dt.day



# making a new column month

train['month'] = train['date'].dt.month



# making a new column year

train['year'] = train['date'].dt.year



# making a new column week

train['week'] = train['date'].dt.week



# checking the new columns

train.columns
# checking which days are most busisiest for the shops



plt.rcParams['figure.figsize'] = (15, 7)

sns.countplot(train['day'])

plt.title('The most busiest days for the shops', fontsize = 30)

plt.xlabel('Days', fontsize = 15)

plt.ylabel('Frequency', fontsize = 15)



plt.show()
# checking which months are most busisiest for the shops



plt.rcParams['figure.figsize'] = (15, 7)

sns.countplot(train['month'], palette = 'dark')

plt.title('The most busiest months for the shops', fontsize = 30)

plt.xlabel('Months', fontsize = 15)

plt.ylabel('Frequency', fontsize = 15)



plt.show()
# checking which years are most busisiest for the shops



plt.rcParams['figure.figsize'] = (15, 7)

sns.countplot(train['year'], palette = 'colorblind')

plt.title('The most busiest years for the shops', fontsize = 30)

plt.xlabel('Year', fontsize = 15)

plt.ylabel('Frequency', fontsize = 15)



plt.show()
# checking the columns of the train data



train.columns

# feature engineering



train['revenue'] = train['item_price'] * train['item_cnt_day']



sns.distplot(train['revenue'], color = 'blue')

plt.title('Distribution of Revenue', fontsize = 30)

plt.xlabel('Range of Revenue', fontsize = 15)

plt.ylabel('Revenue')

plt.show()
train.dtypes
# plotting a box plot for itemprice and item-cnt-day



plt.rcParams['figure.figsize'] = (15, 7)

sns.violinplot(x = train['day'], y = train['revenue'])

plt.title('Box Plot for Days v/s Revenue', fontsize = 30)

plt.xlabel('Days', fontsize = 15)

plt.ylabel('Revenue', fontsize = 15)

plt.show()
# plotting a box plot for itemprice and item-cnt-day



plt.rcParams['figure.figsize'] = (15, 7)

sns.boxplot(x = train['month'], y = train['revenue'])

plt.title('Box Plot for Days v/s Revenue', fontsize = 30)

plt.xlabel('Months', fontsize = 15)

plt.ylabel('Revenue', fontsize = 15)

plt.show()
# plotting a box plot for itemprice and item-cnt-day



plt.rcParams['figure.figsize'] = (15, 7)

sns.boxplot(x = train['year'], y = train['revenue'])

plt.title('Box Plot for Days v/s Revenue', fontsize = 30)

plt.xlabel('Years', fontsize = 15)

plt.ylabel('Revenue', fontsize = 15)

plt.show()
# converting the data into monthly sales data



# making a dataset with only monthly sales data

data = train.groupby([train['date'].apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).sum().reset_index()



# specifying the important attributes which we want to add to the data

data = data[['date','item_id','shop_id','item_cnt_day']]



# at last we can select the specific attributes from the dataset which are important 

data = data.pivot_table(index=['item_id','shop_id'], columns = 'date', values = 'item_cnt_day', fill_value = 0).reset_index()



# looking at the newly prepared datset

data.shape
# let's merge the monthly sales data prepared to the test data set



test = pd.merge(test, data, on = ['item_id', 'shop_id'], how = 'left')



# filling the empty values found in the dataset

test.fillna(0, inplace = True)



# checking the dataset

test.head()
# now let's create the actual training data



x_train = test.drop(['2015-10', 'item_id', 'shop_id'], axis = 1)

y_train = test['2015-10']



# deleting the first column so that it can predict the future sales data

x_test = test.drop(['2013-01', 'item_id', 'shop_id'], axis = 1)



# checking the shapes of the datasets

print("Shape of x_train :", x_train.shape)

print("Shape of x_test :", x_test.shape)

print("Shape of y_test :", y_train.shape)
# let's check the x_train dataset



x_train.head()
# let's check the x_test data



x_test.head()
# splitting the data into train and valid dataset



from sklearn.model_selection import train_test_split



x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size = 0.2, random_state = 0)



# checking the shapes

print("Shape of x_train :", x_train.shape)

print("Shape of x_valid :", x_valid.shape)

print("Shape of y_train :", y_train.shape)

print("Shape of y_valid :", y_valid.shape)
# MODELING



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

# Get the test set predictions and clip values to the specified range

y_pred_lgb = model_lgb.predict(x_test).clip(0., 20.)



# Create the submission file and submit

preds = pd.DataFrame(y_pred_lgb, columns=['item_cnt_month'])

preds.to_csv('submission.csv',index_label='ID')