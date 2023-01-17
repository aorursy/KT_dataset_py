from datetime import datetime, timedelta,date

import numpy as np

import pandas as pd

import seaborn as sns

from matplotlib import pyplot as plt

%matplotlib inline

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

from sklearn.metrics import classification_report, confusion_matrix
sales_train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

item_categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

sample_submission = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

sales_train.head(10)
test.head(10)
sales_train.info()
sales_train['date'] = pd.to_datetime(sales_train['date'])
sales_train.isna().sum()
test.isna().sum()
sns.set(rc={'figure.figsize':(20, 10)})

sns.set_context("talk", font_scale=1)

sales_month_shop_id = pd.DataFrame(sales_train.groupby(['shop_id']).sum().item_cnt_day).reset_index()

sales_month_shop_id.columns = ['shop_id', 'sum_sales']

sns.barplot(x ='shop_id', y='sum_sales', data=sales_month_shop_id, palette='Paired')

plt.title('Distribution of sales per shop');

del sales_month_shop_id



sales_item_id = pd.DataFrame(sales_train.groupby(['item_id']).sum().item_cnt_day)

plt.xlabel('item id')

plt.ylabel('sales')

plt.plot(sales_item_id);
anom_item = sales_item_id.item_cnt_day.argmax()

print(anom_item)
items[items['item_id'] == 20602]
sns.set_context("talk", font_scale=0.8)

sales_item_cat = sales_train.merge(items, how='left', on='item_id').groupby('item_category_id').item_cnt_day.sum()

sns.barplot(x ='item_category_id', y='item_cnt_day',

            data=sales_item_cat.reset_index(), 

            palette='Paired'

           );

del sales_item_cat
sns.set(style = "whitegrid")

plt.plot(sales_train['item_id'], sales_train['item_price'], '*', color='MediumBlue');
sales_train[sales_train['item_price'] > 250000]
items[items['item_id'] == 6066]
item_categories[item_categories['item_category_id'] == 75]
shops[shops['shop_id'] == 12]
sales_train_sub = sales_train

sales_train_sub['month'] = pd.DatetimeIndex(sales_train_sub['date']).month

sales_train_sub['year'] = pd.DatetimeIndex(sales_train_sub['date']).year

sales_train_sub.head(10)
monthly_sales=sales_train_sub.groupby(["date_block_num","shop_id","item_id"])["item_cnt_day"].agg(item_cnt_day = 'sum')



monthly_sales['date_block_num'] = monthly_sales.index.get_level_values('date_block_num') 

monthly_sales['shop_id'] = monthly_sales.index.get_level_values('shop_id') 

monthly_sales['item_id'] = monthly_sales.index.get_level_values('item_id') 

monthly_sales.reset_index(drop=True, inplace=True)



monthly_sales = monthly_sales.reindex(['date_block_num','shop_id','item_id','item_cnt_day'], axis=1)

monthly_sales.head(10)
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier

from sklearn.model_selection import train_test_split # Import train_test_split function

from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
#split dataset in features and target variable

feature_cols = ['shop_id','date_block_num','item_id']

X = monthly_sales[feature_cols] # Features

y = monthly_sales.item_cnt_day # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1) # 70% training and 30% test
# Create Decision Tree classifer object

clf = DecisionTreeClassifier()



# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)
# Model Accuracy

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Create Decision Tree classifer object

clf = DecisionTreeClassifier(criterion="entropy", max_depth=3)

#clf = tree.DecisionTreeClassifier(criterion='gini')



# Train Decision Tree Classifer

clf = clf.fit(X_train,y_train)



#Predict the response for test dataset

y_pred = clf.predict(X_test)



# Model Accuracy, how often is the classifier correct?

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Predictor columns

X2 = monthly_sales[feature_cols]



# Target variable

Y2 = monthly_sales.item_cnt_day



# Fitting Simple Linear Regression model to the data set

from sklearn.tree import DecisionTreeRegressor

model_DTR = DecisionTreeRegressor(random_state = 0)

model_DTR.fit(X2, Y2)



X2_test_DTR = test[['shop_id','item_id']]

X2_test_DTR.insert(loc=1, column='date_block_num', value='34')





predicted_raw_DTR = pd.DataFrame(model_DTR.predict(X2_test_DTR))

predicted_raw_DTR = X2_test_DTR.join(predicted_raw_DTR)



predicted_raw_DTR.columns  = ['shop_id', 'date_block_num','item_id', 'item_cnt']

predicted_DTR = predicted_raw_DTR.reindex(['shop_id','date_block_num','item_id','item_cnt'], axis=1)

predicted_DTR.head(20)