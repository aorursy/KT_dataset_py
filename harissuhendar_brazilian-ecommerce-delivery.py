# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import datetime

plt.style.use('ggplot')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
order = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_orders_dataset.csv')

category = pd.read_csv('/kaggle/input/brazilian-ecommerce/product_category_name_translation.csv')

customer = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_customers_dataset.csv')

orderitems = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_order_items_dataset.csv')

orderpayment = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_order_payments_dataset.csv')

orderreview = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_order_reviews_dataset.csv')

product = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_products_dataset.csv')

sellers = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_sellers_dataset.csv')

geolocation = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_geolocation_dataset.csv')
print('Customer : ', len(customer))

print('Order : ', len(order))

print('Order Items : ', len(orderitems))

print('Order Payment : ', len(orderpayment))

print('Product : ', len(product))

print('Sellers : ', len(sellers))

print('Order Reviews : ', len(orderreview))

print('Geolocation : ', len(geolocation))

print('Category : ', len(category))
delivered_order = order[order['order_status'] == 'delivered']

delivered_order = delivered_order.fillna(0)

delivered_order = delivered_order[delivered_order['order_delivered_customer_date'] != 0]

delivered_order = delivered_order[delivered_order['order_approved_at'] != 0]

time = (pd.to_datetime(delivered_order['order_delivered_customer_date']) - pd.to_datetime(delivered_order['order_approved_at'])).dt.days

delivered_order['delivery_time (days)'] = time

delivered_order = delivered_order[delivered_order['delivery_time (days)'] > 0]

delivered_order = delivered_order.sort_values('order_approved_at')

delivered_order = delivered_order.set_index(pd.to_datetime(delivered_order['order_approved_at']).dt.date, drop=True)

delivery_days_average = delivered_order.groupby(delivered_order.index).mean()

global_average = float(delivery_days_average.mean())



plt.figure(figsize = (12,8))

plt.plot(delivery_days_average.index, delivery_days_average, label = 'daily_delivery_time')

plt.axhline(y=global_average, linestyle='--', color='blue', label = 'average : ' + str(global_average)[:5] + ' days')

plt.xlabel('Date', fontsize=16)

plt.ylabel('average delivery time (days)', fontsize=16)

plt.title('Average delivery time over time since ' + str(delivery_days_average.index[0]) + ' to ' + str(delivery_days_average.index[-1]) , fontsize=16)

plt.legend(loc='best', fontsize=16)

plt.show()
items_seller = pd.merge(orderitems, sellers, on='seller_id')

order_items_seller = pd.merge(items_seller,delivered_order, on='order_id',)

unique_seller = order_items_seller['seller_id'].unique()



seller_mean = {}

for seller_id in unique_seller :

    seller = order_items_seller[order_items_seller['seller_id'] == seller_id]

    seller = seller.set_index(pd.to_datetime(seller['order_approved_at']).dt.date, drop=True)

    seller_days_average = seller.groupby(seller.index).mean()

    global_average = seller_days_average['delivery_time (days)'].mean()

    seller_mean[seller_id] = global_average
delivery_time = pd.DataFrame(seller_mean.items(), columns=['seller_id', 'delivery_time'])

delivery_time.sort_values('delivery_time')[:20]
# Rerata kecepatan pengiriman produk masing-masing seller setiap hari dihitung saat barang dikirim

delivery_time['delivery_time'].hist(bins=1000, figsize=(15,8))

plt.axis([0,35,0,110])

plt.xlabel('average delivery time (days)', fontsize=16)

plt.ylabel('number of seller', fontsize=16)

plt.title('histogram of average delivery time from all seller', fontsize=18)

plt.show()
# Fungsi untuk menghitung jarak dua buah titik berdasarkan perbedaan latitude dan longitude

def distance_km(lat1,lon1,lat2,lon2):

    R = 6373.0  # approximate radius of earth in km

    lat1 = np.radians(lat1)

    lon1 = np.radians(lon1)

    lat2 = np.radians(lat2)

    lon2 = np.radians(lon2)

    dlon = lon2 - lon1

    dlat = lat2 - lat1

    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2

    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c

    return distance



# distance_km(52.2296756,21.0122287,52.406374,16.9251681)
ord_itm_sel_cus = pd.merge(order_items_seller, customer, on='customer_id')

seller_zip = geolocation.iloc[:,:3]

seller_zip.columns = ['seller_zip_code_prefix', 'seller_lat', 'seller_lng']

seller_zip = seller_zip.groupby('seller_zip_code_prefix', ).first()



customer_zip = geolocation.iloc[:,:3]

customer_zip.columns = ['customer_zip_code_prefix', 'customer_lat', 'customer_lng']

customer_zip = customer_zip.groupby('customer_zip_code_prefix', ).first()



ord_itm_sel_cus_locsel = pd.merge(ord_itm_sel_cus, seller_zip, on='seller_zip_code_prefix')

ord_itm_sel_cus_locselcus = pd.merge(ord_itm_sel_cus_locsel, customer_zip, on='customer_zip_code_prefix')

ord_itm_sel_cus_locselcus['distance (km)'] = distance_km(ord_itm_sel_cus_locselcus['seller_lat'],ord_itm_sel_cus_locselcus['seller_lng'],ord_itm_sel_cus_locselcus['customer_lat'],ord_itm_sel_cus_locselcus['customer_lng'])

ord_itm_sel_cus_locselcus['delivery_time/distance'] = ord_itm_sel_cus_locselcus['delivery_time (days)']/ord_itm_sel_cus_locselcus['distance (km)']

ord_itm_sel_cus_locselcus = ord_itm_sel_cus_locselcus[ord_itm_sel_cus_locselcus['distance (km)'] > 0]

ord_itm_sel_cus_locselcus = ord_itm_sel_cus_locselcus.sort_values('delivery_time/distance', ascending=True).reset_index()

group_category = pd.merge(ord_itm_sel_cus_locselcus, product, on='product_id')

group_category_trans = pd.merge(group_category, category, on='product_category_name')
fastest_delivery_by_product_id = group_category_trans.groupby('product_id').agg({'product_category_name': 'first',

                                                                                 'product_category_name_english': 'first',

                                                                                  'delivery_time/distance':'mean',

                                                                                   'order_item_id': 'sum',

                                                                                })



fastest_delivery_by_product_id = fastest_delivery_by_product_id.sort_values('delivery_time/distance', ascending=True)

fastest_delivery_by_product_id[:10]
fastest_delivery_by_product_id = group_category_trans.groupby('product_id').agg({'product_category_name': 'first',

                                                                                 'product_category_name_english': 'first',

                                                                                  'delivery_time/distance':'mean',

                                                                                   'order_item_id': 'sum',

                                                                                })

fastest_delivery_by_product_id = fastest_delivery_by_product_id[fastest_delivery_by_product_id['order_item_id'] > 100]

fastest_delivery_by_product_id = fastest_delivery_by_product_id.sort_values('delivery_time/distance', ascending=True)

fastest_delivery_by_product_id[:10]
fastest_delivery_by_product_id = group_category_trans.groupby('product_category_name').agg({'product_category_name_english':'first',

                                                                                           'delivery_time/distance':'mean',

                                                                                           'order_item_id':'sum'})

fastest_delivery_by_product_id = fastest_delivery_by_product_id.sort_values('delivery_time/distance')

fastest_delivery_by_product_id[:10]
fastest_delivery_by_product_id = group_category_trans.groupby('product_category_name').agg({'product_category_name_english':'first',

                                                                                           'delivery_time/distance':'mean',

                                                                                           'order_item_id':'sum'})



fastest_delivery_by_product_id = fastest_delivery_by_product_id[fastest_delivery_by_product_id['order_item_id'] > 100]

fastest_delivery_by_product_id = fastest_delivery_by_product_id.sort_values('delivery_time/distance', ascending=True)

fastest_delivery_by_product_id[:10]
fastest_delivery_by_product_id = group_category_trans.groupby('seller_id').agg({'delivery_time/distance':'mean',

                                                                                'order_item_id':'sum'})

fastest_delivery_by_product_id = fastest_delivery_by_product_id.sort_values('delivery_time/distance', ascending=True)

fastest_delivery_by_product_id[:10]
fastest_delivery_by_product_id = group_category_trans.groupby('seller_id').agg({'delivery_time/distance':'mean',

                                                                                'order_item_id':'sum'})

fastest_delivery_by_product_id = fastest_delivery_by_product_id[fastest_delivery_by_product_id['order_item_id'] > 100]

fastest_delivery_by_product_id = fastest_delivery_by_product_id.sort_values('delivery_time/distance', ascending=True)

fastest_delivery_by_product_id[:10]
group_category_trans['volume'] = group_category_trans['product_length_cm'] * group_category_trans['product_height_cm'] * group_category_trans['product_width_cm']

group_category_trans['weight/volume'] = group_category_trans['product_weight_g']/group_category_trans['volume']

group_category_trans['price/distance'] = group_category_trans['price']/group_category_trans['distance (km)']
find_corr = group_category_trans[['price', 'freight_value', 'seller_zip_code_prefix', 'delivery_time (days)',

       'customer_zip_code_prefix', 'seller_lat', 'seller_lng', 'customer_lat',

       'customer_lng', 'distance (km)','product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm',

       'volume']]

corr_matrix = find_corr.corr()

corr_matrix['delivery_time (days)'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix



find_corr = group_category_trans[['freight_value', 'delivery_time (days)',

       'customer_zip_code_prefix', 'distance (km)','product_weight_g']]



scatter_matrix(find_corr, figsize=(20,20))

plt.show()
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score, train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline



num_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy = 'median')),

    ('std_scaler', StandardScaler())

])



X = group_category_trans[['freight_value', 'customer_zip_code_prefix', 'distance (km)','product_weight_g']]

y = group_category_trans['delivery_time (days)']



X_data = num_pipeline.fit_transform(X)



X_train, X_test, y_train, y_test = train_test_split(X_data, y.values, test_size=0.2, random_state = 14)

lin_reg = LinearRegression()

lin_reg.fit(X_train,y_train)

y_model = lin_reg.predict(X_test)

print("Predictions : ", y_model)

print("Actual : ", y_test)
y_predictions = lin_reg.predict(X_test)

lin_mse = mean_squared_error(y_test, y_predictions)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
def display_scores(scores):

    print("Score : ", scores)

    print("Mean : ", scores.mean())

    print("Standar deviasion : ", scores.std())



lin_scores = cross_val_score(lin_reg, X_data, y.values, scoring='neg_mean_squared_error', cv=10)

lin_rmse = np.sqrt(-lin_scores)

display_scores(lin_rmse)
lin_reg.coef_
from sklearn.tree import DecisionTreeRegressor



tree_reg = DecisionTreeRegressor(max_depth=100)

tree_reg.fit(X_train, y_train)

y_predict = tree_reg.predict(X_test)

print("Prediction : ", y_predict)

print("Actual : ", y_test)
y_predictions = tree_reg.predict(X_test)

lin_mse = mean_squared_error(y_test, y_predictions)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
tree_scores = cross_val_score(tree_reg, X_data, y.values, scoring='neg_mean_squared_error', cv=10)

lin_rmse = np.sqrt(-tree_scores)

display_scores(lin_rmse)
from sklearn.ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor(n_estimators = 10)

forest_reg.fit(X_train, y_train)

y_predict = forest_reg.predict(X_test)

print('Predictions : ', y_predict)

print('Actual : ', y_test)
y_predictions = forest_reg.predict(X_test)

forest_mse = mean_squared_error(y_test, y_predictions)

forest_rmse = np.sqrt(forest_mse)

forest_rmse
forest_scores = cross_val_score(forest_reg, X_data, y.values, scoring='neg_mean_squared_error', cv=5)

forest_rmse = np.sqrt(-forest_scores)

display_scores(forest_rmse)
from sklearn.linear_model import SGDRegressor



sgd_reg = SGDRegressor(max_iter=1000, penalty = 'l2', eta0=0.05)

sgd_reg.fit(X_train, y_train)

y_predict = sgd_reg.predict(X_test)

print("Prediction : ", y_predict)

print("Actual : ", y_test)
y_predictions = sgd_reg.predict(X_test)

sgd_mse = mean_squared_error(y_test, y_predictions)

sgd_rmse = np.sqrt(sgd_mse)

sgd_rmse
sgd_scores = cross_val_score(sgd_reg, X_data, y.values, scoring='neg_mean_squared_error', cv=10)

sgd_rmse = np.sqrt(-sgd_scores)

display_scores(sgd_rmse)
sgd_reg.coef_
from sklearn.model_selection import GridSearchCV



param_grid = [{'penalty': [None, 'l2', 'l1'], 'eta0': [0.01, 0.05, 0.1, 0.15]}]



sgd_reg = SGDRegressor()

grid_search = GridSearchCV(sgd_reg, param_grid, cv=5, scoring='neg_mean_squared_error')

grid_search.fit(X_data, y.values)
# Hyperparameter terbaik dari pencarian grid cv

grid_search.best_params_
final_model = grid_search.best_estimator_

y_predict = final_model.predict(X_test)

final_mse = mean_squared_error(y_test, y_predict)

final_rmse = np.sqrt(final_mse)

final_rmse
final_scores = cross_val_score(final_model, X_data, y.values, scoring = 'neg_mean_squared_error', cv=5)

final_rmse = np.sqrt(-final_scores)

display_scores(final_rmse)
final_model.coef_
group_cat_review = pd.merge(group_category_trans, orderreview, on='order_id')

class_data = group_cat_review[['price', 'freight_value', 'seller_zip_code_prefix','delivery_time (days)',

                              'customer_zip_code_prefix','seller_lat', 'seller_lng', 'customer_lat',

                               'customer_lng', 'distance (km)', 'delivery_time/distance','product_weight_g',

                               'product_length_cm', 'product_height_cm', 'product_width_cm', 'volume', 'review_score']]
corr_matrix = class_data.corr()

corr_matrix['review_score'].sort_values(ascending=False)
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score



X = class_data = group_cat_review[['freight_value','delivery_time (days)',

                                'distance (km)']]



y = class_data = group_cat_review['review_score']



scaler = StandardScaler()

scaler.fit(X)

X_data = scaler.transform(X)



X_train, X_test, y_train, y_test = train_test_split(X_data, y.values, test_size=0.2)



log_reg = LogisticRegression(multi_class='auto', solver='lbfgs')

log_reg.fit(X_train, y_train)
y_model = log_reg.predict(X_test)

accuracy_score(y_test, y_model )
from sklearn.linear_model import SGDClassifier



sgd_clf = SGDClassifier(random_state=42)

sgd_clf.fit(X_train, y_train)
y_model = sgd_clf.predict(X_test)

accuracy_score(y_test, y_model)
# from sklearn.svm import SVC



# svm_clf = SVC()

# svm_clf.fit(X_train, y_train)
# y_model = svm_clf.predict(X_test)

# accuracy_score(y_test, y_model)