# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import datetime

import math



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

directory = []

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        directory.append(os.path.join(dirname, filename))

directory

# Any results you write to the current directory are saved as output.
Sellers = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_sellers_dataset.csv')

Customers = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_customers_dataset.csv')

Order = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_orders_dataset.csv')

Product  = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_products_dataset.csv')

Order_Payment = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_order_payments_dataset.csv')

Geolocation = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_geolocation_dataset.csv')

Order_Reviews = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_order_reviews_dataset.csv')

Order_Items = pd.read_csv('/kaggle/input/brazilian-ecommerce/olist_order_items_dataset.csv')

Category = pd.read_csv('/kaggle/input/brazilian-ecommerce/product_category_name_translation.csv')
top_product = {}

for product in Order_Items['product_id']:

    if product in top_product:

        top_product[product] += 1

    else :

        top_product[product] = 1



top_product = sorted(top_product.items(), key = lambda x: x[1], reverse=True)

top_product = pd.DataFrame(top_product, columns = ['product_id', 'total'])



top_product_category = []

for i in range(len(top_product['product_id'])):

    val = (Product.product_category_name[Product.product_id ==top_product['product_id'][i]]).to_string(index=False)

    top_product_category.append(val)

    

top_product['product_category_name'] = top_product_category



x_key = np.arange(len(top_product[:20]))

plt.figure(figsize=(20,5))

plt.bar(x_key, top_product.total[:20], color='green')

plt.xticks(x_key, top_product.product_id[:20], rotation= -80)

plt.ylabel('Number of purchased product', fontsize=12)

plt.xlabel('product_id', fontsize=14)

plt.title('Top of Purchased Product', fontsize=14)

plt.show()
top_product_category = top_product.groupby('product_category_name').sum().sort_values('total', ascending=False)

x_key = np.arange(len(top_product_category[:20]))

plt.figure(figsize=(15,5))

plt.bar(x_key, top_product_category.total[:20], color='red')

plt.xticks(x_key, top_product_category.index[:20], rotation= -80)

plt.ylabel('Number of purchased product', fontsize=12)

plt.xlabel('product_id', fontsize=14)

plt.title('Top of Purchased Product by Category', fontsize=14)

plt.show()
# topseller_purchased = {}

# for seller in Order_Items['seller_id']:

#     if seller in topseller_purchased:

#         topseller_purchased[seller] += 1

#     else :

#         topseller_purchased[seller] = 1



# topseller_purchased = sorted(topseller_purchased.items(), key = lambda x: x[1], reverse=True)

# topseller_purchased = pd.DataFrame(topseller_purchased, columns=['seller_id', 'total_purchased'])

# total_purchased = topseller_purchased['total_purchased'].sum()

# topseller_purchased['tot_fract_purchased'] = (topseller_purchased['total_purchased']/total_purchased)*100



# x_key = np.arange(len(topseller_purchased[:20]))

# plt.figure(figsize=(15,5))

# plt.bar(x_key, topseller_purchased.total_purchased[:20], color='blue')

# plt.xticks(x_key, topseller_purchased.seller_id[:20], rotation= -80)

# plt.ylabel('Number of purchased product', fontsize=12)

# plt.xlabel('product_id', fontsize=14)

# plt.title('Top seller with most purchased product', fontsize=14)

# plt.show()
topseller_purchased =  Order_Items.groupby('seller_id').sum().sort_values('order_item_id', ascending=False)['order_item_id']

topseller_purchased[:10]
topseller_byomset =  Order_Items.groupby('seller_id').sum().sort_values('price', ascending=False)['price']

topseller_byomset[:10]
Order['order_delivered_customer_date'] = Order['order_delivered_customer_date'].fillna(method='ffill')

Order['customer_delivered_date'] = pd.to_datetime(Order['order_delivered_customer_date'],format='%Y-%m-%d').dt.date

Order['estimated_delivered_date'] = pd.to_datetime(Order['order_estimated_delivery_date'],format='%Y-%m-%d').dt.date

Order['delivered_difference'] = (Order['estimated_delivered_date'] - Order['customer_delivered_date']).dt.days

Delivered_Order = (Order[Order.order_status == 'delivered']).sort_values('estimated_delivered_date')

delivered_diff = Delivered_Order.groupby('order_estimated_delivery_date').mean()



delay_mean = delivered_diff['delivered_difference'].mean()

x_ticks = np.arange(0, len(delivered_diff.index), 15)

plt.figure(figsize=(20,7))

plt.plot(delivered_diff.index, delivered_diff['delivered_difference'], label = 'delivered difference')

plt.axhline(y=delay_mean, color='r', linestyle='--', label = 'y (mean)= '+ str(delay_mean))

plt.xticks(delivered_diff.index[x_ticks], rotation=-80, fontsize=12)

plt.yticks(fontsize=14)

plt.xlabel('Date', fontsize=16)

plt.ylabel('Difference', fontsize=16)

plt.legend(loc='best')

plt.title('Differences between delivered time and estimated time Since 2016-10-04 to 2018-10-25', fontsize=16)

plt.show()
approved = pd.to_datetime(Order['order_approved_at'],format='%Y-%m-%d').dt.date

delivered = pd.to_datetime(Order['order_delivered_customer_date'],format='%Y-%m-%d').dt.date

diff = (delivered - approved).dt.days

Delivered_Order = (Order[Order.order_status == 'delivered']).sort_values('order_approved_at')

Delivered_Order['diff_approved_delivered'] = diff

Delivered_Order['approved_date'] = pd.to_datetime(Order['order_approved_at'],format='%Y-%m-%d').dt.date

Delivered_Order = Delivered_Order.groupby('approved_date').mean()



x_ticks = np.arange(0, len(Delivered_Order.index), 15)

avg = Delivered_Order['diff_approved_delivered'].mean()

plt.figure(figsize=(20,7))

plt.plot(Delivered_Order.index, Delivered_Order['diff_approved_delivered'], label = 'diff_approved_delivered')

plt.axhline(y=avg, color='r', linestyle='--', label = 'y (mean)= '+ str(avg))

plt.xticks(Delivered_Order.index[x_ticks], rotation=-80, fontsize=12)

plt.yticks(fontsize=14)

plt.xlabel('Date', fontsize=16)

plt.ylabel('Difference', fontsize=16)

plt.legend(loc='best')

plt.title('Differences between approved date and delivered date Since 2016-10-04 to 2018-10-25', fontsize=16)

plt.show()
OrderItems = pd.merge(Order, Order_Items, on='order_id').fillna(method='ffill')

DeliveredOrder = OrderItems[OrderItems.order_status == 'delivered'].reset_index()

approved = pd.to_datetime(DeliveredOrder['order_approved_at'],format='%Y-%m-%d').dt.date

delivered = pd.to_datetime(DeliveredOrder['order_delivered_customer_date'],format='%Y-%m-%d').dt.date

diff = (delivered - approved).dt.days

DeliveredOrder['aprroved_delivered_diff'] = diff

DeliveredOrder = DeliveredOrder[DeliveredOrder.order_status == 'delivered']



tanggal = []

for tang in DeliveredOrder.order_approved_at:

    tanggal.append(tang[:10])



DeliveredOrder['approved_date'] = tanggal

plt.figure(figsize=(15,30))

plt.subplots_adjust(hspace=0.5, wspace=0.4)

for i in range(len(topseller_purchased[:10])):

    plt.subplot(math.ceil(len(topseller_purchased[:10])),2,i+1)

    grouped = DeliveredOrder[DeliveredOrder.seller_id == topseller_purchased.index[i]]

    avg = grouped['aprroved_delivered_diff'].mean()

    grouped = grouped.sort_values('approved_date').reset_index()

    grouped = grouped.groupby('approved_date').mean()

    x_axis = np.arange(len(grouped.index))

    plt.plot(x_axis, grouped['aprroved_delivered_diff'], '.-', label='approved_delivered_difference')

    plt.axhline(y=avg, color='r', linestyle='--', label = 'y (mean)= '+ str(avg))

    plt.xlabel('Since ' + str(grouped.index[0]) + " to " + str(grouped.index[len(grouped.index)-1]))

    plt.title('Seller_id : ' + topseller_purchased.index[i])    

    plt.legend(loc='best')

plt.show()
data_avg = {}

coef = []

intercept = []

RMSE = []

for i in range(len(topseller_purchased)):

    try :

        grouped = DeliveredOrder[DeliveredOrder.seller_id == topseller_purchased.index[i]]

        avg = grouped['aprroved_delivered_diff'].mean()

        data_avg[topseller_purchased.index[i]] = avg

        grouped = grouped.sort_values('approved_date').reset_index()

        grouped = grouped.groupby('approved_date').mean()

        x_axis = np.arange(len(grouped.index)).reshape(-1,1)

        y_axis = np.array(grouped['aprroved_delivered_diff']).reshape(-1,1)

        

        lin_reg = LinearRegression()

        lin_reg.fit(x_axis, y_axis)

        coef.append(float(lin_reg.coef_))

        intercept.append(float(lin_reg.intercept_))

        y_pred = lin_reg.predict(x_axis)

        lin_rmse = mean_squared_error(y_axis, y_pred)

        lin_rmse = np.sqrt(lin_rmse)

        RMSE.append(float(lin_rmse))

        

    except :

        coef.append(np.nan)

        intercept.append(np.nan)

        RMSE.append(np.nan)
result_calculated = pd.DataFrame(data_avg.items(), columns=['seller_id', 'mean_delivery_time'])

result_calculated['coef'] = coef

result_calculated['intercept'] = intercept

result_calculated['RMSE'] = RMSE

positif = result_calculated[result_calculated.coef > 0].count()[0]

negatif = result_calculated[result_calculated.coef < 0].count()[0]

total = positif + negatif

print('positif (%) : ', positif*100/total)

print('negatif (%) : ', negatif*100/total)
seller_delivery_performance = pd.DataFrame(data_avg.items(), columns=['seller_id', 'delivery_time_avg(days)'])

seller_delivery_performance = seller_delivery_performance[seller_delivery_performance['delivery_time_avg(days)'] > 0]

seller_delivery_performance['delivery_time_avg(days)'].hist(bins=1000, figsize=(10,8))

plt.axis([0,40,0,115])

plt.xlabel('average delivery time (days)', fontsize=14)

plt.ylabel('counts', fontsize=14)

plt.show()
unique_customer = {}

for uniq in Customers.customer_unique_id:

    if uniq in unique_customer:

        unique_customer[uniq] += 1

        

    else :

        unique_customer[uniq] = 1

        

unique_customer_sorted = sorted(unique_customer.items(), key = lambda x: x[1], reverse=True)

unique_customer_sorted = pd.DataFrame(unique_customer_sorted, columns =['customer_unique_id','total purchase'])

Customer_grouped = Customers.groupby('customer_unique_id').first()

Customer_grouped = (Customer_grouped['customer_state']).reset_index()

Customer_grouped = pd.merge(Customer_grouped, unique_customer_sorted, on='customer_unique_id')

Customer_grouped = Customer_grouped.sort_values('total purchase', ascending=False)

total_customer = Customer_grouped['customer_unique_id'].nunique()

plt.figure(figsize=(20,6))

plt.ylabel('Total Customer', fontsize=16)

plt.xlabel('City Name', fontsize=16)

plt.xticks(fontsize=14)

plt.yticks(fontsize=14)

Customer_grouped['customer_state'].value_counts().plot.bar()

plt.title('Number of unique customers by city', fontsize=18)

plt.text(20,35000, 'Total Customer = ' + str(total_customer), fontsize=20)

plt.show()
OrderItems = pd.merge(Order, Order_Items, on='order_id').fillna(method='ffill')

OrderItems['total'] = OrderItems['price'] + OrderItems['freight_value']

Sorted_OrderItems = OrderItems.sort_values('order_approved_at').reset_index()



purchased_perday = {}

for i in range(len(Sorted_OrderItems['order_approved_at'])):

    tanggal = Sorted_OrderItems['order_approved_at'][i][:10]

    if tanggal in purchased_perday:

        purchased_perday[tanggal] += Sorted_OrderItems['total'][i]

    else :

        purchased_perday[tanggal] = Sorted_OrderItems['total'][i]

    

purchased_perday = pd.DataFrame(purchased_perday.items(), columns=['Date', 'total (USD)'])



# x_axis = np.arange(0, len(purchased_perday['Date']))

x_ticks = np.arange(0, len(purchased_perday['Date']), 15)

avg_payment = purchased_perday['total (USD)'].mean()

plt.figure(figsize=(20,7))

plt.plot(purchased_perday['Date'], purchased_perday['total (USD)'], label='Payment')

plt.xticks(purchased_perday['Date'][x_ticks], rotation=-80, fontsize=12)

plt.axhline(y= avg_payment, color='r', linestyle='--', label = 'y (mean)= '+ str(avg_payment) + ' USD')

plt.yticks(fontsize=14)

plt.xlabel('Date', fontsize=16)

plt.ylabel('Total Payment', fontsize=16)

plt.title('Total Payment Everyday Since 2016-09-15 to 2018-09-03', fontsize=16)



from sklearn.linear_model import LinearRegression

x_ticks = np.arange(len(purchased_perday['Date'])).reshape(-1,1)

lin_reg = LinearRegression()

lin_reg.fit(x_ticks, purchased_perday['total (USD)'])

x =[x_ticks[0],x_ticks[-1]]

y_LR = lin_reg.intercept_ + x * lin_reg.coef_

plt.plot(x, y_LR, 'g-', label='Linear Regression m : ' + str(lin_reg.coef_) + ' , c : [' + str(lin_reg.intercept_) +']')



from sklearn.svm import LinearSVR

svm_reg = LinearSVR(epsilon=1.5)

svm_reg.fit(x_ticks, purchased_perday['total (USD)'])

y_SVR = svm_reg.intercept_ + x * svm_reg.coef_

plt.plot(x, y_SVR, 'y-', label='SVM Regression, m : ' + str(svm_reg.coef_) + ' , c : ' + str(svm_reg.intercept_))

plt.legend(loc='best', fontsize=16)

plt.show()



from sklearn.metrics import mean_squared_error

LR_predict = lin_reg.predict(x_ticks)

lin_rmse = mean_squared_error(purchased_perday['total (USD)'], LR_predict)

lin_rmse = np.sqrt(lin_rmse)

print('Linear RMSE : ',lin_rmse)



SVR_predict = svm_reg.predict(x_ticks)

svr_rmse = mean_squared_error(purchased_perday['total (USD)'], SVR_predict)

svr_rmse = np.sqrt(svr_rmse)

print('SVR RMSE : ' ,svr_rmse)
DeliveredOrder = OrderItems[OrderItems.order_status == 'delivered'].reset_index()

tanggal = []

for tang in DeliveredOrder.order_purchase_timestamp:

    tanggal.append(tang[:10])



DeliveredOrder['order_purchased_date'] = tanggal

plt.figure(figsize=(15,30))

plt.subplots_adjust(hspace=0.5, wspace=0.4)

for i in range(len(topseller_purchased[:10])):

    plt.subplot(math.ceil(len(topseller_purchased[:10])),2,i+1)

    grouped = DeliveredOrder[DeliveredOrder.seller_id == topseller_purchased.index[i]]

    grouped = grouped.sort_values('order_purchase_timestamp').reset_index()

    grouped = grouped.groupby('order_purchased_date').sum()

    x_axis = np.arange(len(grouped.index))

    plt.plot(x_axis, grouped['order_item_id'], '.-')

    plt.xlabel('Since ' + str(grouped.index[0]) + " to " + str(grouped.index[len(grouped.index)-1]))

    plt.title('Seller_id : ' + topseller_purchased.index[i])



plt.show()
coef = []

for i in range(len(topseller_purchased)):

    try :

        grouped = DeliveredOrder[DeliveredOrder.seller_id == topseller_purchased.index[i]]

        grouped = grouped.sort_values('order_purchase_timestamp').reset_index()

        grouped = grouped.groupby('order_purchased_date').sum()

        x_axis = np.arange(len(grouped.index)).reshape(-1,1)

        y_axis = grouped['order_item_id']

        

        lin_reg.fit(x_axis, y_axis)

        coef.append(float(lin_reg.coef_))

        

    except :

        coef.append(np.nan)
data_coef = pd.DataFrame(topseller_purchased.items(), columns=['seller_id', 'total_purchased'])

data_coef['coef'] = coef

positif = data_coef[data_coef.coef > 0]['coef'].count()

negatif = data_coef[data_coef.coef < 0]['coef'].count()

total = positif + negatif

print('positif(%) : ', positif*100/total)

print('negatif(%) : ', negatif*100/total)