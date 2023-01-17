import pandas as pd

import numpy as np



dfOrdersItems = pd.read_csv('../input/brazilian-ecommerce/olist_order_items_dataset.csv')

dfOrders = pd.read_csv('../input/brazilian-ecommerce/olist_orders_dataset.csv')

dfProducts = pd.read_csv('../input/brazilian-ecommerce/olist_products_dataset.csv')

dfOrdersReviews = pd.read_csv('../input/brazilian-ecommerce/olist_order_reviews_dataset.csv')

dfSellers = pd.read_csv('../input/brazilian-ecommerce/olist_sellers_dataset.csv')

dfCustomers = pd.read_csv('../input/brazilian-ecommerce/olist_customers_dataset.csv')

dfGeolocation = pd.read_csv('../input/olist-modified-data/geolocation_v2.csv')

dfTranslation = pd.read_csv('../input/brazilian-ecommerce/product_category_name_translation.csv')
dfProducts = dfProducts.merge(dfTranslation, on='product_category_name')

dfProducts.drop('product_category_name', axis=1, inplace=True)

dfProducts.rename(columns={

    'product_category_name_english' : 'product_category'

}, inplace=True)

dfProducts = dfProducts[['product_id','product_category']]

dfProducts.head()
dfSellerx = pd.merge(dfSellers, dfGeolocation, left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix')

dfSellerx.rename(columns={

    'geolocation_lat' : 'seller_lat',

    'geolocation_lng' : 'sellet_lng',

}, inplace=True)

dfSellerx = dfSellerx[['seller_id','seller_lat','sellet_lng']]

dfSellerx.head()
dfCustomerx = pd.merge(dfCustomers, dfGeolocation, left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix')

dfCustomerx.rename(columns={

    'geolocation_lat' : 'customer_lat',

    'geolocation_lng' : 'customer_lng',

}, inplace=True)

dfCustomerx = dfCustomerx[['customer_id','customer_lat','customer_lng']]

dfCustomerx.head()
df = pd.merge(dfOrders, dfOrdersItems, on='order_id', how='right')

df = df.merge(dfProducts, on='product_id')

df = df.merge(dfOrdersReviews, on='order_id')

df = df.merge(dfSellerx, on='seller_id')

df = df.merge(dfCustomerx, on='customer_id')

df = df.rename(columns={'price':'product_price','order_item_id':'quantity'})

df = df.drop(['review_id', 'review_creation_date','review_answer_timestamp','review_comment_title','review_comment_message','customer_id','product_id',], axis=1)

df.columns
df = df[df['order_status'] == 'delivered']

df.head()
df = pd.read_csv('../input/olist-modified-data/final_project_v2.csv')

df.head()
from math import sin, cos, sqrt, atan2, radians

df['distance'] = df[['seller_lat','seller_lng','customer_lat','customer_lng']].apply(

    lambda row : round(6373.0 * (2 * atan2(sqrt((sin((radians(row['customer_lat']) - radians(row['seller_lat']))/2))**2 + cos(radians(row['seller_lat'])) * cos(radians(row['customer_lat'])) * (sin((radians(row['customer_lng']) - radians(row['seller_lng']))/2))**2), sqrt(1-((sin((radians(row['customer_lat']) - radians(row['seller_lat']))/2))**2 + cos(radians(row['seller_lat'])) * cos(radians(row['customer_lat'])) * (sin((radians(row['customer_lng']) - radians(row['seller_lng']))/2))**2)))))

    , axis=1

)

df.head()
df['freight_rate'] = df[['freight_value','quantity']].apply(

    lambda row : round(row['freight_value'] / row['quantity'],2), axis=1

)

df.head()
df = pd.read_csv('../input/olist-modified-data/final_project_v4.csv')

df.head()
for item in ['order_purchase_timestamp','order_approved_at','order_delivered_carrier_date','order_delivered_customer_date','order_estimated_delivery_date','shipping_limit_date']:

    df[item] = pd.to_datetime(df[item])
df['approving_time'] = df[['order_purchase_timestamp','order_approved_at']].apply(

    lambda row : str(row['order_approved_at'] - row['order_purchase_timestamp']).split(' ')[0], axis=1

)

df['processing_time'] = df[['order_approved_at','order_delivered_carrier_date']].apply(

    lambda row : str(row['order_delivered_carrier_date'] - row['order_approved_at']).split(' ')[0], axis=1

)

df['delivery_time'] = df[['order_delivered_carrier_date','order_delivered_customer_date']].apply(

    lambda row : str(row['order_delivered_customer_date'] - row['order_delivered_carrier_date']).split(' ')[0], axis=1

)

df['courier_on_time'] = df[['order_delivered_customer_date','order_estimated_delivery_date']].apply(

    lambda row : 1 if row['order_delivered_customer_date'] <= row['order_estimated_delivery_date'] else 0, axis=1

)

df['seller_on_time'] = df[['order_delivered_carrier_date','shipping_limit_date']].apply(

    lambda row : 1 if row['order_delivered_carrier_date'] <= row['shipping_limit_date'] else 0, axis=1

)

df.head()
df = df[df.approving_time != 'NaT']

df = df[df.processing_time != 'NaT']

df = df[df.delivery_time != 'NaT']
df['higher_product_price'] = df[['product_price','avg_price']].apply(

    lambda row: 1 if row['product_price'] > row['avg_price'] else 0, axis = 1

)

df['higher_freight_rate'] = df[['freight_rate','avg_freight']].apply(

    lambda row: 1 if row['freight_rate'] > row['avg_freight'] else 0, axis = 1

)

df['better_product'] = df[['product_rating','avg_rating']].apply(

    lambda row: 1 if row['product_rating'] > row['avg_rating'] else 0, axis = 1

)

df.head()
df = df[['product_rating','avg_rating','better_product','product_price','avg_price','higher_product_price','seller_rating','product_sold','distance','freight_rate','avg_freight','higher_freight_rate','approving_time','processing_time','delivery_time','courier_on_time','seller_on_time','review_score']]

df = df.rename(columns={

    'avg_rating' : 'avg_product_category_rating',

    'avg_price' : 'avg_product_category_price',

    'avg_freight' : 'avg_freight_rate',

    'product_sold' : 'product_sold_by_seller'

})

df.head()
df['approving_time'] = pd.to_numeric(df['approving_time'])

df['processing_time'] = pd.to_numeric(df['processing_time'])

df['delivery_time'] = pd.to_numeric(df['delivery_time'])
df = pd.read_csv('../input/olist-modified-data/final_project_v5.csv')

df = df.rename(columns={

    'better_product' : 'better_product_rating'

})

df = df.drop(['avg_product_category_rating','avg_product_category_price','avg_freight_rate'], axis=1)
corr = df.corr()

corr['review_score']
df = df.drop(['higher_product_price','higher_freight_rate','freight_rate','product_price'], axis=1)

corr = df.corr()

corr['review_score']
from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(

    df.drop('review_score', axis=1),

    df['review_score'],

    test_size = .1

)
from sklearn import tree 

model = tree.DecisionTreeClassifier()

model.fit(xTrain,yTrain)

print('model score =',model.score(xTrain,yTrain))
from sklearn.metrics import accuracy_score

prediction = model.predict(xTest)

print('accuracy =',accuracy_score(yTest, prediction)*100,'%')
from sklearn.metrics import mean_squared_error

predictions = model.predict(xTrain)

forest_mse = mean_squared_error(yTrain, predictions)

forest_rmse = np.sqrt(forest_mse)

print('error = ',forest_rmse)
from sklearn.ensemble import RandomForestClassifier

model2 = RandomForestClassifier(n_estimators=100)

model2.fit(xTrain,yTrain)

print('model score =',model2.score(xTrain,yTrain))
from sklearn.metrics import accuracy_score

prediction = model2.predict(xTest)

print('accuracy =',accuracy_score(yTest, prediction)*100,'%')
from sklearn.metrics import mean_squared_error

predictions = model2.predict(xTrain)

forest_mse = mean_squared_error(yTrain, predictions)

forest_rmse = np.sqrt(forest_mse)

print('error = ',forest_rmse)
import matplotlib.pyplot as plt

import seaborn as sns

fig = plt.figure(figsize = (18,15))

for i, item in enumerate (df.drop(['better_product_rating','courier_on_time','seller_on_time','review_score'], axis=1)):

    plt.subplot(3,3,i+1)

    sns.distplot(df[item])
fig = plt.figure(figsize = (18,15))

for i, item in enumerate (df.drop(['better_product_rating','courier_on_time','seller_on_time','review_score'], axis=1)):

    plt.subplot(3,3,i+1)

    sns.boxplot(df[item])
df1 = df

for item in df1.describe().columns:

    df1 = df1[df[item] < (df1.describe()[item].iloc[1] + (df1.describe()[item].iloc[2]*3))]

    df1 = df1[df[item] > (df1.describe()[item].iloc[1] - (df1.describe()[item].iloc[2]*3))]
df1.shape
fig = plt.figure(figsize = (18,15))

for i, item in enumerate (df1.drop(['better_product_rating','courier_on_time','seller_on_time','review_score'], axis=1)):

    plt.subplot(3,3,i+1)

    sns.distplot(df1[item])
fig = plt.figure(figsize = (18,15))

for i, item in enumerate (df1.drop(['better_product_rating','courier_on_time','seller_on_time','review_score'], axis=1)):

    plt.subplot(3,3,i+1)

    sns.boxplot(df1[item])
corr = df1.corr()

corr['review_score']
from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(

    df1.drop('review_score', axis=1),

    df1['review_score'],

    test_size = .1

)
from sklearn.ensemble import RandomForestClassifier

modelR1 = RandomForestClassifier(n_estimators=100)

modelR1.fit(xTrain,yTrain)

print('model score =',modelR1.score(xTrain,yTrain))
from sklearn.metrics import accuracy_score

prediction = modelR1.predict(xTest)

print('accuracy =',accuracy_score(yTest, prediction)*100,'%')
from sklearn.metrics import mean_squared_error

predictions = modelR1.predict(xTrain)

forest_mse = mean_squared_error(yTrain, predictions)

forest_rmse = np.sqrt(forest_mse)

print('error = ',forest_rmse)
df2 = df

from scipy import stats

z = np.abs(stats.zscore(df2.drop(['better_product_rating','courier_on_time','seller_on_time','review_score'],axis=1)))

df2 = df2[(z<3).all(axis=1)]
df2.shape
fig = plt.figure(figsize = (18,15))

for i, item in enumerate (df2.drop(['better_product_rating','courier_on_time','seller_on_time','review_score'], axis=1)):

    plt.subplot(3,3,i+1)

    sns.distplot(df2[item])
fig = plt.figure(figsize = (18,15))

for i, item in enumerate (df2.drop(['better_product_rating','courier_on_time','seller_on_time','review_score'], axis=1)):

    plt.subplot(3,3,i+1)

    sns.boxplot(df2[item])
from sklearn.model_selection import train_test_split

xTrain, xTest, yTrain, yTest = train_test_split(

    df2.drop('review_score', axis=1),

    df2['review_score'],

    test_size = .1

)
from sklearn.ensemble import RandomForestClassifier

modelR2 = RandomForestClassifier(n_estimators=100)

modelR2.fit(xTrain,yTrain)

print('model score =',modelR2.score(xTrain,yTrain))
from sklearn.metrics import accuracy_score

prediction = modelR2.predict(xTest)

print('accuracy =',accuracy_score(yTest, prediction)*100,'%')
from sklearn.metrics import mean_squared_error

predictions = modelR2.predict(xTrain)

forest_mse = mean_squared_error(yTrain, predictions)

forest_rmse = np.sqrt(forest_mse)

print('error = ',forest_rmse)
xTrainPrediction = df.drop('review_score', axis=1).iloc[100:]

yTrainPrediction = df['review_score'].iloc[100:]

xTestPrediction = df.drop('review_score', axis=1).iloc[:100]

yTestPrediction = df['review_score'].iloc[:100]



xTrainProduct = df[['product_rating','better_product_rating']].iloc[100:]

yTrainProduct = df['review_score'].iloc[100:]

xTestProduct = df[['product_rating','better_product_rating']].iloc[:100]

yTestProduct = df['review_score'].iloc[:100]



xTrainSeller = df[['product_sold_by_seller','seller_rating','approving_time','processing_time','seller_on_time']].iloc[100:]

yTrainSeller = df['review_score'].iloc[100:]

xTestSeller = df[['product_sold_by_seller','seller_rating','approving_time','processing_time','seller_on_time']].iloc[:100]

yTestSeller = df['review_score'].iloc[:100]



xTrainCourier = df[['delivery_time','courier_on_time','distance']].iloc[100:]

yTrainCourier = df['review_score'].iloc[100:]

xTestCourier = df[['delivery_time','courier_on_time','distance']].iloc[:100]

yTestCourier = df['review_score'].iloc[:100]
from sklearn.ensemble import RandomForestClassifier

modelPrediction = RandomForestClassifier(n_estimators=100)

modelPrediction.fit(xTrainPrediction,yTrainPrediction)

print('model score =',modelPrediction.score(xTrainPrediction,yTrainPrediction))
from sklearn.ensemble import RandomForestClassifier

modelProduct = RandomForestClassifier(n_estimators=100)

modelProduct.fit(xTrainProduct,yTrainProduct)

print('model score =',modelProduct.score(xTrainProduct,yTrainProduct))
from sklearn.metrics import accuracy_score

prediction = modelProduct.predict(xTestProduct)

print('accuracy =',accuracy_score(yTestProduct, prediction)*100,'%')
from sklearn.metrics import mean_squared_error

predictions = modelProduct.predict(xTrainProduct)

forest_mse = mean_squared_error(yTrainProduct, predictions)

forest_rmse = np.sqrt(forest_mse)

print('error = ',forest_rmse)
from sklearn.ensemble import RandomForestClassifier

modelSeller = RandomForestClassifier(n_estimators=100)

modelSeller.fit(xTrainSeller,yTrainSeller)

print('model score =',modelSeller.score(xTrainSeller,yTrainSeller))
from sklearn.metrics import accuracy_score

prediction = modelSeller.predict(xTestSeller)

print('accuracy =',accuracy_score(yTestSeller, prediction)*100,'%')
from sklearn.metrics import mean_squared_error

predictions = modelSeller.predict(xTrainSeller)

forest_mse = mean_squared_error(yTrainSeller, predictions)

forest_rmse = np.sqrt(forest_mse)

print('error = ',forest_rmse)
from sklearn.ensemble import RandomForestClassifier

modelCourier = RandomForestClassifier(n_estimators=100)

modelCourier.fit(xTrainCourier,yTrainCourier)

print('model score =',modelCourier.score(xTrainCourier,yTrainCourier))
from sklearn.metrics import accuracy_score

prediction = modelCourier.predict(xTestCourier)

print('accuracy =',accuracy_score(yTestCourier, prediction)*100,'%')
from sklearn.metrics import mean_squared_error

predictions = modelCourier.predict(xTrainCourier)

forest_mse = mean_squared_error(yTrainCourier, predictions)

forest_rmse = np.sqrt(forest_mse)

print('error = ',forest_rmse)
for i in range (0,100,10):

    print('index ',i)

    print('actual data =', yTestPrediction.iloc[i])

    print('rating prediction = ', modelPrediction.predict([xTestPrediction.iloc[i]]))

    print('product performance = ', modelProduct.predict([xTestProduct.iloc[i]]))

    print('seller performance = ', modelSeller.predict([xTestSeller.iloc[i]]))

    print('courier performane = ', modelCourier.predict([xTestCourier.iloc[i]]))

    print('----------------------------------------------------------------------------')
for i in range (0,100,10):

    print('index ',i+5)

    print('actual data =', yTestPrediction.iloc[i+5])

    print('rating prediction = ', modelPrediction.predict([xTestPrediction.iloc[i+5]]))

    print('product performance = ', modelProduct.predict([xTestProduct.iloc[i+5]]))

    print('seller performance = ', modelSeller.predict([xTestSeller.iloc[i+5]]))

    print('courier performane = ', modelCourier.predict([xTestCourier.iloc[i+5]]))

    print('----------------------------------------------------------------------------')