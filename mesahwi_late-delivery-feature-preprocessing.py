import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('../input/dataco-smart-supply-chain-for-big-data-analysis/DataCoSupplyChainDataset.csv',header= 0,encoding='unicode_escape')
pd.set_option('display.max_columns',None)
data.head()
#Converting categorical features that represent date and time to datetime datatype.
data['order_date'] = pd.to_datetime(data['order date (DateOrders)'])
data['shipping_date']=pd.to_datetime(data['shipping date (DateOrders)'])
# Handling Time and date variables
data['order_year'] = pd.DatetimeIndex(data['order_date']).year
data['order_month'] = pd.DatetimeIndex(data['order_date']).month
data['order_day'] = pd.DatetimeIndex(data['order_date']).day
data['shipping_year'] = pd.DatetimeIndex(data['shipping_date']).year
data['shipping_month'] = pd.DatetimeIndex(data['shipping_date']).month
data['shipping_day'] = pd.DatetimeIndex(data['shipping_date']).day
data.shape
# new_dataset_features = ['Type','Days for shipment (scheduled)','Late_delivery_risk','Benefit per order',
#                         'Sales per customer','Latitude','Longitude','Shipping Mode','Order Status','Order Region',
#                         'Order Country','Order City','Market','Delivery Status','order_day','order_month','order_year',
#                         'shipping_day','shipping_month','shipping_year']

# benefit per order / sales per customer / delivery status/ late delivery_risk / category name/ customer state / Order status / order item quantity / Product Price  / shipping mode
# len(new_dataset_features)
new_dataset_features = ['Days for shipment (scheduled)','Benefit per order','Sales per customer','Delivery Status','Category Name',
                        'Customer State', 'Order Status','Order Item Quantity',
                        'Product Price','Shipping Mode','Late_delivery_risk']

# benefit per order / sales per customer / delivery status / category name/ 
# customer state / Order status / order item quantity / 
# Product Price  / shipping mode / late delivery_risk
len(new_dataset_features)
new_data = data[new_dataset_features]
new_data['Late_delivery_risk'] = ['Late' if x==1 else 'Not Late' for x in new_data['Late_delivery_risk']]
new_data.head()
new_data['Delivery Status'].unique()
for i, x in enumerate(new_data['Late_delivery_risk']):
    if x=='Not Late':
        if new_data['Delivery Status'][i] == 'Late delivery':
            print('wow')
corrmap = new_data.corr()
top=corrmap.index
plt.figure(figsize=(10,5))
g=sns.heatmap(new_data[top].corr(),annot=True,cmap="RdYlGn")
new_data.to_csv('Late Delivery.csv', header=True)
