import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.ticker as mtick

geolocation = pd.read_csv('../input/geolocation_olist_public_dataset.csv')
geolocation.head()
public_classified = pd.read_csv('../input/olist_classified_public_dataset.csv')
public_classified.head()
# Columns of dataset
public_classified.columns
public_dataset = pd.read_csv('../input/olist_public_dataset_v2.csv')
public_dataset.head()
# Dataset Columns
public_dataset.columns
customers = pd.read_csv('../input/olist_public_dataset_v2_customers.csv')
customers.head()
payments = pd.read_csv('../input/olist_public_dataset_v2_payments.csv')
payments.head()
public_classified.shape
types_class = public_classified.groupby('most_voted_class').sum()
types = types_class[['votes_before_estimate',
             'votes_delayed',
             'votes_low_quality',
             'votes_return',
             'votes_not_as_anounced',
             'votes_partial_delivery',
             'votes_other_delivery',
             'votes_other_order',
             'votes_satisfied']]
sns.set()
types.plot(kind='bar', stacked=True, figsize=(12,6))
type_class_city = public_classified.groupby(['most_voted_class']).agg({'most_voted_class':'count'})
type_class_city.plot.pie(subplots=True, figsize=(8,8), startangle=90, autopct='%.2f')
k = public_classified.groupby(['customer_state','most_voted_class'])['most_voted_class']\
                        .size().groupby(level=0)\
                        .apply(lambda x: 100 * x / x.sum())\
                        .unstack()
k.fillna(0)

ax = k.plot(kind='bar',stacked=True, figsize=(20,10), title='% to Type of Response')
pl = ax.legend(bbox_to_anchor=(1.2, 0.5))

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.1f} %'.format(height), (p.get_x()+.15*width, p.get_y()+.4*height))

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

invoice_state = public_dataset.groupby(['customer_state']).sum()
invoice_state['order_products_value'].plot.bar(figsize=(16,8))
invoice_state = public_dataset.groupby(['customer_state']).count()
invoice_state['order_products_value'].plot.bar(figsize=(16,8))
invoice_state = public_dataset.groupby(['customer_state']).median()
invoice_state['order_products_value'].plot.bar(figsize=(16,8))
invoice_state = public_dataset.groupby(['customer_state']).median()

#invoice_state['order_products_value'].plot.pie(subplots=True, figsize=(8,8), startangle=90, autopct='%.2f')
invoice_state['order_freight_value'].plot.bar(figsize=(16,8))
type_payments = payments.groupby(['payment_type']).agg({'payment_type':'count'})
type_payments.plot.pie(subplots=True, figsize=(8,8), startangle=90, autopct='%.2f')
merge_payments = pd.merge(payments, public_dataset, on=['order_id'])

result_payments = merge_payments.groupby(['customer_state']).median()

#invoice_state['order_products_value'].plot.pie(subplots=True, figsize=(8,8), startangle=90, autopct='%.2f')
result_payments['installments'].plot.bar(figsize=(16,8))
k = merge_payments.groupby(['customer_state','payment_type'])['value'].median().unstack().fillna(0)

ax = k.plot(kind='bar', stacked=True, figsize=(20,10))

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.2f}'.format(height), (p.get_x()+.15*width, p.get_y()+.4*height))


k = merge_payments.groupby(['customer_state','payment_type'])['payment_type']\
                        .size().groupby(level=0)\
                        .apply(lambda x: 100 * x / x.sum())\
                        .unstack()
k.fillna(0)

ax = k.plot(kind='bar',stacked=True, figsize=(20,10))
pl = ax.legend(bbox_to_anchor=(1.2, 0.5))

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.1f} %'.format(height), (p.get_x()+.15*width, p.get_y()+.4*height))

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())

order_p = pd.to_datetime(merge_payments['order_purchase_timestamp'], errors='coerce')
order_a = pd.to_datetime(merge_payments['order_aproved_at'], errors='coerce')
order_estimated_delivery = pd.to_datetime(merge_payments['order_estimated_delivery_date'], errors='coerce')
order_delivery = pd.to_datetime(merge_payments['order_delivered_customer_date'], errors='coerce')

# difference time in payment (in hours)
difference_time_payment = (order_a - order_p).astype('timedelta64[m]')
merge_payments['difference_time_payment'] = difference_time_payment

# difference time in estimetad delivery (in days)
difference_time_delivery = (order_delivery - order_estimated_delivery).astype('timedelta64[h]')/24
merge_payments['difference_time_delivery'] = difference_time_delivery

# difference time in aprovad to delivery
time_to_delivery = (order_delivery - order_a ).astype('timedelta64[h]')/24
merge_payments['time_to_delivery'] = time_to_delivery
k = merge_payments.groupby(['payment_type'])['difference_time_payment'].median().fillna(0)

ax = k.plot(kind='bar', stacked=True, figsize=(20,10))

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.2f}'.format(height), (p.get_x()+.15*width, p.get_y()+.4*height))

k = merge_payments.groupby(['customer_state'])['difference_time_delivery'].median().fillna(0)

ax = k.plot(kind='bar', stacked=True, figsize=(20,10))

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.2f}'.format(height), (p.get_x()+.15*width, p.get_y()+.4*height))
k = merge_payments.groupby(['customer_state'])['time_to_delivery'].median().fillna(0)

ax = k.plot(kind='bar', stacked=True, figsize=(20,10))

for p in ax.patches:
    width, height = p.get_width(), p.get_height()
    x, y = p.get_xy() 
    ax.annotate('{:.2f}'.format(height), (p.get_x()+.15*width, p.get_y()+.4*height))
