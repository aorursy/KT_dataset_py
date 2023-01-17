import numpy as np
import pandas as pd
# Seaborn and matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
# Plotly library
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
import colorlover as cl
# Others
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
init_notebook_mode(connected=True)

# Load datasets
payments = pd.read_csv("../input/olist_public_dataset_v2_payments.csv")
orders = pd.read_csv("../input/olist_public_dataset_v2.csv")
reviews = pd.read_csv("../input/olist_classified_public_dataset.csv")
geo = pd.read_csv("../input/geolocation_olist_public_dataset.csv")
customers = pd.read_csv("../input/olist_public_dataset_v2_customers.csv")
translation = pd.read_csv("../input/product_category_name_translation.csv")
payments['value_log'] = payments['value'].apply(lambda x: np.log(x) if x > 0 else 0)
unique_ = payments['order_id'].nunique()
print("DataFrame shape: {}; unique order ids: {}".format(payments.shape, unique_))
payments.head()
def plot_dist(values, log_values, title, color="#D84E30"):
    fig, axis = plt.subplots(1, 2, figsize=(12,4))
    axis[0].set_title("{} - linear scale".format(title))
    axis[1].set_title("{} - logn scale".format(title))
    ax1 = sns.distplot(values, color=color, ax=axis[0])
    ax2 = sns.distplot(log_values, color=color, ax=axis[1])
log_value = payments.value.apply(lambda x: np.log(x) if x > 0 else 0)
plot_dist(payments.value, log_value, "Value distribution")
payments.describe()
method_count = payments['payment_type'].value_counts().to_frame().reset_index()
method_value = payments.groupby('payment_type')['value'].sum().to_frame().reset_index()
# Plotly piechart
colors = None
trace1 = go.Pie(labels=method_count['index'], values=method_count['payment_type'],
                domain= {'x': [0, .48]}, marker=dict(colors=colors))
trace2 = go.Pie(labels=method_value['payment_type'], values=method_value['value'],
                domain= {'x': [0.52, 1]}, marker=dict(colors=colors))
layout = dict(title= "Number of payments (left) and Total payments value (right)", 
              height=400, width=800,)
fig = dict(data=[trace1, trace2], layout=layout)
iplot(fig)
ax = sns.catplot(x="payment_type", y="value",data=payments, aspect=2, height=3.8)
plt.figure(figsize=(10,4))
plt.title("Payments distributions - logn scale")
p1 = sns.kdeplot(payments[payments.payment_type == 'credit_card']['value_log'], color="navy", label='Credit card')
p2 = sns.kdeplot(payments[payments.payment_type == 'boleto']['value_log'], color="orange", label='Boleto')
p3 = sns.kdeplot(payments[payments.payment_type == 'voucher']['value_log'], color="green", label='Voucher')
p4 = sns.kdeplot(payments[payments.payment_type == 'debit_card']['value_log'], color="red", label='Debit card')
payments[payments['installments'] > 1]['payment_type'].value_counts().to_frame()
ins_count = payments.groupby('installments').size()
ins_mean = payments.groupby('installments')['value'].mean()

trace0 = go.Bar(
    x=ins_count.index,
    y=ins_count.values,
    name='Number of orders',
    marker=dict(color='rgb(49,130,189)')
)
trace1 = go.Bar(
    x=ins_mean.index,
    y=ins_mean.values,
    name='Mean value',
    marker=dict(color='rgb(204,204,204)')
)
fig = tools.make_subplots(rows=1, cols=2, print_grid=False)
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=400, width=800, title='Number of installments',
                     legend=dict(orientation="h"))
iplot(fig)
pay_one_inst = payments[payments['installments'] == 1]
method_count = pay_one_inst['payment_type'].value_counts().to_frame().reset_index()
method_value = pay_one_inst.groupby('payment_type')['value'].sum().to_frame().reset_index()
# Plotly piechart
colors = None
trace1 = go.Pie(labels=method_count['index'], values=method_count['payment_type'],
                domain= {'x': [0, .48]}, marker=dict(colors=colors))
trace2 = go.Pie(labels=method_value['payment_type'], values=method_value['value'],
                domain= {'x': [0.52, 1]}, marker=dict(colors=colors))
layout = dict(title= "Orders and value for a single installment", 
              height=400, width=800,)
fig = dict(data=[trace1, trace2], layout=layout)
iplot(fig)
unique_ = orders['order_id'].nunique()
print("DataFrame shape: {}; unique order ids: {}".format(orders.shape, unique_))
orders.head(3)
orders[orders['order_id'] == '000330af600103828257923c9aa98ae2']
count_products = orders.groupby('order_id').size().value_counts()
trace = go.Bar(
    x= count_products.index,
    y= count_products.values,
    marker=dict(
        color=['rgba(204,204,204,1)', 'rgba(222,45,38,0.8)',
               'rgba(204,204,204,1)', 'rgba(204,204,204,1)',
               'rgba(204,204,204,1)']),
)
layout = go.Layout(title='Number of orders for number of products', height=420, width=800)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename='color-bar')
# Products value
sum_value = orders.groupby('order_id')['order_products_value'].sum()
plot_dist(sum_value, np.log(sum_value), 'Products value')
# Freights value
sum_value = orders.groupby('order_id')['order_freight_value'].sum()
plot_dist(sum_value, sum_value.apply(lambda x: np.log(x) if x > 0 else 0), 'Freight value', color="#122aa5")
# Product value by date
orders['datetime'] =  pd.to_datetime(orders['order_purchase_timestamp'])
value_date = orders.groupby([orders['datetime'].dt.date])['order_products_value'].sum()
freight_date = orders.groupby([orders['datetime'].dt.date])['order_freight_value'].sum()
# Plot timeseries
trace0 = go.Scatter(x=value_date.index.astype(str), y=value_date.values, opacity = 0.8, name='Product value')
trace1 = go.Scatter(x=freight_date.index.astype(str), y=freight_date.values, opacity = 0.8, name='Freight value')
layout = dict(
    title= "Product and freight value by date",
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label='1m', step='month', stepmode='backward'),
                dict(count=6, label='6m', step='month', stepmode='backward'),
                dict(count=12, label='12m', step='month', stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(visible = True),
        type='date'
    )
)
fig = dict(data= [trace0, trace1], layout=layout)
iplot(fig)

# Sales for month
value_month = orders[['datetime', 'order_products_value']].copy()
value_month.set_index('datetime', inplace=True)
value_month = value_month.groupby(pd.Grouper(freq="M"))['order_products_value'].sum()
trace = go.Bar(x= value_month.index, y= value_month.values)
layout = go.Layout(title='Sales per month (product value)', height=420, width=800)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)
# Orders by category (less 1000 orders grouped into others)
orders_count = orders.groupby('product_category_name').size()
orders_count['others'] = orders_count[orders_count < 1000].sum()
orders_count = orders_count[orders_count >= 1000].sort_values(ascending=True)
orders_value = orders.groupby('product_category_name')['order_products_value'].sum()
orders_value = orders_value[orders_count.index]
translation = pd.Series(translation.product_category_name_english.values, index=translation.product_category_name)

trace0 = go.Bar(
    y=translation[orders_count.index],
    x=orders_count.values,
    name='Number of orders',
    orientation='h',
    marker=dict(color='rgb(49,130,189)')
)
trace1 = go.Bar(
    y=translation[orders_value.index],
    x=orders_value.values,
    name='Total value',
    orientation='h',
    marker=dict(color='rgb(204,204,204)')
)
fig = tools.make_subplots(rows=1, cols=2, print_grid=False)
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)

fig['layout'].update(
    height=1000,
    width=800,
    title='Products category',
    margin=dict(l=150, r=10, t=100, b=100),
    legend=dict(orientation="h")
)
fig['layout']['xaxis1'].update(title='Orders by category', domain=[0, 0.40])
fig['layout']['xaxis2'].update(title='Products value by category', domain=[0.6, 1])
iplot(fig)
items_count = orders.groupby('order_items_qty').size()
sellers_count = orders.groupby('order_sellers_qty').size()

trace0 = go.Bar(
    x=items_count.index,
    y=items_count.values,
    name='#Orders',
    marker=dict(color='rgb(49,130,189)')
)
trace1 = go.Bar(
    x=sellers_count.index,
    y=sellers_count.values,
    name='#Orders',
    marker=dict(color='rgb(204,204,204)')
)
fig = tools.make_subplots(rows=1, cols=2, print_grid=False)
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=400, width=800, title='Items and Sellers quantity')
fig['layout']['xaxis1'].update(title='Items quantity', domain=[0, 0.40])
fig['layout']['xaxis2'].update(title='Sellers quantity', domain=[0.6, 1])
iplot(fig)
#product_name_lenght	product_description_lenght	product_photos_qty
fig, axis = plt.subplots(1, 2, figsize=(12,4))
axis[0].set_title("Produt name lenght")
axis[1].set_title("Product description lenght")
ax1 = sns.distplot(orders['product_name_lenght'], color="#D84E30", ax=axis[0]) #rgba(204,204,204,1)', 'rgba(222,45,38,0.8)'
ax2 = sns.distplot(orders['product_description_lenght'], color="#7E7270", ax=axis[1]) #"#D84E30"
photo_qty = orders.groupby('product_photos_qty').size()
photo_value = orders.groupby('product_photos_qty')['order_products_value'].mean()
trace0 = go.Bar(
    x=photo_qty.index,
    y=photo_qty.values,
    name='Number of Orders',
    marker=dict(color='rgba(222,45,38,0.8)')
)
trace1 = go.Bar(
    x=photo_value.index,
    y=photo_value.values,
    name='Produt mean value',
    marker=dict(color='rgba(204,204,204, 0.8)')
)
fig = tools.make_subplots(rows=1, cols=2, print_grid=False)
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=400, width=800, title='Photo quantity',
                     legend=dict(orientation="h"))
#fig['layout']['xaxis1'].update(title='photo quantity', domain=[0, 0.40])
#fig['layout']['xaxis2'].update(title='photo quantity', domain=[0.6, 1])
iplot(fig)
review_qty = orders.groupby('review_score').size()
review_value = orders.groupby('review_score')['order_products_value'].mean()
trace0 = go.Bar(
    x=review_qty.index,
    y=review_qty.values,
    name='Number of orders',
    marker=dict(color='rgb(49,130,189)')
)
trace1 = go.Bar(
    x=review_value.index,
    y=review_value.values,
    name='Produt mean value',
    marker=dict(color='rgb(204,204,204)')
)
fig = tools.make_subplots(rows=1, cols=2, print_grid=False)
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=400, width=800, title='Review Score')
fig['layout']['xaxis1'].update(title='review score', domain=[0, 0.40])
fig['layout']['xaxis2'].update(title='review score', domain=[0.6, 1])
iplot(fig)
# Convert columns to datetime
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
orders['order_aproved_at'] = pd.to_datetime(orders['order_aproved_at'])
orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])
orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
# Calculate differences in hours
orders['delivery_time'] = (orders['order_delivered_customer_date'] - orders['order_aproved_at']).dt.total_seconds() / 86400
orders['estimated_delivery_time'] = (orders['order_estimated_delivery_date'] - orders['order_aproved_at']).dt.total_seconds() / 86400
# Delivery estimated time and actual delivery time
plt.figure(figsize=(10,4))
plt.title("Delivery time in days")
ax1 = sns.kdeplot(orders['delivery_time'].dropna(), color="#D84E30", label='Delivery time')
ax2 = sns.kdeplot(orders['estimated_delivery_time'].dropna(), color="#7E7270", label='Estimated delivery time')
ax = sns.catplot(x="review_score", y="delivery_time", kind="box",
                 data=orders[orders.delivery_time < 60], height=4, aspect=1.5)
reviews.head(3)
class_voted = reviews.groupby('most_voted_class').size()
subclass_voted = reviews.groupby('most_voted_subclass').size()
trace0 = go.Bar(
    x=class_voted.index,
    y=subclass_voted.values,
    name='Number of reviews',
    marker=dict(color='rgb(49,130,189)')
)
trace1 = go.Bar(
    x=subclass_voted.index,
    y=subclass_voted.values,
    name='Number of reviews',
    marker=dict(color='rgba(204,204,204, 0.8)')
)
fig = tools.make_subplots(rows=1, cols=2, print_grid=False)
fig.append_trace(trace0, 1, 1)
fig.append_trace(trace1, 1, 2)

fig['layout'].update(height=400, width=810, title='Most voted class (left) and subclass (right)')
iplot(fig)
review_qty = orders.groupby('review_score').size()
review_value = orders.groupby('review_score')['order_products_value'].mean()

fig = tools.make_subplots(rows=4, cols=2, print_grid=False)
cols = ['votes_before_estimate', 'votes_delayed', 'votes_low_quality', 'votes_return',
        'votes_not_as_anounced', 'votes_partial_delivery', 'votes_other_delivery', 'votes_other_order',
        'votes_satisfied']
cols_color = ["#F97B40", "#DA6C38", "#BB5C30", "#9C4D28",
             "#7C3D20", "#5D2E18", "#3E1F10", "#1F0F08"]

col_index = 0
for i in range(4):
    for j in range(2):
        count_ = reviews.groupby(cols[col_index]).size()
        trace = go.Bar(
            x=count_.index,
            y=count_.values,
            name=cols[col_index],
            marker=dict(color=cols_color[col_index])
        )
        fig.append_trace(trace, i+1, j+1)
        col_index += 1

fig['layout'].update(height=900, width=800, title='Votes')
iplot(fig)
print("DataFrame shape:", geo.shape)
geo.head(3)
count_state = geo['state'].value_counts()
count_state['others'] = count_state[count_state < 10000].sum()
trace = go.Bar(
    x= count_state[count_state >= 10000].index,
    y= count_state[count_state >= 10000].values,
    marker=dict(color='rgba(204,204,204, 0.9)')
)
layout = go.Layout(title='Number of rows for each state', height=400, width=800)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename='color-bar')
count_city = geo['city'].value_counts()
count_city['others'] = count_city[count_city < 3000].sum()
trace = go.Bar(
    x= count_city[count_city >= 3000].index,
    y= count_city[count_city >= 3000].values,
    marker=dict(color='rgba(222,45,38,0.8)')
)
layout = go.Layout(title='Number of rows for each city', height=400, width=800)
fig = go.Figure(data=[trace], layout=layout)
iplot(fig, filename='color-bar')
unique_ = customers['customer_unique_id'].nunique()
print("DataFrame shape: {}; unique customers: {}".format(customers.shape, unique_))
customers.head(3)
orders[orders.customer_id == '109cf3ecc53afd27745a79a618cb5ec4']
customers[customers.customer_id == "109cf3ecc53afd27745a79a618cb5ec4"]
customers[customers.customer_unique_id == 'b237307cd63e0bd318ec30a97ad25fce']
orders[orders.customer_id == 'f64b4d4b9e4185ce59b00e617e565bca']