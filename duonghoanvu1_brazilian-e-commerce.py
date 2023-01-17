Image(url= "https://i.imgur.com/iLNlNfs.jpg")
Image(url= "https://i.imgur.com/BX0QYJh.jpg")
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# Visualization
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('retina')
import seaborn as sns
sns.set_style('whitegrid', {'grid.linestyle': '--'})
from IPython.display import Image
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
from plotly import tools
import colorlover as cl

# Others
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
init_notebook_mode(connected=True)
mql = pd.read_csv('../input/marketing-funnel/marketing funnel by olist/olist_marketing_qualified_leads_dataset.csv', 
                  parse_dates=['first_contact_date'])
cd = pd.read_csv('../input/marketing-funnel/marketing funnel by olist/olist_closed_deals_dataset.csv',
                parse_dates=['won_date'])
sellers = pd.read_csv('../input/brazilian-ecommerce/olist_sellers_dataset.csv')
order_items = pd.read_csv('../input/brazilian-ecommerce/olist_order_items_dataset.csv')

products = pd.read_csv("../input/brazilian-ecommerce/olist_products_dataset.csv")
payments = pd.read_csv("../input/brazilian-ecommerce/olist_order_payments_dataset.csv")
orders = pd.read_csv("../input/brazilian-ecommerce/olist_orders_dataset.csv")
reviews = pd.read_csv("../input/brazilian-ecommerce/olist_order_reviews_dataset.csv")
geo = pd.read_csv("../input/brazilian-ecommerce/olist_geolocation_dataset.csv")
customers = pd.read_csv("../input/brazilian-ecommerce/olist_customers_dataset.csv")
product_translation = pd.read_csv('../input/brazilian-ecommerce/product_category_name_translation.csv')
#helper function
def pichart_with_table(main_df,column_name,title,top_n,filename):
    fig = plt.figure(figsize=(10,6))

    summary = main_df.groupby(column_name)["mql_id"].nunique().sort_values(ascending=False)
    df = pd.DataFrame({'source':summary.index, 'counts':summary.values})
    labels = df['source']
    counts = df['counts']

    ax1 = fig.add_subplot(121)
    if top_n > 0:
        ax1.pie(counts[0:top_n], labels=labels[0:top_n], autopct='%1.1f%%', startangle=180)
    else:
        ax1.pie(counts, labels=labels, autopct='%1.1f%%', startangle=180)
    ax1.set_title(title)
    ax1.axis('equal')

    ax2 = fig.add_subplot(122)
    font_size=10
    ax2.axis('off')
    if top_n > 0:
        df_table = ax2.table(cellText=df.values[0:top_n], colLabels=df.columns, loc='center',colWidths=[0.8,0.2])
    else:
        df_table = ax2.table(cellText=df.values, colLabels=df.columns, loc='center',colWidths=[0.8,0.2])

    df_table.auto_set_font_size(False)
    df_table.set_fontsize(font_size)

    fig.tight_layout()
    plt.savefig(filename)
    plt.show()
mql.describe()
cd.describe(include="all")
zero_count = (mql.isnull()).sum() # (df1 == 0).sum()
zero_count_df = pd.DataFrame(zero_count)
#zero_count_df.drop('Survived', axis=0, inplace=True)
zero_count_df.columns = ['Missing Value Count']

# https://stackoverflow.com/questions/31859285/rotate-tick-labels-for-seaborn-barplot/60530167#60530167
sns.set(style='whitegrid')
plt.figure(figsize=(13,4))
sns.barplot(x=zero_count_df.index, y=zero_count_df['Missing Value Count'])
plt.title('Marketing Qualified Lead')
plt.xticks(rotation=30)


zero_count = (cd.isnull()).sum() # (df1 == 0).sum()
zero_count_df = pd.DataFrame(zero_count)
#zero_count_df.drop('Survived', axis=0, inplace=True)
zero_count_df.columns = ['Missing Value Count']

sns.set(style='whitegrid')
plt.figure(figsize=(13,5))
sns.barplot(x=zero_count_df.index, y=zero_count_df['Missing Value Count'])
plt.title('Closed Deals')
plt.xticks(rotation=90)
# Remove unnecessary columns
cd = cd.drop(['has_company','has_gtin','average_stock','declared_product_catalog_size'], axis = 1) 

#filling the rest NAs
rest_cols = ['lead_behaviour_profile','business_segment','lead_type','business_type']
cd[rest_cols] = cd[rest_cols].fillna('unknown')
mql['origin'].fillna('unknown')

#checking if any null is remaining or not
cd[cd.isnull().any(axis=1)]
geo =  geo.drop_duplicates(subset=['geolocation_zip_code_prefix'], keep='first')

# Check duplicated value
geo[geo['geolocation_zip_code_prefix'] == 1037]

#geo['geolocation_zip_code_prefix'].duplicated().sum()
data = pd.merge(products,order_items,
                how='inner', on='product_id')
data = pd.merge(data, orders,
                how='inner', on='order_id')
data = pd.merge(data, payments,
                how='inner', on='order_id')
data = pd.merge(data, reviews,
                how='inner', on='order_id')
data = pd.merge(data, customers,
                how='inner', on='customer_id')
data = pd.merge(data, sellers,
                how='inner', on='seller_id')
data = pd.merge(data, geo, 
                how='inner', 
                left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix')
data = pd.merge(data, product_translation,
                how='left', on='product_category_name') # There are some data without english names
data.shape
# Add a 'year-month' column
mql['first_contact_date(y-m)'] = mql['first_contact_date'].dt.to_period('M')
mql[['first_contact_date', 'first_contact_date(y-m)']].head(3)
# Create time series table
monthly_mql = mql.groupby(by='first_contact_date(y-m)').mql_id.count()
monthly_mql.to_frame().T
a = monthly_mql.to_frame()

a.plot.line(figsize=(12, 6))

"""
a['first_contact_date(y-m)'] = a['first_contact_date(y-m)'].astype('str')

plt.figure(figsize = (15, 6))

sns.lineplot(x = a['first_contact_date(y-m)'], y = a['mql_id'])
""" 
pichart_with_table(mql,"origin","Origin Sources",-1,"origin_mql.png")
mql_origin = pd.pivot_table(mql,
                            index='origin',
                            columns='first_contact_date(y-m)',
                            values='mql_id',                            
                            aggfunc='count',
                            fill_value=0)
mql.groupby('origin')['mql_id'].count().sort_values(ascending=False)
# Sort index from largest to smallest in volume
# == origin_list = mql['origin'].value_counts()
origin_list = mql.groupby('origin')['mql_id'] \
                                   .count().sort_values(ascending=False).index
mql_origin = mql_origin.reindex(origin_list)
mql_origin
# Plot the monthly volume by channel
plt.figure(figsize=(20,8))
sns.heatmap(mql_origin, annot=True, fmt='g');
landing_page_origin = mql['landing_page_id'].value_counts()
landing_page_origin[:10].plot.bar()
# Merge 'MQL' with 'closed deals'
# Merge by 'left' in order to evaluate conversion rate
mql_cd = pd.merge(mql,
                  cd,
                  how='left',
                  on='mql_id')
# Add a column to distinguish signed MOLs from MQLs who left without signing up
mql_cd['seller_id(bool)'] = mql_cd['seller_id'].notna()
mql_cd.head()
# alternative: mql_cd.groupby('first_contact_date(y-m)')['seller_id'].count()
# Compute monthly closed deals
monthly_cd = mql_cd.groupby('first_contact_date(y-m)')['seller_id(bool)'].sum()
monthly_cd
# Plot the monthly volume of closed deals
monthly_cd.plot.line(figsize=(12, 6))
plt.title('Closed Deal Volume (Jun 2017 - May 2018)', fontsize=14);
monthly_conversion = mql_cd.groupby(by='first_contact_date(y-m)')['seller_id(bool)'].agg(['count', 'sum'])

monthly_conversion['conversion_rate(%)'] = ((monthly_conversion['sum'] / monthly_conversion['count']) * 100).round(1)
monthly_conversion
# Plot the monthly conversion rate
monthly_conversion['conversion_rate(%)'].plot.line(figsize=(12, 6))
plt.ylabel('In Percentage Terms')
plt.title('Conversion Rate (Jun 2017 - May 2018)', fontsize=14);
mql_cd.head(3)
# .dt.days -> work for  TimedeltaProperties

# Calculate sales length in days
mql_cd['sales_length(day)'] = (mql_cd['won_date'] - mql_cd['first_contact_date']).dt.days
mql_cd[['first_contact_date', 'won_date', 'sales_length(day)']].head()
# won_date always occur after first_contact_date, thus sales_length(day) must be > 0
# remove the outliers
mql_cd = mql_cd[mql_cd['sales_length(day)'] > 0]
sns.distplot(mql_cd['sales_length(day)'])
# Separate sales length for each year
closed_deal = (mql_cd['seller_id'].notna())
lead_2017 = (mql_cd['first_contact_date'].dt.year.astype('str') == '2017')
lead_2018 = (mql_cd['first_contact_date'].dt.year.astype('str') == '2018')

sales_length_2017 = mql_cd[closed_deal & lead_2017]['sales_length(day)']
sales_length_2018 = mql_cd[closed_deal & lead_2018]['sales_length(day)']
figure, ax = plt.subplots(figsize=(12,6))

sns.kdeplot(sales_length_2017,
            cumulative=True,
            label='2017 (Jun-Dec)',
            ax=ax)
sns.kdeplot(sales_length_2018,
            cumulative=True,
            label='2018 (Jan-May)',
            ax=ax)

ax.set_title('Sales Length in Days', fontsize=14)
ax.xaxis.set_major_locator(plt.MaxNLocator(10))
plt.xlim(0,500);
print('There are',len(cd[cd.declared_monthly_revenue>0]), 'MQLs in profit') #total count
print('Only around',round(len(cd[cd.declared_monthly_revenue>0])/len(cd)*100, 2), '% in total') #total percent of data
fig = plt.figure(figsize=(8,5))
cd[cd['declared_monthly_revenue']>0]['declared_monthly_revenue'].value_counts().plot.bar()
plt.title("Revenue of closed deals ($)")
plt.xlabel("Amount")
plt.ylabel("Total number of sellers")
fig.tight_layout()
#plt.savefig("revenue_disclosed.png")
plt.show()
# Bring 'closed deals' data
cd_profile = cd[cd['lead_behaviour_profile'].notna()].copy()

cd_profile['lead_behaviour_profile'].value_counts()
# Combine four types of mixed profiles(2.4%) into 'others'
profile_list = ['cat', 'eagle', 'wolf', 'shark']

cd_profile['lead_behaviour_profile(upd)'] = cd_profile['lead_behaviour_profile'].map(lambda profile: profile 
                                                                                     if profile in profile_list else 'others')
cd_profile['lead_behaviour_profile(upd)'].value_counts()
cd_profile[['lead_type', 'lead_behaviour_profile(upd)']].head()
# Create 'profile - lead type' table
cols = cd_profile['lead_type'].value_counts().index
index = cd_profile['lead_behaviour_profile(upd)'].value_counts().index
index = index.rename('lead_behaviour_profile(upd)')

profile_leadType = pd.pivot_table(cd_profile,
                                  index='lead_behaviour_profile(upd)',
                                  columns='lead_type',
                                  values='seller_id',
                                  aggfunc='count',
                                  fill_value=0)
profile_leadType = profile_leadType.reindex(index)[cols]
profile_leadType
# Create 'profile - business type' table
cols = cd_profile['business_type'].value_counts().index
index = cd_profile['lead_behaviour_profile(upd)'].value_counts().index

profile_businessType = pd.pivot_table(cd_profile,
                                      index='lead_behaviour_profile(upd)',
                                      columns='business_type',
                                      values='seller_id',
                                      aggfunc='count',
                                      fill_value=0)

profile_businessType = profile_businessType.reindex(index)[cols]
profile_businessType
# Create 'profile - business segment' table
cols = cd_profile['business_segment'].value_counts().index
index = cd_profile['lead_behaviour_profile(upd)'].value_counts().index

profile_segment = pd.pivot_table(cd_profile,
                                 index='lead_behaviour_profile(upd)',
                                 columns='business_segment',
                                 values='seller_id',
                                 aggfunc='count',
                                 fill_value=0)

profile_segment = profile_segment.reindex(index)[cols]
profile_segment
Image(url= "https://i.imgur.com/hujp7Hc.jpg")
# Plot the above three tables
figure, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20,20))
figure.subplots_adjust(hspace=0.3)

sns.heatmap(profile_leadType,
            annot=True,
            fmt='g',
            ax=ax1)
sns.heatmap(profile_businessType,
            annot=True,
            fmt='g',
            ax=ax2)
sns.heatmap(profile_segment,
            annot=True,
            fmt='g',
            ax=ax3)

ax1.set_title('Behaviour Profile - Lead Type', fontsize=14)
ax2.set_title('Behaviour Profile - Business Type', fontsize=14)
ax3.set_title('Behaviour Profile - Business Segement', fontsize=14);
# Create 'profile-SDR' table
cols = cd_profile['sdr_id'].value_counts().index
index = cd_profile['lead_behaviour_profile(upd)'].value_counts().index

profile_sdr = pd.pivot_table(cd_profile,
                             index='lead_behaviour_profile(upd)',
                             columns='sdr_id',
                             values='seller_id',
                             aggfunc='count',
                             fill_value=0)

profile_sdr = profile_sdr.reindex(index)[cols] # Sort SDR in descending order of volume 
profile_sdr
# Create 'profile-SR' table
cols = cd_profile['sr_id'].value_counts().index
index = cd_profile['lead_behaviour_profile(upd)'].value_counts().index

profile_sr = pd.pivot_table(cd_profile,
                            index='lead_behaviour_profile(upd)',
                            columns='sr_id',
                            values='seller_id',
                            aggfunc='count',
                            fill_value=0)

profile_sr = profile_sr.reindex(index)[cols] # Sort SR in descending order of volume
profile_sr
# Plot the two tables
figure, (ax1,ax2) = plt.subplots(2, 1, figsize=(20,14))
figure.subplots_adjust(hspace=0.2)

sns.heatmap(profile_sdr,
            annot=True,
            fmt='g',
            ax=ax1)
sns.heatmap(profile_sr,
            annot=True,
            fmt='g',
            ax=ax2)

ax1.set_title('SDR Performance in Descending Volume', fontsize=14)
ax2.set_title('SR Performance in Descending Volume', fontsize=14)
ax1.set_xticks([])
ax2.set_xticks([]);
all_data = pd.merge(cd,order_items,
                how='inner', on='seller_id')
all_data = pd.merge(all_data, orders,
                how='inner', on='order_id')
all_data = pd.merge(all_data, products,
                how='inner', on='product_id')
all_data = pd.merge(all_data, product_translation,
                how='left', on='product_category_name') # There are some data without english names
all_data.shape
all_data['order_purchase_timestamp'] = pd.to_datetime(all_data['order_purchase_timestamp'])
# Sort out orders not devliered to customers
all_data = all_data[all_data['order_status'] == 'delivered']

# Add a 'year-month' column
all_data['order_purchase_timestamp(y-m)'] = all_data['order_purchase_timestamp'].dt.to_period('M')

print(all_data.shape)
all_data.head(3)
cols = all_data.groupby(by='business_segment') \
           .price \
           .sum() \
           .sort_values(ascending=False) \
           .index

monthly_segment_revenue = all_data.groupby(['order_purchase_timestamp(y-m)', 'business_segment']) \
                              .price \
                              .sum() \
                              .unstack(level=1, fill_value=0)

monthly_segment_revenue = monthly_segment_revenue[cols]
monthly_segment_revenue
# Plot the monthly revenues by segment
monthly_segment_revenue.plot.area(figsize=(20,15))

plt.title('Monthly Revenues by Business Segment', fontsize=14)
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5));
# Create watches segment dataframe
watches = all_data[all_data['business_segment'] == 'watches']
watches.shape
# Create monthly revenues by product category
cols = watches.groupby('product_category_name_english')['price'].sum().sort_values(ascending=False).index

# monthly_revenue_category = pd.pivot_table(watches,
#                                  index='order_purchase_timestamp(y-m)',
#                                  columns='product_category_name_english',
#                                  values='price',
#                                  aggfunc='sum',
#                                  fill_value=0)

monthly_revenue_category = watches.groupby(['order_purchase_timestamp(y-m)', 'product_category_name_english']) \
                                  .price \
                                  .sum() \
                                  .unstack(level=1, fill_value=0)

monthly_revenue_category = monthly_revenue_category[cols]
monthly_revenue_category
# Plot the monthly revenues by category
monthly_revenue_category.plot.area(figsize=(15,6))
plt.title('Monthly Revenues by Product Category of Watches', fontsize=14);
# Create 'seller - product category' table
cols = watches.groupby('product_category_name_english')['price'].sum().sort_values(ascending=False).index

watches_seller_revenue = watches.groupby(['seller_id', 'product_category_name_english']) \
                                ['price'].sum().unstack(level=1, fill_value=0)

watches_seller_revenue = watches_seller_revenue[cols]
watches_seller_revenue['total'] = watches_seller_revenue.sum(axis=1)

watches_seller_revenue
watches_seller_revenue.T
# Plot the above table
watches_seller_revenue.T.plot.barh(stacked=True, figsize=(12,6))

plt.title('Watches Revenue by Seller', fontsize=14)
plt.legend(loc='lower right');

index = watches[watches['product_category_name_english'] == 'watches_gifts'].groupby('product_id') \
                ['price'].sum().sort_values(ascending=False).index
product_seller_revenue = watches[watches['product_category_name_english'] == 'watches_gifts'] \
                                  .groupby(['seller_id', 'product_id']) \
                                  ['price'].sum().unstack(level=0, fill_value=0)

product_seller_revenue = product_seller_revenue.reindex(index)
product_seller_revenue.head()
product_seller_revenue.plot.bar(stacked=True, figsize=(20, 6))
plt.title('Product Revenues in Watches_gifts Category', fontsize=14);
a = product_seller_revenue.copy()
total = a['7d13fca15225358621be4086e1eb0964'].sum()
a['Percentage'] = round(a['7d13fca15225358621be4086e1eb0964'] / total * 100, 2)

t = 0
def sum_percentage(data):
    global t
    t = t + data
    return t

a['sum_percentage'] = a['Percentage'].apply(sum_percentage)

a['sum_percentage'].plot.bar(figsize=(20, 6))
payments.describe(include="all")
def plot_dist(values, log_values, title, color="#D84E30"):
    fig, axis = plt.subplots(1, 2, figsize=(12,4))
    axis[0].set_title("{} - linear scale".format(title))
    axis[1].set_title("{} - logn scale".format(title))
    ax1 = sns.distplot(values, color=color, ax=axis[0])
    ax2 = sns.distplot(log_values, color=color, ax=axis[1])
payments['payment_value_log'] = payments['payment_value'].apply(lambda x: np.log(x) if x > 0 else 0)

plot_dist(payments['payment_value'], payments['payment_value_log'], 'Payment Value distribution')
method_count = payments['payment_type'].value_counts().to_frame().reset_index()
method_value = payments.groupby('payment_type')['payment_value'].sum().to_frame().reset_index()

# Plotly piechart
colors = None
trace1 = go.Pie(labels=method_count['index'], values=method_count['payment_type'],
                domain= {'x': [0, .48]}, marker=dict(colors=colors))
trace2 = go.Pie(labels=method_value['payment_type'], values=method_value['payment_value'],
                domain= {'x': [0.52, 1]}, marker=dict(colors=colors))

layout = dict(title= "Number of payments (left) and Total payments value (right)", 
              height=400, width=800)
fig = dict(data=[trace1, trace2], layout=layout)
iplot(fig)
sns.catplot(x="payment_type", y="payment_value",data=payments, aspect=2, height=3.8)
gr = payments.groupby('payment_type')['payment_value_log']
plt.figure(figsize=(10,4))
for label, arr in gr:
    sns.kdeplot(arr, label=label)
payments[payments['payment_installments'] == 1]['payment_type'].value_counts().to_frame()
payments[payments['payment_installments'] > 1]['payment_type'].value_counts().to_frame()
payments.groupby('payment_installments')['payment_value'].mean()
ins_count = payments.groupby('payment_installments').size()
sns.barplot(x=ins_count.index, y=ins_count)
pay_one_inst = payments[payments['payment_installments'] == 1]
method_count = pay_one_inst['payment_type'].value_counts().to_frame().reset_index()
method_value = pay_one_inst.groupby('payment_type')['payment_value'].sum().to_frame().reset_index()
# Plotly piechart
colors = None
trace1 = go.Pie(labels=method_count['index'], values=method_count['payment_type'],
                domain= {'x': [0, .48]}, marker=dict(colors=colors))
trace2 = go.Pie(labels=method_value['payment_type'], values=method_value['payment_value'],
                domain= {'x': [0.52, 1]}, marker=dict(colors=colors))
layout = dict(title= "Orders and value for a single installment", 
              height=400, width=800,)
fig = dict(data=[trace1, trace2], layout=layout)
iplot(fig)
data.groupby('order_id').size().value_counts().to_frame().plot.bar(figsize=(12, 6))
# Products value
sum_value = data.groupby('order_id')['price'].sum()
plot_dist(sum_value, np.log1p(sum_value), 'Products value')

# Freights value
sum_value = data.groupby('order_id')['freight_value'].sum()
plot_dist(sum_value, np.log1p(sum_value), 'Freight value', color="#122aa5")
data['order_purchase_timestamp'] = pd.to_datetime(data['order_purchase_timestamp'])
value_month = data[['order_purchase_timestamp', 'price']].copy()
value_month['year-month'] = value_month['order_purchase_timestamp'].dt.to_period('M')
value_month.set_index('year-month', inplace=True)
value_month.groupby(pd.Grouper(freq="M"))['price'].sum().plot.bar()
orders_count = data.groupby('product_category_name_english').size()
orders_count['others'] = orders_count[orders_count < 1000].sum()
orders_count = orders_count[orders_count >= 1000].sort_values(ascending=True)

orders_value = data.groupby('product_category_name_english')['price'].sum()
orders_value = orders_value[orders_count.index]

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
axes = axes.flatten()
sns.barplot(x=orders_count.values[:-1], y=orders_count.index[:-1], ax=axes[0])
plt.ylabel('');
sns.barplot(x=orders_value.values[:-1], y=orders_value.index[:-1], ax=axes[1])
plt.ylabel('');
review_qty = data.groupby('review_score').size()
review_value = data.groupby('review_score')['price'].mean()

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
axes = axes.flatten()
sns.barplot(x=review_qty.index, y=review_qty.values, ax=axes[0])
plt.xlabel('');
sns.barplot(x=review_value.index, y=review_value.values, ax=axes[1])
plt.xlabel('review_score for mean price');
# Convert columns to datetime
data['order_purchase_timestamp'] = pd.to_datetime(data['order_purchase_timestamp'])
data['order_estimated_delivery_date'] = pd.to_datetime(data['order_estimated_delivery_date'])
data['order_delivered_customer_date'] = pd.to_datetime(data['order_delivered_customer_date'])

data['delivery_time'] = (data['order_delivered_customer_date'] - data['order_purchase_timestamp']).dt.days
data['estimated_delivery_time'] = (data['order_estimated_delivery_date'] - data['order_purchase_timestamp']).dt.days
plt.figure(figsize=(10,4))
plt.title("Delivery time in days")
ax1 = sns.kdeplot(data['delivery_time'].dropna(), color="#D84E30", label='Delivery time')
ax2 = sns.kdeplot(data['estimated_delivery_time'].dropna(), color="#7E7270", label='Estimated delivery time')
sns.boxplot(x="review_score", y="delivery_time",
            data=data[data['delivery_time'] < 60])
geo_state = data.groupby('geolocation_state')['geolocation_lat','geolocation_lng'].mean().reset_index()
geo_city = data.groupby('geolocation_city')['geolocation_lat','geolocation_lng'].mean().reset_index()
geo = [go.Scattermapbox(lon = geo_state['geolocation_lng'],
                        lat = geo_state['geolocation_lat'],
                        text = geo_state['geolocation_state'],
                        marker = dict(size = 18,
                                      color = 'tomato',))]

layout = dict(title = 'Brazil State',
              mapbox = dict(accesstoken = 'pk.eyJ1IjoiaG9vbmtlbmc5MyIsImEiOiJjam43cGhpNng2ZmpxM3JxY3Z4ODl2NWo3In0.SGRvJlToMtgRxw9ZWzPFrA',
                            center= dict(lat=-22,lon=-43),
                            bearing=10,
                            pitch=0,
                            zoom=2,))

fig = dict(data=geo, layout=layout)
iplot(fig, validate=False)
customer_by_location = data.copy()

geo_city.columns = ['geolocation_city', 'geolocation_city_lat', 'geolocation_city_lng']
geo_state.columns = ['geolocation_state', 'geolocation_state_lat', 'geolocation_state_lng']

customer_by_location = pd.merge(customer_by_location, geo_city,
                how='inner', on='geolocation_city')
customer_by_location = pd.merge(customer_by_location, geo_state,
                how='inner', on='geolocation_state')

city_spend = customer_by_location.groupby(['customer_city','geolocation_city_lng','geolocation_city_lat'])['price'].sum().to_frame().reset_index()
city_freight = customer_by_location.groupby(['customer_city','geolocation_city_lng','geolocation_city_lat'])['freight_value'].mean().reset_index()
state_spend = customer_by_location.groupby(['customer_state','geolocation_state_lng','geolocation_state_lat'])['price'].sum().to_frame().reset_index()
state_freight = customer_by_location.groupby(['customer_state','geolocation_state_lng','geolocation_state_lat'])['freight_value'].mean().reset_index()
state_freight['text'] = 'state: ' + state_freight['customer_state'] + ' | freight: ' + round(state_freight['freight_value'],1).astype(str)
state_order = [go.Scattergeo(lon = state_spend['geolocation_state_lng'],
                      lat = state_spend['geolocation_state_lat'],
                      text = state_freight['text'],
                      marker = dict(size = state_spend['price']/2000,
                                    sizemin = 5,
                                    color= state_freight['freight_value'],
                                    colorscale= 'balance',
                                    cmin = 20,
                                    cmax = 50,
                                    line = dict(width=0.1, color='rgb(40,40,40)'),
                                    sizemode = 'area'
                                   ),
                      name = 'State'),]
layout = dict(
        title = 'Brazilian E-commerce Order by States',
        showlegend = True,
        autosize=True,
        width = 900,
        height = 600,
        geo = dict(
            scope = "south america",
            projection = dict(type='winkel tripel', scale = 1),
            center = dict(lon=-47,lat=-22),
            showland = True,
            showcountries= True,
            showsubunits=True,
            landcolor = 'rgb(155, 155, 155)',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)"
        )
    )

fig = dict( data=state_order, layout=layout )
iplot( fig, validate=False)
count_state = data['geolocation_state'].value_counts()
count_state['others'] = count_state[count_state < 5000].sum()
count_state = count_state[count_state > 5000]

sns.barplot(x=count_state.index, y=count_state.values)
freigh_value = [go.Scattergeo(lon = state_freight['geolocation_state_lng'],
                      lat = state_freight['geolocation_state_lat'],
                      text = state_freight['text'],
                      marker = dict(size = (state_freight['freight_value']-10)**2,
                                    sizemin = 5,
                                    color= state_freight['freight_value'],
                                    colorscale= 'balance',
                                    cmin = 20,
                                    cmax = 50,
                                    line = dict(width=0.1, color='rgb(40,40,40)'),
                                    sizemode = 'area'
                                   ),
                      name = 'State'),]
layout = dict(
        title = 'Freight Values',
        showlegend = True,
        autosize=True,
        width = 900,
        height = 600,
        geo = dict(
            scope = "south america",
            projection = dict(type='winkel tripel', scale = 1.0),
            center = dict(lon=-47,lat=-22),
            showland = True,
            showcountries= True,
            showsubunits=True,
            landcolor = 'rgb(155, 155, 155)',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)"
        )
    )

fig = dict( data=freigh_value, layout=layout)
iplot( fig, validate=False)
state_delivery_time = data.groupby('customer_state')['delivery_time'].mean().to_frame()

state_delivery_time.columns = ['avg_state_delivery_time']
state_delivery_time = state_delivery_time.reset_index()

state_delivery_time = pd.merge(state_delivery_time,state_spend,
                how='inner', on='customer_state')

# Clean delivery time
state_delivery_time = state_delivery_time[state_delivery_time['avg_state_delivery_time'] > 0]

# Show text on the graph
state_delivery_time['text'] = 'state: ' + state_delivery_time['customer_state'].astype('str') + ' ' + \
                            (state_delivery_time['avg_state_delivery_time'].apply(np.ceil)).astype('str') + ' days'
State_Delivery_Days = [go.Scattergeo(lon = state_delivery_time['geolocation_state_lng'],
                      lat = state_delivery_time['geolocation_state_lat'],
                      text = state_delivery_time['text'],
                      marker = dict(size = 20,
                                    sizemin = 5,
                                    color= ((state_delivery_time['avg_state_delivery_time'])**2),
                                    colorscale= 'agsunset',
                                    cmin = 10,
                                    cmax = 255,
                                    line = dict(width=0.1, color='rgb(40,40,40)'),
                                    sizemode = 'area'
                                   ),
                      name = 'State'),]
layout = dict(
        title = 'State Delivery Days',
        showlegend = True,
        autosize=True,
        width = 900,
        height = 600,
        geo = dict(
            scope = "south america",
            projection = dict(type='winkel tripel', scale = 1.6),
            center = dict(lon=-47,lat=-22),
            showland = True,
            showcountries= True,
            showsubunits=True,
            landcolor = 'rgb(155, 155, 155)',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)"
        )
    )

fig = dict( data=State_Delivery_Days, layout=layout)
iplot( fig, validate=False)
geo = [go.Scattermapbox(lon = geo_city['geolocation_city_lng'],
                        lat = geo_city['geolocation_city_lat'],
                        text = geo_city['geolocation_city'],
                        marker = dict(size = 3,
                                      color = 'tomato',))]

layout = dict(title = 'Brazil City',
              mapbox = dict(accesstoken = 'pk.eyJ1IjoiaG9vbmtlbmc5MyIsImEiOiJjam43cGhpNng2ZmpxM3JxY3Z4ODl2NWo3In0.SGRvJlToMtgRxw9ZWzPFrA',
                            center= dict(lat=-22,lon=-43),
                            bearing=10,
                            pitch=0,
                            zoom=2,))

fig = dict(data=geo, layout=layout)
iplot(fig, validate=False)
count_city = data['geolocation_city'].value_counts()

plt.figure(figsize=(13,4))
sns.barplot(x=count_city.index[:10], y=count_city.values[:10])
city_delivery_time = data.groupby('customer_city')['delivery_time'].mean().to_frame()

city_delivery_time.columns = ['avg_city_delivery_time']
city_delivery_time = city_delivery_time.reset_index()

city_delivery_time = pd.merge(city_delivery_time,city_spend,
                how='inner', on='customer_city')

# Clean delivery time
city_delivery_time = city_delivery_time[city_delivery_time['avg_city_delivery_time'] > 0]

# Show text on the graph
city_delivery_time['text'] = 'city: ' + city_delivery_time['customer_city'].astype('str') + ' ' + city_delivery_time['avg_city_delivery_time'].astype('str') + ' days'
City_Delivery_Days = [go.Scattergeo(lon = city_delivery_time['geolocation_city_lng'],
                      lat = city_delivery_time['geolocation_city_lat'],
                      text = city_delivery_time['text'],
                      marker = dict(size = 3,
                                    sizemin = 5,
                                    color= ((city_delivery_time['avg_city_delivery_time'])**2),
                                    colorscale= 'agsunset',
                                    cmin = 10,
                                    cmax = 255,
                                    line = dict(width=0.1, color='rgb(40,40,40)'),
                                    sizemode = 'area'
                                   ),
                      name = 'City'),]
layout = dict(
        title = 'City Delivery Days',
        showlegend = True,
        autosize=True,
        width = 900,
        height = 600,
        geo = dict(
            scope = "south america",
            projection = dict(type='winkel tripel', scale = 1.6),
            center = dict(lon=-47,lat=-22),
            showland = True,
            showcountries= True,
            showsubunits=True,
            landcolor = 'rgb(155, 155, 155)',
            subunitwidth=1,
            countrywidth=1,
            subunitcolor="rgb(255, 255, 255)",
            countrycolor="rgb(255, 255, 255)"
        )
    )

fig = dict( data=City_Delivery_Days, layout=layout)
iplot( fig, validate=False)
delivery_time_city = city_delivery_time['customer_city'].value_counts()

plt.figure(figsize=(13,4))
sns.barplot(x=delivery_time_city.index[:10], y=delivery_time_city.values[:10])
print(customers['customer_id'].duplicated().sum())
print(customers['customer_unique_id'].duplicated().sum())