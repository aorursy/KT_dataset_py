import numpy as np

import pandas as pd

import seaborn as sns

sns.set(style="ticks")

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import matplotlib.ticker as ticker



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session



import sklearn

import scipy

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

pd.set_option('display.max_columns', 100)

%matplotlib inline
def format_spines(ax, right_border=True):

    

    ax.spines['bottom'].set_color('#666666')

    ax.spines['left'].set_color('#666666')

    ax.spines['top'].set_visible(False)

    if right_border:

        ax.spines['right'].set_color('#FFFFFF')

    else:

        ax.spines['right'].set_color('#FFFFFF')

    ax.patch.set_facecolor('#FFFFFF')



def count_plot(feature, df, colors='Greens_d', hue=False, ax=None, title=''):

    

    # Preparing variables

    ncount = len(df)

    if hue != False:

        ax = sns.countplot(x=feature, data=df, palette=colors, hue=hue, ax=ax)

    else:

        ax = sns.countplot(x=feature, data=df, palette=colors, ax=ax)

        

    format_spines(ax)

    

    # Setting percentage

    for p in ax.patches:

        x=p.get_bbox().get_points()[:,0]

        y=p.get_bbox().get_points()[1,1]

#        ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 

#                ha='center', va='bottom') # set the alignment of the text

        ax.annotate(y, (x.mean(), y), 

                ha='center', va='bottom') # set the alignment of the text

    

    # Final configuration

    if not hue:

        ax.set_title(df[feature].describe().name + ' Analysis', size=13, pad=15)

    else:

        ax.set_title(df[feature].describe().name + ' Analysis by ' + hue, size=13, pad=15)  

    if title != '':

        ax.set_title(title)       

    plt.tight_layout()

    

    

def bar_plot(x, y, df, colors='Blues_d', hue=False, ax=None, value=False, title=''):

    

    # Preparing variables

    try:

        ncount = sum(df[y])

    except:

        ncount = sum(df[x])

    #fig, ax = plt.subplots()

    if hue != False:

        ax = sns.barplot(x=x, y=y, data=df, palette=colors, hue=hue, ax=ax, ci=None)

    else:

        ax = sns.barplot(x=x, y=y, data=df, palette=colors, ax=ax, ci=None)



    # Setting borders

    format_spines(ax)



    # Setting percentage

    for p in ax.patches:

        xp=p.get_bbox().get_points()[:,0]

        yp=p.get_bbox().get_points()[1,1]

        if value:

            ax.annotate('{:.2f}k'.format(yp/1000), (xp.mean(), yp), 

                    ha='center', va='bottom') # set the alignment of the text

        else:

            ax.annotate('{:.1f}%'.format(100.*yp/ncount), (xp.mean(), yp), 

                    ha='center', va='bottom') # set the alignment of the text

    if not hue:

        ax.set_title(df[x].describe().name + ' Analysis', size=12, pad=15)

    else:

        ax.set_title(df[x].describe().name + ' Analysis by ' + hue, size=12, pad=15)

    if title != '':

        ax.set_title(title)  

    plt.tight_layout()

    

    

def categorical_plot(cols_cat, axs, df):

    

    idx_row = 0

    for col in cols_cat:

        # Returning column index

        idx_col = cols_cat.index(col)



        # Verifying brake line in figure (second row)

        if idx_col >= 3:

            idx_col -= 3

            idx_row = 1



        # Plot params

        names = df[col].value_counts().index

        heights = df[col].value_counts().values



        # Bar chart

        axs[idx_row, idx_col].bar(names, heights, color='navy')

        if (idx_row, idx_col) == (0, 2):

            y_pos = np.arange(len(names))

            axs[idx_row, idx_col].tick_params(axis='x', labelrotation=30)

        if (idx_row, idx_col) == (1, 1):

            y_pos = np.arange(len(names))

            axs[idx_row, idx_col].tick_params(axis='x', labelrotation=90)



        total = df[col].value_counts().sum()

        axs[idx_row, idx_col].patch.set_facecolor('#FFFFFF')

        format_spines(axs[idx_row, idx_col], right_border=False)

        for p in axs[idx_row, idx_col].patches:

            w, h = p.get_width(), p.get_height()

            x, y = p.get_xy()

            axs[idx_row, idx_col].annotate('{:.1%}'.format(h/1000), (p.get_x()+.29*w,

                                            p.get_y()+h+20), color='k')



        # Plot configuration

        axs[idx_row, idx_col].set_title(col, size=12)

        axs[idx_row, idx_col].set_ylim(0, heights.max()+120)
#Load Data

order = pd.read_csv('../input/brazilian-ecommerce/olist_orders_dataset.csv')

customer = pd.read_csv('../input/brazilian-ecommerce/olist_customers_dataset.csv')

review = pd.read_csv('../input/brazilian-ecommerce/olist_order_reviews_dataset.csv')

payment = pd.read_csv('../input/brazilian-ecommerce/olist_order_payments_dataset.csv')

order_item = pd.read_csv('../input/brazilian-ecommerce/olist_order_items_dataset.csv')

product = pd.read_csv('../input/brazilian-ecommerce/olist_products_dataset.csv')

seller = pd.read_csv('../input/brazilian-ecommerce/olist_sellers_dataset.csv')

geo = pd.read_csv('../input/brazilian-ecommerce/olist_geolocation_dataset.csv')

translation = pd.read_csv('../input/brazilian-ecommerce/product_category_name_translation.csv')
#Data Pre-processing 

#Translate Product Category: 

#product = product.merge(trans, on = 'product_category_name', how = 'left')

#product = product.drop('product_category_name', axis =1)



#Convert Timestamp Data

order_date=['order_purchase_timestamp', u'order_approved_at',

            u'order_delivered_carrier_date', u'order_delivered_customer_date',

            u'order_estimated_delivery_date']

for items in order_date:

    order[items] = pd.to_datetime(order[items],format='%Y-%m-%d %H:%M:%S')

    
# creating master dataframe 

payment.head()

print(payment.shape)

df1 = payment.merge(order_item, on='order_id')

print(df1.shape)

df2 = df1.merge(product, on='product_id')

print(df2.shape)

df3 = df2.merge(seller, on='seller_id')

print(df3.shape)

df4 = df3.merge(review, on='order_id')

print(df4.shape)

df5 = df4.merge(order, on='order_id')

print(df5.shape)

df6 = df5.merge(translation, on='product_category_name')

print(df6.shape)

df = df6.merge(customer, on='customer_id')

print(df.shape)
#cleaning up and re-engineering some columns

df['order_purchase_year'] = df.order_purchase_timestamp.apply(lambda x: x.year)

df['order_purchase_month'] = df.order_purchase_timestamp.apply(lambda x: x.month)

df['order_purchase_dayofweek'] = df.order_purchase_timestamp.apply(lambda x: x.dayofweek)

df['order_purchase_hour'] = df.order_purchase_timestamp.apply(lambda x: x.hour)

df['order_purchase_day'] = df['order_purchase_dayofweek'].map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})

df['order_purchase_mon'] = df.order_purchase_timestamp.apply(lambda x: x.month).map({1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'})

df['order_count']=1

df['year_month'] = df['order_purchase_timestamp'].dt.strftime('%Y-%m')



df['ship_duration']=(df['order_delivered_customer_date']-df['order_purchase_timestamp'])/24

df['ship_duration']=df['ship_duration'].astype('timedelta64[h]')



df['tocarrier_duration']=(df['order_delivered_carrier_date']-df['order_purchase_timestamp'])/24

df['tocarrier_duration']=df['tocarrier_duration'].astype('timedelta64[h]')



df['lastmile_duration']=(df['order_delivered_customer_date']-df['order_delivered_carrier_date'])/24

df['lastmile_duration']=df['lastmile_duration'].astype('timedelta64[h]')



df['expected_vs_shipdate']=(df['order_estimated_delivery_date']-df['order_delivered_customer_date'])/24

df['expected_vs_shipdate']=df['expected_vs_shipdate'].astype('timedelta64[h]')



df['expected_duration']=(df['order_estimated_delivery_date']-df['order_purchase_timestamp'])/24

df['expected_duration']=df['expected_duration'].astype('timedelta64[h]')
#CODE TO DELETE

#df['year_month']=df.order_purchase_timestamp.dt.to_period('M')

#df['month_year'] = df['order_purchase_year'].astype(str) + '-' + df['order_purchase_month'].astype(str) #won't be able to plot a correct x-axis with this

#df['order_purchase_month'] = df['order_purchase_month'].astype(int) #won't be able to plot a correct x-axis with this

#dropping non-needed columns

#df = df.drop(["product_name_lenght", "product_description_lenght", "product_photos_qty", "product_length_cm", "product_height_cm", "product_width_cm", "product_length_cm", "review_id","review_comment_title", "review_comment_message", "product_category_name"], axis=1)



# displaying missing value counts and corresponding percentage against total observations ===> BUG-ALERT: THE CODE CAUSING 2016 and 2017 DATA LOSS

#missing_values = df.isnull().sum().sort_values(ascending = False)

#percentage = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)

#pd.concat([missing_values, percentage], axis=1, keys=['Values', 'Percentage']).transpose()



# dropping missing values

#df.dropna(inplace=True)

#df.isnull().values.any()



# Creating new year-month column

#df_cus_count['month_year'] = df_cus_count['order_purchase_year'].astype(str) + '-' + df_cus_count['order_purchase_month'].astype(str)

#df_cus_count['order_purchase_month'] = df_cus_count['order_purchase_month'].astype(int)

#df_cus_st = df.groupby(['customer_state'], as_index=False).sum().loc[:, ['customer_state', 'payment_value']].sort_values(by='payment_value', ascending=False)

#df_cus_ct = df.groupby(['customer_city'], as_index=False).sum().loc[:, ['customer_city', 'payment_value']].sort_values(by='payment_value', ascending=False).head(20)



# Creating new year-month column

#df_ytsales['month_year'] = df_ytsales['order_purchase_year'].astype(str) + '-' + df_ytsales['order_purchase_month'].astype(str)

#df_ytsales['order_purchase_month'] = df_ytsales['order_purchase_month'].astype(int)

#df_ytsales['month_year_1'] = pd.to_datetime(df[['order_purchase_year', 'order_purchase_month']].assign(DAY=1)).dt.strftime('%Y-%m')

#df_ytsales.rename(columns={'order_purchase_year':'year'})

#df_ytsales.rename(columns={'order_purchase_month':'month'})

#df_ytsales['month_year_1'] = pd.to_datetime(df[['order_purchase_year', 'order_purchase_month']].assign(DAY=1))

#df_ytsales['month_year_1'] = pd.to_datetime(df_ytsales[['year', 'month']].assign(DAY=1))
# displaying first 3 rows of master dataframe

df.head(3)
#Debug code - to delete

print(df.shape)

df_2016 = df.query('order_purchase_year=="2016"')

print(df_2016.shape)
# Creating new datasets for each year

df_2016 = df.query('order_purchase_year=="2016"')

df_2017 = df.query('order_purchase_year=="2017"')

df_2018 = df.query('order_purchase_year=="2018"')



fig, axs = plt.subplots(1, 3, figsize=(22, 5))

count_plot(feature='order_purchase_year', df=df, ax=axs[0], title='Total Order Purchase by Year')

count_plot(feature='order_purchase_year', df=df_2017, ax=axs[1], hue='order_purchase_month', title='Number of Orders by Month in 2017')

count_plot(feature='order_purchase_year', df=df_2018, ax=axs[2], hue='order_purchase_month', title='Number of Orders by Month in 2018')





#count_plot(feature='order_purchase_year', df=df, ax=axs[2], hue='order_purchase_dayofweek', title='Total Yearly order Purchase by Day of the Week')

#format_spines(ax, right_border=False)

plt.suptitle('Score Counting Through the Years', y=1.1)

plt.show()
#df_ytsales_ym = df.groupby(['order_purchase_year', 'order_purchase_month','year_month'], as_index=False).sum().loc[:, ['order_purchase_year', 'order_purchase_month','year_month', 'payment_value','order_count']]

#df_ytsales_ym.head(3)
# Creating new datasets for each year

df_2016 = df.query('order_purchase_year=="2016"')

df_2017 = df.query('order_purchase_year=="2017"')

df_2018 = df.query('order_purchase_year=="2018"')



df_ytsales = df.groupby(['order_purchase_year', 'order_purchase_month','year_month'], as_index=False).sum().loc[:, ['order_purchase_year', 'order_purchase_month','year_month', 'payment_value','order_count']]

#df_ytsales = df.groupby(['order_purchase_year', 'order_purchase_month'], as_index=False).sum().loc[:, ['order_purchase_year', 'order_purchase_month', 'payment_value','order_count']]



df_ytsales_2016 = df_2016.groupby(['order_purchase_year', 'order_purchase_month'], as_index=False).sum().loc[:, ['order_purchase_year', 'order_purchase_month', 'payment_value']]

df_ytsales_2017 = df_2017.groupby(['order_purchase_year', 'order_purchase_month'], as_index=False).sum().loc[:, ['order_purchase_year', 'order_purchase_month', 'payment_value']]

df_ytsales_2018 = df_2018.groupby(['order_purchase_year', 'order_purchase_month'], as_index=False).sum().loc[:, ['order_purchase_year', 'order_purchase_month', 'payment_value']]



fig, axs = plt.subplots(1, 3, figsize=(22, 5))

bar_plot(x='order_purchase_year', y='payment_value', df=df_ytsales, ax=axs[0], value=True)

bar_plot(x='order_purchase_month', y='payment_value', df=df_ytsales_2017, ax=axs[1], value=True)

bar_plot(x='order_purchase_month', y='payment_value', df=df_ytsales_2018, ax=axs[2], value=True)



axs[0].set_title('Monthly Sales in 2016 to 2018')

axs[1].set_title('Monthly Sales in 2017')

axs[2].set_title('Monthly Sales in 2018', pad=10)



plt.suptitle('Order Payment Value Through the Years', y=1.1)

plt.show()
fig, ax = plt.subplots(figsize=(20, 4.5))

ax = sns.lineplot(x='year_month', y='payment_value', data=df_ytsales)

bar_plot(x='year_month', y='payment_value', df=df_ytsales, value=True)

format_spines(ax, right_border=False)

ax.set_title('Brazilian E-Commerce Monthly Sales from 2016 to 2018')
fig, ax = plt.subplots(figsize=(20, 4.5))

ax = sns.lineplot(x='year_month', y='order_count', data=df_ytsales)

bar_plot(x='year_month', y='order_count', df=df_ytsales, value=True)

format_spines(ax, right_border=False)

ax.set_title('Brazilian E-Commerce Monthly Order Volume from 2016 to 2018')
# Grouping by customer state

df_cus_count = df.groupby(['order_purchase_year', 'order_purchase_month','year_month'], as_index=False).nunique().loc[:, ['order_purchase_year', 'order_purchase_month','year_month', 'customer_unique_id','seller_id']]

df_cus_count.head(10)
#df_ytsales.head(10)
fig, ax = plt.subplots(figsize=(20, 4.5))

ax = sns.lineplot(x='year_month', y='customer_unique_id', data=df_cus_count)

bar_plot(x='year_month', y='customer_unique_id', df=df_cus_count, value=True)

format_spines(ax, right_border=False)

ax.set_title('Brazilian E-Commerce Number of Customers from 2016 to 2018')

plt.show()
df_cus_state = df.groupby(['customer_state','order_purchase_year'], as_index=False).sum().loc[:, ['customer_state','order_purchase_year', 'payment_value']].sort_values(by='payment_value', ascending=False)

df_cus_state.head(10)
df_cus_state = df.groupby(['customer_state','year_month'], as_index=False).sum().loc[:, ['customer_state','year_month', 'payment_value']].sort_values(by='payment_value', ascending=False)

#df_top5_state = df_cus_state.query('customer_state=="2016"')


top5 = ['SP', 'RJ','MG','RS','PR']

df_top5_state = df_cus_state.loc[df_cus_state['customer_state'].isin(top5)]

df_top5_state.head(3)



fig, ax = plt.subplots(figsize=(20, 4.5))

for state in top5:

    ax = sns.lineplot(x='year_month', y='payment_value', data=df_top5_state[df_top5_state['customer_state']==state], label=state)

format_spines(ax, right_border=False)

ax.set_title('Sales from the top 5 states from 2016 to 2018')
top4_noSP = ['RJ','MG','RS','PR']

df_top4_noSP = df_cus_state.loc[df_cus_state['customer_state'].isin(top4_noSP)]

df_top4_noSP.head(3)



fig, ax = plt.subplots(figsize=(20, 4.5))

for state in top4_noSP:

    ax = sns.lineplot(x='year_month', y='payment_value', data=df_top4_noSP[df_top4_noSP['customer_state']==state], label=state)

format_spines(ax, right_border=False)

ax.set_title('Sales from the top 4 states from 2016 to 2018, excluding SP')
fig, ax = plt.subplots(figsize=(20, 4.5))

ax = sns.lineplot(x='year_month', y='seller_id', data=df_cus_count)

bar_plot(x='year_month', y='seller_id', df=df_cus_count, value=True)

format_spines(ax, right_border=False)

ax.set_title('Brazilian E-Commerce Number of Sellers from 2016 to 2018')

plt.show()
#Create a dataframe to count how many times a customer shop 

df_order = df.groupby(['order_id','year_month','order_purchase_year','customer_unique_id'], as_index=False).sum().loc[:, ['order_id','customer_unique_id','year_month','order_purchase_year', 'payment_value']].sort_values(by='year_month', ascending=True)

df_order['time_to_shop'] = 1

df_order['time_to_shop']=df_order.groupby(['customer_unique_id']).cumcount() + 1 #cumcount() starts at 0, add 1 so that it starts at 1

#print(df_order.shape)

df_order.head(10)



#df_order['new_order']=order_customer.groupby(['customer_unique_id']).cumcount() + 1

#indices = order_customer['new_order'] != 1

#order_customer.loc[indices,'new_order'] = 0



df_order_2016 = df_order[df_order['order_purchase_year']==2016]

df_order_2017 = df_order[df_order['order_purchase_year']==2017]

df_order_2018 = df_order[df_order['order_purchase_year']==2018]
df_count_cust = df_order.groupby(['customer_unique_id']).count().reset_index()

df_count_cust["order_count"] = df_count_cust["order_id"]

df_count_cust = df_count_cust.drop(["order_id", "year_month", "payment_value", "time_to_shop","order_purchase_year"], axis=1)

df_count_cust = df_count_cust.groupby(["order_count"]).count().reset_index().rename(columns={"customer_unique_id": "num_customer"})

df_count_cust["percentage_customer"] = 100.0 * df_count_cust["num_customer"] / df_count_cust["num_customer"].sum()

df_count_cust
df_count_cust= df_order_2016.groupby(['customer_unique_id']).count().reset_index()

df_count_cust["order_count"] = df_count_cust["order_id"]

df_count_cust = df_count_cust.drop(["order_id", "year_month", "payment_value", "time_to_shop"], axis=1)

df_count_cust = df_count_cust.groupby(["order_count"]).count().reset_index().rename(columns={"customer_unique_id": "num_customer"})

df_count_cust["percentage_customer"] = 100.0 * df_count_cust["num_customer"] / df_count_cust["num_customer"].sum()

df_count_cust
df_count_cust= df_order_2017.groupby(['customer_unique_id']).count().reset_index()

df_count_cust["order_count"] = df_count_cust["order_id"]

df_count_cust = df_count_cust.drop(["order_id", "year_month", "payment_value", "time_to_shop"], axis=1)

df_count_cust = df_count_cust.groupby(["order_count"]).count().reset_index().rename(columns={"customer_unique_id": "num_customer"})

df_count_cust["percentage_customer"] = 100.0 * df_count_cust["num_customer"] / df_count_cust["num_customer"].sum()

df_count_cust
df_count_cust= df_order_2018.groupby(['customer_unique_id']).count().reset_index()

df_count_cust["order_count"] = df_count_cust["order_id"]

df_count_cust = df_count_cust.drop(["order_id", "year_month", "payment_value", "time_to_shop", 'order_purchase_year'], axis=1)

df_count_cust = df_count_cust.groupby(["order_count"]).count().reset_index().rename(columns={"customer_unique_id": "num_customer"})

df_count_cust["percentage_customer"] = 100.0 * df_count_cust["num_customer"] / df_count_cust["num_customer"].sum()

df_count_cust
df_quality = df.groupby(['order_purchase_year','year_month'], as_index=False).mean().loc[:, ['order_purchase_year','year_month','expected_duration','ship_duration', 'tocarrier_duration', 'lastmile_duration','expected_vs_shipdate','review_score']]

df_quality.head(10)
df_quality = df.groupby(['order_purchase_year'], as_index=False).mean().loc[:, ['order_purchase_year','expected_duration','ship_duration', 'tocarrier_duration', 'lastmile_duration','expected_vs_shipdate','review_score']]

df_quality.head(10)
fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(20,10),dpi=120)



df['expected_duration'].plot.hist(bins=30, alpha = 1,ax = axes[0,0])

axes[0,0].set_title('Expected Ship Duration (days)')



df['ship_duration'].plot.hist(bins=30, alpha = 1,ax = axes[1,0])

axes[1,0].set_title('End-to-End Ship Duration (days)')



df['tocarrier_duration'].plot.hist(bins=30, alpha = 1,ax = axes[0,1])

axes[0,1].set_title('Middle mile lead-time: from retailers to carriers (days)')



df['lastmile_duration'].plot.hist(bins=30, alpha = 1,ax = axes[1,1])

axes[1,1].set_title('Last mile lead-time: from carriers to customers (days)')
#drop outliers to make the histograms clearer

df_quality_chart_1 = df[df.expected_duration < 50] #drop any expected duration more than 60 days from purchase date

df_quality_chart_2 = df[df.ship_duration < 50] #drop any end-to-end ship duration more than 60 days from purchase date

df_quality_chart_3 = df[(df['tocarrier_duration'] < 30) & (df['tocarrier_duration'] > 0)] #drop any end-to-end ship duration more than 10 days from purchase date

df_quality_chart_4 = df[(df['lastmile_duration'] < 30) & (df['lastmile_duration'] > 0)] #drop any end-to-end ship duration more than 60 days from purchase date

df_quality_chart_5 = df[(df['expected_vs_shipdate'] < 30) & (df['expected_vs_shipdate'] > -30)] #drop any difference beyond 50 days btw expected ship date and actual ship date



fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(20,10),dpi=120)



df_quality_chart_1['expected_duration'].plot.hist(bins=30, alpha = 1,ax = axes[0,0])

axes[0,0].set_title('Expected Ship Duration 2016-2018 (days)')



df_quality_chart_2['ship_duration'].plot.hist(bins=30, alpha = 1,ax = axes[1,0])

axes[1,0].set_title('End-to-End Ship Duration 2016-2018 (days)')



df_quality_chart_3['tocarrier_duration'].plot.hist(bins=20, alpha = 0.5,ax = axes[0,1])

axes[0,1].set_title('Middle mile lead-time: from retailers to carriers 2016-2018 (days)')



df_quality_chart_4['lastmile_duration'].plot.hist(bins=20, alpha = 0.5,ax = axes[1,1])

axes[1,1].set_title('Last mile lead-time: from carriers to customers 2016-2018 (days)')



#df_quality_chart['lastmile_duration'].plot.hist(bins=30, alpha = 1)

df_quality_chart_5['expected_vs_shipdate'].plot.hist(bins=15, alpha = 1)

plt.title("Difference between expected ship date and delivered date")
df['review_score'].plot.hist(bins=10, alpha = 1)

plt.title("Review Score 2016-2018")