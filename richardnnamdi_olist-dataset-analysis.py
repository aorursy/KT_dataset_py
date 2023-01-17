#importing libraries

import numpy as np

from numpy import sort

import pandas as pd

import gc

from datetime import datetime

import time

import warnings

import matplotlib.pyplot as plt

from matplotlib import style



import seaborn as sns

sns.set(style="ticks")

#sns.set_context("poster", font_scale = .2, rc={"grid.linewidth": 2})

import sklearn

import scipy



import random

np.random.seed(42)

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

pd.set_option('display.max_columns', 100)

%matplotlib inline
#defining visualizaition functions

def format_spines(ax, right_border=True):

    

    ax.spines['bottom'].set_color('#666666')

    ax.spines['left'].set_color('#666666')

    ax.spines['top'].set_visible(False)

    if right_border:

        ax.spines['right'].set_color('#FFFFFF')

    else:

        ax.spines['right'].set_color('#FFFFFF')

    ax.patch.set_facecolor('#FFFFFF')

    



def count_plot(feature, df, colors='Blues_d', hue=False, ax=None, title=''):

    

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

        ax.annotate('{:.1f}%'.format(100.*y/ncount), (x.mean(), y), 

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

        



def individual_cat_pie_plot(col, ax, cs, df):

    

    # Creating figure and showing data

    names = df[col].value_counts().index

    heights = df[col].value_counts().values

    total = df[col].value_counts().sum()

    #if cs:

    #cs = cm.viridis(np.arange(len(names))/len(names))

    explode = np.zeros(len(names))

    explode[0] = 0.05

    wedges, texts, autotexts = ax.pie(heights, labels=names, explode=explode,

                                       startangle=90, shadow=False, 

                                      autopct='%1.1f%%', colors=cs[:len(names)])

    plt.setp(autotexts, size=12, color='w')

    



def donut_plot(col, ax, df, text='', colors=['navy', 'crimson', 'green', 'red', 'cyan'], labels=['good', 'bad', 'fair', 'bald', 'none']):

    

    sizes = df[col].value_counts().values

    #labels = df[col].value_counts().index

    center_circle = plt.Circle((0,0), 0.80, color='white')

    ax.pie((sizes[0], sizes[1], sizes[2], sizes[3], sizes[4]), labels=labels, colors=colors, autopct='%1.1f%%')

    ax.add_artist(center_circle)

    kwargs = dict(size=20, fontweight='bold', va='center')

    ax.text(0, 0, text, ha='center', **kwargs)

    

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

        



def individual_cat_pie_plot(col, ax, cs, df):

    

    # Creating figure and showing data

    names = df[col].value_counts().index

    heights = df[col].value_counts().values

    total = df[col].value_counts().sum()

    #if cs:

    #cs = cm.viridis(np.arange(len(names))/len(names))

    explode = np.zeros(len(names))

    explode[0] = 0.05

    wedges, texts, autotexts = ax.pie(heights, labels=names, explode=explode,

                                       startangle=90, shadow=False, 

                                      autopct='%1.1f%%', colors=cs[:len(names)])

    plt.setp(autotexts, size=12, color='w')

    



def donut_plot(col, ax, df, text='', colors=['navy', 'crimson'], labels=['good', 'bad']):

    

    sizes = df[col].value_counts().values

    #labels = df[col].value_counts().index

    center_circle = plt.Circle((0,0), 0.80, color='white')

    ax.pie((sizes[0], sizes[1]), labels=labels, colors=colors, autopct='%1.1f%%')

    ax.add_artist(center_circle)

    kwargs = dict(size=20, fontweight='bold', va='center')

    ax.text(0, 0, text, ha='center', **kwargs)
# loading data

customers_ = pd.read_csv("../input/brazilian-ecommerce/olist_customers_dataset.csv")

geolocation_ = pd.read_csv("../input/brazilian-ecommerce/olist_geolocation_dataset.csv")

order_items_ = pd.read_csv("../input/brazilian-ecommerce/olist_order_items_dataset.csv")

order_payments_ = pd.read_csv("../input/brazilian-ecommerce/olist_order_payments_dataset.csv")

order_reviews_ = pd.read_csv("../input/brazilian-ecommerce/olist_order_reviews_dataset.csv")

orders_ = pd.read_csv("../input/brazilian-ecommerce/olist_orders_dataset.csv")

products_ = pd.read_csv("../input/brazilian-ecommerce/olist_products_dataset.csv")

sellers_ = pd.read_csv("../input/brazilian-ecommerce/olist_sellers_dataset.csv")

category_name_translation_ = pd.read_csv("../input/brazilian-ecommerce/product_category_name_translation.csv")
# displaying data shape

#dataset = [customers, geolocation, order_items, order_payments, order_reviews, orders, products, sellers, category_name_translation]

dataset = {

    'Customers': customers_,

    'Geolocation': geolocation_,

    'Order Items': order_items_,

    'Payments': order_payments_,

    'Reviews': order_reviews_,

    'Orders': orders_,

    'Products': products_,

    'Sellers': sellers_,

    'Translations': category_name_translation_

}



for x, y in dataset.items():

    print(f'{x}', (list(y.shape)))
# displaying dataset column names

for x, y in dataset.items():

    print(f'{x}', f'{list(y.columns)}\n')
# checking for null values in datasets

for x, y in dataset.items():

    print(f'{x}: {y.isnull().any().any()}')
# taking count for dataset with missing values

for x, y in dataset.items():

    if y.isnull().any().any():

        print(f'{x}', (list(y.shape)),'\n')

        print(f'{y.isnull().sum()}\n')
# creating master dataframe 

order_payments_.head()

print(order_payments_.shape)

df1 = order_payments_.merge(order_items_, on='order_id')

print(df1.shape)

df2 = df1.merge(products_, on='product_id')

print(df2.shape)

df3 = df2.merge(sellers_, on='seller_id')

print(df3.shape)

df4 = df3.merge(order_reviews_, on='order_id')

print(df4.shape)

df5 = df4.merge(orders_, on='order_id')

print(df5.shape)

df6 = df5.merge(category_name_translation_, on='product_category_name')

print(df6.shape)

df = df6.merge(customers_, on='customer_id')

print(df.shape)
# converting date columns to datetime

date_columns = ['shipping_limit_date', 'review_creation_date', 'review_answer_timestamp', 'order_purchase_timestamp', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date']

for col in date_columns:

    df[col] = pd.to_datetime(df[col], format='%Y-%m-%d %H:%M:%S')
# cleaning up name columns, and engineering new/essential columns

df['customer_city'] = df['customer_city'].str.title()

df['seller_city'] = df['seller_city'].str.title()

df['product_category_name_english'] = df['product_category_name_english'].str.title()

df['payment_type'] = df['payment_type'].str.replace('_', ' ').str.title()

df['product_category_name_english'] = df['product_category_name_english'].str.replace('_', ' ')

df['review_response_time'] = (df['review_answer_timestamp'] - df['review_creation_date']).dt.days

df['delivery_against_estimated'] = (df['order_estimated_delivery_date'] - df['order_delivered_customer_date']).dt.days

df['product_size_cm'] = df['product_length_cm'] * df['product_height_cm'] * df['product_width_cm']

df['order_purchase_year'] = df.order_purchase_timestamp.apply(lambda x: x.year)

df['order_purchase_month'] = df.order_purchase_timestamp.apply(lambda x: x.month)

df['order_purchase_dayofweek'] = df.order_purchase_timestamp.apply(lambda x: x.dayofweek)

df['order_purchase_hour'] = df.order_purchase_timestamp.apply(lambda x: x.hour)

df['order_purchase_day'] = df['order_purchase_dayofweek'].map({0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'})

df['order_purchase_mon'] = df.order_purchase_timestamp.apply(lambda x: x.month).map({0:'Jan',1:'Feb',2:'Mar',3:'Apr',4:'May',5:'Jun',6:'Jul',7:'Aug',8:'Sep',9:'Oct',10:'Nov',11:'Dec'})
# dropping non-needed columns

df = df.drop(["product_name_lenght", "product_description_lenght", "product_photos_qty", "product_length_cm", "product_height_cm", "product_width_cm", "product_length_cm", "review_id","review_comment_title", "review_comment_message", "product_category_name"], axis=1)
# displaying summary staticstics of columns

df.describe(include='all')
# displaying missing value counts and corresponding percentage against total observations

missing_values = df.isnull().sum().sort_values(ascending = False)

percentage = (df.isnull().sum()/df.isnull().count()*100).sort_values(ascending = False)

pd.concat([missing_values, percentage], axis=1, keys=['Values', 'Percentage']).transpose()
# dropping missing values

df.dropna(inplace=True)

df.isnull().values.any()
# displaying dataframe info

df.info()
# displaying first 3 rows of master dataframe

df.head(3)
# Creating new datasets for each year

df_2016 = df.query('order_purchase_year=="2016"')

df_2017 = df.query('order_purchase_year=="2017"')

df_2018 = df.query('order_purchase_year=="2018"')



#displaying total orders in years, comparitive year on month, and month on days of the week

fig, axs = plt.subplots(1, 3, figsize=(22, 5))

count_plot(feature='order_purchase_year', df=df, ax=axs[0], title='Total Order Purchase by Year')

count_plot(feature='order_purchase_year', df=df, ax=axs[1], hue='order_purchase_month', title='Total Yearly order Purchase by Month')

count_plot(feature='order_purchase_year', df=df, ax=axs[2], hue='order_purchase_dayofweek', title='Total Yearly order Purchase by Day of the Week')

#format_spines(ax, right_border=False)

plt.suptitle('Score Counting Through the Years', y=1.1)

plt.show()
# Grouping by annual and monthly sales

df_ytsales = df.groupby(['order_purchase_year', 'order_purchase_month'], as_index=False).sum().loc[:, ['order_purchase_year', 'order_purchase_month', 'payment_value']]

df_ytsales2 = df.groupby(['order_purchase_year', 'order_purchase_dayofweek'], as_index=False).sum().loc[:, ['order_purchase_year', 'order_purchase_dayofweek', 'payment_value']]

#df_ytsales = df.groupby(['order_purchase_year', 'order_purchase_month', 'order_purchase_dayofweek'], as_index=False).sum().loc[:, ['order_purchase_year', 'order_purchase_month', 'order_purchase_dayofweek', 'payment_value']]



df_s2016 = df_ytsales[df_ytsales['order_purchase_year']==2016]

df_s2017 = df_ytsales[df_ytsales['order_purchase_year']==2017]

df_s2018 = df_ytsales[df_ytsales['order_purchase_year']==2018]



fig, axs = plt.subplots(1, 3, figsize=(22, 5))

bar_plot(x='order_purchase_month', y='payment_value', df=df_s2016, ax=axs[0], value=True)

bar_plot(x='order_purchase_month', y='payment_value', df=df_s2017, ax=axs[1], value=True)

bar_plot(x='order_purchase_month', y='payment_value', df=df_s2018, ax=axs[2], value=True)

axs[0].set_title('Monthly Sales in 2016')

axs[1].set_title('Monthly Sales in 2017')

axs[2].set_title('Monthly Sales in 2018', pad=10)

plt.xticks(np.arange(8), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug'])

plt.show()
fig, axs = plt.subplots(1, 3, figsize=(22, 5))

bar_plot(x='order_purchase_year', y='payment_value', df=df_ytsales, ax=axs[0], value=True)

bar_plot(x='order_purchase_month', y='payment_value', df=df_ytsales, ax=axs[1], value=True)

bar_plot(x='order_purchase_dayofweek', y='payment_value', df=df_ytsales2, ax=axs[2], value=True)

axs[0].set_title('Monthly Sales in 2016')

axs[1].set_title('Monthly Sales in 2017')

axs[2].set_title('Monthly Sales in 2018', pad=10)

plt.xticks(np.arange(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])



plt.show()
# Changing the month attribute for correct ordenation

df_ytsales['order_purchase_month'] = df_ytsales['order_purchase_month'].astype(str).apply(lambda x: '0' + x if len(x) == 1 else x)



# Creating new year-month column

df_ytsales['month_year'] = df_ytsales['order_purchase_year'].astype(str) + '-' + df_ytsales['order_purchase_month'].astype(str)

df_ytsales['order_purchase_month'] = df_ytsales['order_purchase_month'].astype(int)



# PLotting

fig, ax = plt.subplots(figsize=(14, 4.5))

ax = sns.lineplot(x='month_year', y='payment_value', data=df_ytsales.iloc[:-1, :])

format_spines(ax, right_border=False)

ax.tick_params(axis='x', labelrotation=90)

ax.set_title('Brazilian E-Commerce Sales Evolution')

plt.show()



fig, ax = plt.subplots(figsize=(14, 4.5))



ax = sns.lineplot(x='order_purchase_month', y='payment_value', data=df_s2016, label='2016')

ax = sns.lineplot(x='order_purchase_month', y='payment_value', data=df_s2017, label='2017')

ax = sns.lineplot(x='order_purchase_month', y='payment_value', data=df_s2018, label='2018')

format_spines(ax, right_border=False)

ax.set_title('Brazilian E-Commerce Sales Comparison')

plt.xticks(np.arange(13), ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])

plt.show()
#customer and eliverybehaviour



purchase_count = df.groupby(['order_purchase_day', 'order_purchase_hour']).count()['price'].unstack()

plt.figure(figsize=(22,7))

sns.heatmap(purchase_count.reindex(index = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']), cmap="YlGnBu", annot=True, fmt="d", linewidths=0.2)

plt.show()
df_ypt = df.groupby(['order_purchase_year', 'payment_type'], as_index=False).sum().loc[:, ['order_purchase_year', 'payment_type', 'payment_value']]

df_dpt = df.groupby(['order_purchase_year', 'payment_type'], as_index=False).mean().loc[:, ['order_purchase_year', 'payment_type', 'payment_value']]



#displaying total orders in years, comparitive year on month, and month on days of the week

fig, axs = plt.subplots(1, 3, figsize=(22, 5))

count_plot(feature='payment_type', df=df, ax=axs[0], title='Total Order Purchase by Year')

bar_plot(x='order_purchase_year', y='payment_value', ax=axs[1], hue='payment_type', df=df_ypt.sort_values(by='payment_value', ascending=False), value=True)

bar_plot(x='payment_type', y='payment_value', ax=axs[2], hue='order_purchase_year', df=df_dpt.sort_values(by='payment_value', ascending=False), value=True)



format_spines(ax, right_border=False)

plt.suptitle('Score Counting Through the Years', y=1.1)

plt.show()
df_ypt
df_dpt
# Grouping by customer state

df_cus_st = df.groupby(['customer_state'], as_index=False).sum().loc[:, ['customer_state', 'payment_value']].sort_values(by='payment_value', ascending=False)

df_cus_ct = df.groupby(['customer_city'], as_index=False).sum().loc[:, ['customer_city', 'payment_value']].sort_values(by='payment_value', ascending=False).head(20)



fig, axs = plt.subplots(1, 2, figsize=(22, 7))

bar_plot(x='payment_value', y='customer_state', df=df_cus_st, ax=axs[0], value=False)

bar_plot(x='payment_value', y='customer_city', df=df_cus_ct, ax=axs[1], value=False)

format_spines(ax, right_border=False)

axs[0].set_title('Monthly Sales in 2016')

axs[1].set_title('Monthly Sales in 2017')



plt.show()
df_cus_ct.sort_values(by='payment_value', ascending=False)
df_cus_st.head(20).sort_values(by='payment_value', ascending=False)
customer_value_sum = (df.groupby(['customer_unique_id'])[['payment_value', 'payment_installments','review_score']]

.agg({'payment_value':['count', 'mean', 'sum'], 'payment_installments': ['mean'], 'review_score': ['mean']})

                ).sort_values(by=('payment_value','sum'), ascending=False)



customer_value_sum.head(20)