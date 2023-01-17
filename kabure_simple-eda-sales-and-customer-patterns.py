import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from scipy import stats 

import os

import matplotlib.pyplot as plt

import seaborn as sns 
df_item = pd.read_csv("../input/olist_order_items_dataset.csv")

df_reviews = pd.read_csv("../input/olist_order_reviews_dataset.csv")

df_orders = pd.read_csv("../input/olist_orders_dataset.csv")

df_products = pd.read_csv("../input/olist_products_dataset.csv")

df_geolocation = pd.read_csv("../input/olist_geolocation_dataset.csv")

df_sellers = pd.read_csv("../input/olist_sellers_dataset.csv")

df_order_pay = pd.read_csv("../input/olist_order_payments_dataset.csv")

df_customers = pd.read_csv("../input/olist_customers_dataset.csv")

df_category = pd.read_csv("../input/product_category_name_translation.csv")
df_train = df_orders.merge(df_item, on='order_id', how='left')

df_train = df_train.merge(df_order_pay, on='order_id', how='outer', validate='m:m')

df_train = df_train.merge(df_reviews, on='order_id', how='outer')

df_train = df_train.merge(df_products, on='product_id', how='outer')

df_train = df_train.merge(df_customers, on='customer_id', how='outer')

df_train = df_train.merge(df_sellers, on='seller_id', how='outer')



print(df_train.shape)
def resumetable(df):

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values



    for name in summary['Name'].value_counts().index:

        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 



    return summary



def cross_heatmap(df, cols, normalize=False, values=None, aggfunc=None):

    temp = cols

    cm = sns.light_palette("green", as_cmap=True)

    return pd.crosstab(df[temp[0]], df[temp[1]], 

                       normalize=normalize, values=values, aggfunc=aggfunc).style.background_gradient(cmap = cm)
resumetable(df_train)
id_cols = ['order_id', 'seller_id', 'customer_id', 'order_item_id', 'product_id', 

           'review_id', 'customer_unique_id', 'seller_zip_code_prefix']

#categorical columns

cat_cols = df_train.nunique()[df_train.nunique() <= 27].keys().tolist()

# cat_cols = [x for x in cat_cols if x not in target_col]



#numerical columns

num_cols = [x for x in df_train.columns if x not in cat_cols + id_cols]



#Binary columns with 2 values

bin_cols = df_train.nunique()[df_train.nunique() == 2].keys().tolist()



#Columns more than 2 values

multi_cols = [i for i in cat_cols if i not in bin_cols]
df_train['price'].fillna(-1, inplace=True)



plt.figure(figsize=(16,12))

plt.suptitle('Price Distributions', fontsize=22)

plt.subplot(221)

g = sns.distplot(df_train['price'])

g.set_title("Price Distributions", fontsize=18)

g.set_xlabel("Price Values")

g.set_ylabel("Probability", fontsize=15)



plt.subplot(222)

g1 = sns.distplot(np.log(df_train['price']+1.5))

g1.set_title("Price(LOG) Distributions", fontsize=18)

g1.set_xlabel("Price Values")

g1.set_ylabel("Probability", fontsize=15)



plt.subplot(212)

g4 = plt.scatter(range(df_train.shape[0]),

                 np.sort(df_train['price'].values), 

                 alpha=.1)

g4= plt.title("ECDF of Prices", fontsize=18)

g4 = plt.xlabel("Index")

g4 = plt.ylabel("Price Distribution", fontsize=15)

g4 = plt.axhline(df_train[df_train['price'] != -1]['price'].mean(), color='black', 

           label='Mean Price', linewidth=2)

g4 = plt.axhline(df_train[df_train['price'] != -1]['price'].mean() + (2.5*df_train[df_train['price'] != -1]['price'].std()),

                 color='red', 

           label='Mean + 2*Stdev', linewidth=2)

g4 = plt.legend()



plt.subplots_adjust(hspace = 0.4, top = 0.85)



plt.show()
df_train['price_log'] = np.log(df_train['price'] + 1.5)
total = len(df_train)



plt.figure(figsize=(14,6))



plt.suptitle('Payment Type Distributions', fontsize=22)



plt.subplot(121)

g = sns.countplot(x='payment_type', data=df_train[df_train['payment_type'] != 'not_defined'])

g.set_title("Payment Type Count Distribution", fontsize=20)

g.set_xlabel("Payment Type Name", fontsize=17)

g.set_ylabel("Count", fontsize=17)



sizes = []

for p in g.patches:

    height = p.get_height()

    sizes.append(height)

    g.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=14) 

    

g.set_ylim(0, max(sizes) * 1.1)



plt.subplot(122)

g = sns.boxplot(x='payment_type', y='price_log', data=df_train[df_train['payment_type'] != 'not_defined'])

g.set_title("Payment Type by Price Distributions", fontsize=20)

g.set_xlabel("Payment Type Name", fontsize=17)

g.set_ylabel("Price(Log)", fontsize=17)



plt.subplots_adjust(hspace = 0.5, top = 0.8)



plt.show()
plt.figure(figsize=(16,12))



plt.suptitle('CUSTOMER State Distributions', fontsize=22)



plt.subplot(212)

g = sns.countplot(x='customer_state', data=df_train, orient='h')

g.set_title("Customer's State Distribution", fontsize=20)

g.set_xlabel("State Name Short", fontsize=17)

g.set_ylabel("Count", fontsize=17)

g.set_xticklabels(g.get_xticklabels(),rotation=45)

sizes = []

for p in g.patches:

    height = p.get_height()

    sizes.append(height)

    g.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=12) 

g.set_ylim(0, max(sizes) * 1.1)



plt.subplot(221)

g2 = sns.boxplot(x='customer_state', y='price_log', 

                 data=df_train[df_train['price'] != -1])

g2.set_title("Customer's State by Price", fontsize=20)

g2.set_xlabel("State Name Short", fontsize=17)

g2.set_ylabel("Price(Log)", fontsize=17)

g2.set_xticklabels(g2.get_xticklabels(),rotation=45)



plt.subplot(222)

g3 = sns.boxplot(x='customer_state', y='freight_value', 

                 data=df_train[df_train['price'] != -1])

g3.set_title("CUSTOMER's State by Freight Value", fontsize=20)

g3.set_xlabel("State Name Short", fontsize=17)

g3.set_ylabel("Freight Value", fontsize=17)

g3.set_xticklabels(g3.get_xticklabels(),rotation=45)



plt.subplots_adjust(hspace = 0.5, top = 0.9)



plt.show()
plt.figure(figsize=(16,12))



plt.suptitle('SELLER State Distributions', fontsize=22)



plt.subplot(212)

g = sns.countplot(x='seller_state', data=df_train, orient='h')

g.set_title("Seller's State Distribution", fontsize=20)

g.set_xlabel("State Name Short", fontsize=17)

g.set_ylabel("Count", fontsize=17)

g.set_xticklabels(g.get_xticklabels(),rotation=45)

sizes = []

for p in g.patches:

    height = p.get_height()

    sizes.append(height)

    g.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=12) 

g.set_ylim(0, max(sizes) * 1.1)



plt.subplot(221)

g2 = sns.boxplot(x='seller_state', y='price_log', 

                 data=df_train[df_train['price'] != -1])

g2.set_title("Seller's State by Price", fontsize=20)

g2.set_xlabel("State Name Short", fontsize=17)

g2.set_ylabel("Price(Log)", fontsize=17)

g2.set_xticklabels(g2.get_xticklabels(),rotation=45)



plt.subplot(222)

g3 = sns.boxplot(x='seller_state', y='freight_value', 

                 data=df_train[df_train['price'] != -1])

g3.set_title("Seller's State by Freight Value", fontsize=20)

g3.set_xlabel("State Name Short", fontsize=17)

g3.set_ylabel("Freight Value", fontsize=17)

g3.set_xticklabels(g3.get_xticklabels(),rotation=45)



plt.subplots_adjust(hspace = 0.5, top = 0.9)



plt.show()
# Seting regions

sudeste = ['SP', 'RJ', 'ES','MG']

nordeste= ['MA', 'PI', 'CE', 'RN', 'PE', 'PB', 'SE', 'AL', 'BA']

norte =  ['AM', 'RR', 'AP', 'PA', 'TO', 'RO', 'AC']

centro_oeste = ['MT', 'GO', 'MS' ,'DF' ]

sul = ['SC', 'RS', 'PR']



df_train.loc[df_train['customer_state'].isin(sudeste), 'cust_Region'] = 'Southeast'

df_train.loc[df_train['customer_state'].isin(nordeste), 'cust_Region'] = 'Northeast'

df_train.loc[df_train['customer_state'].isin(norte), 'cust_Region'] = 'North'

df_train.loc[df_train['customer_state'].isin(centro_oeste), 'cust_Region'] = 'Midwest'

df_train.loc[df_train['customer_state'].isin(sul), 'cust_Region'] = 'South'
cross_heatmap(df_train[df_train['price'] != -1], ['seller_state', 'cust_Region'], 

              values=df_train[df_train['price'] != -1]['freight_value'], aggfunc='mean')
df_train['ord_new'] = df_train['order_item_id'].copy()



df_train.loc[df_train['order_item_id'].isin([7,8,9,10]), 'ord_new'] = '7 to 10'

df_train.loc[(df_train['order_item_id'] > 10), 'ord_new'] = '10 to 20'
plt.figure(figsize=(14,10))





plt.subplot(211)

g = sns.countplot(x='ord_new', data=df_train)

g.set_title("Order Item Id Distribution", fontsize=20)

g.set_xlabel("State Name Short", fontsize=17)

g.set_ylabel("Count", fontsize=17)

sizes = []

for p in g.patches:

    height = p.get_height()

    sizes.append(height)

    g.text(p.get_x()+p.get_width()/2.,

            height + 3,

            '{:1.2f}%'.format(height/total*100),

            ha="center", fontsize=12) 

g.set_ylim(0, max(sizes) * 1.1)



plt.subplot(212)

g1 = sns.scatterplot(x='order_item_id', y='price_log',

                     data=df_train, alpha=.2)

g1.set_title("Seller's State Distribution", fontsize=20)

g1.set_xlabel("State Name Short", fontsize=17)

g1.set_ylabel("Count", fontsize=17)



plt.subplots_adjust(hspace = 0.5, top = 0.9)



plt.show()
round(pd.crosstab(df_train['order_item_id'], df_train['review_score'], normalize='index') *100,2)[:12].T