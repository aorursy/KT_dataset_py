# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        

import seaborn as sns

import matplotlib.pyplot as plt



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
dt = pd.read_excel('/kaggle/input/sample-order-shopee/DATA TES.xlsx')
dt['Outbound Time'] = pd.to_datetime(dt['Outbound Time'], errors='coerce')

orders = dt[dt.columns[:30]].set_index(['Carton No','Shopee Tracking'])

order_product = dt.set_index(

    ['Carton No','Shopee Tracking','Receiver Telephone','Outbound Time'])[dt.columns[30:]]
dt.describe()
orders.reset_index().groupby('Carton No')['LM Tracking'].count().describe()
orders.reset_index().groupby('Carton No')['LM Tracking'].count().plot.hist(bins=17,legend=True)
orders.reset_index().groupby('Carton No')['Carton Volume'].min().describe()
orders.reset_index().groupby('Carton No')['Carton Volume'].min().plot.hist(bins=10,legend=True)
orders.reset_index().groupby('Carton No')['Carton Weight(KG)'].min().describe()
orders.reset_index().groupby('Carton No')['Carton Weight(KG)'].min().plot.hist(bins=10,legend=True)
orders['Sender Country'].value_counts()
orders['Sender Province'].value_counts()
orders['Sender City'].value_counts()

orders['Receiver Province/State'].value_counts()
orders['Receiver City'].value_counts()
import re 



def rename_order_product_column(string):

    return re.sub('\s\d+', '', string) 



order_product = pd.concat([

    order_product[order_product.columns[i:i+5]].rename(rename_order_product_column, axis=1) 

     for i in range(0,len(order_product.columns),5)

]).dropna()



order_product['Total Value'] = order_product['Declared Value'] * order_product['Declared QTY']
order_product.describe()[['Total Value','Declared QTY']].style.format("{:,.2f}")
order_product.reset_index()[

    ['Declared QTY','Total Value','Shopee Tracking']

].describe().style.format("{:,.2f}")
order_product.reset_index().groupby('Shopee Tracking').agg({

    'Product Name': len,

    'Declared QTY': sum,

    'Total Value': sum

}).describe().rename(columns={

    'Product Name': 'product',

    'Declared QTY': 'quantity',

    'Total Value':  'value'

}).style.format("{:,.2f}")
order_product.reset_index().groupby('Product Name').agg({

    'Shopee Tracking': len,

    'Declared QTY': sum,

    'Total Value': sum}).sort_values(['Shopee Tracking'], ascending=False)[:10].rename(columns={

    'Shopee Tracking': 'order',

    'Declared QTY': 'qty',

    'Total Value': 'value'

})
order_product.reset_index().groupby('Product Name').agg({

    'Shopee Tracking': len,

    'Declared QTY': sum,

    'Total Value': sum}).sort_values(['Declared QTY'], ascending=False)[:10].rename(columns={

    'Shopee Tracking': 'order',

    'Declared QTY': 'qty',

    'Total Value': 'value'

})
order_product.reset_index().groupby('Product Name').agg({

    'Shopee Tracking': len,

    'Declared QTY': sum,

    'Total Value': sum}).sort_values(['Total Value'], ascending=False)[:10].rename(columns={

    'Shopee Tracking': 'order',

    'Declared QTY': 'qty',

    'Total Value': 'value'

})
order_product.reset_index().groupby('HS CODE').agg({

    'Shopee Tracking': len,

    'Declared QTY': sum,

    'Total Value': sum}).sort_values(['Shopee Tracking'], ascending=False).head(10).rename(columns={

    'Shopee Tracking': 'order',

    'Declared QTY': 'qty',

    'Total Value': 'value'

})
order_product.reset_index().groupby('Declared Name').agg({

    'Shopee Tracking': len,

    'Declared QTY': sum,

    'Total Value': sum}).sort_values(['Declared QTY'], ascending=False)[:10].rename(columns={

    'Shopee Tracking': 'order',

    'Declared QTY': 'qty',

    'Total Value': 'value'

})
order_product.reset_index().groupby('Declared Name').agg({

    'Shopee Tracking': len,

    'Declared QTY': sum,

    'Total Value': sum}).sort_values(['Total Value'], ascending=False)[:10].rename(columns={

    'Shopee Tracking': 'order',

    'Declared QTY': 'qty',

    'Total Value': 'value'

})
order_product.reset_index().groupby('HS CODE').agg({

    'Shopee Tracking': len,

    'Declared QTY': sum,

    'Total Value': sum}).sort_values(['Shopee Tracking'], ascending=False).head(10).rename(columns={

    'Shopee Tracking': 'order',

    'Declared QTY': 'qty',

    'Total Value': 'value'

})
order_product.reset_index().groupby('HS CODE').agg({

    'Shopee Tracking': len,

    'Declared QTY': sum,

    'Total Value': sum}).sort_values(['Shopee Tracking'], ascending=False).head(10).rename(columns={

    'Shopee Tracking': 'order',

    'Declared QTY': 'qty',

    'Total Value': 'value'

})
order_product.reset_index().groupby('HS CODE').agg({

    'Shopee Tracking': len,

    'Declared QTY': sum,

    'Total Value': sum}).sort_values(['Total Value'], ascending=False).head(10).rename(columns={

    'Shopee Tracking': 'order',

    'Declared QTY': 'qty',

    'Total Value': 'value'

})
repeat_order = order_product.reset_index().groupby('Product Name')[[

    'Receiver Telephone','Shopee Tracking']].nunique().rename(columns={

    'Receiver Telephone':'Customer Count',

    'Shopee Tracking': 'Order Count'

})



repeat_order['Rasio'] = repeat_order['Order Count'] / repeat_order['Customer Count']

repeat_order.sort_values('Rasio', ascending=False).head(10)
daily_order_product = order_product.reset_index().groupby(['Product Name', pd.Grouper(key='Outbound Time', freq='D')]).agg({

    'Shopee Tracking': len    

}).rename(columns={'Shopee Tracking':'Order Count'}).sort_values(by=['Order Count'])
top_daily_product = daily_order_product.reset_index(-1)[

    daily_order_product.reset_index(-1).index.isin(top_product)]
plot = top_daily_product.loc['[HOT SALE] Kids PM2.5 Cotton Mouth Mask with Breath Valve Filter Papers Children Anti-Dust Anti Pollution Mask Cloth Activated C']

plot['Outbound Time'] = plot['Outbound Time'].apply(lambda x: x.strftime('%Y-%m-%d'))

plot.set_index('Outbound Time')

fig, ax = plt.subplots(figsize=(12, 5))

g = sns.lineplot(data=plot.set_index('Outbound Time'))

plt.show()
daily_order_category = order_product.reset_index().groupby(['Declared Name', pd.Grouper(key='Outbound Time', freq='D')]).agg({

    'Shopee Tracking': len    

}).rename(columns={'Shopee Tracking':'Order Count'}).sort_values(by=['Order Count'])
plot = daily_order_category.loc['Mobile & Accessories-Casing & Covers'].reset_index()

plot['Outbound Time'] = plot['Outbound Time'].apply(lambda x: x.strftime('%Y-%m-%d'))

plot.set_index('Outbound Time')

fig, ax = plt.subplots(figsize=(12, 5))

g = sns.lineplot(data=plot.set_index('Outbound Time'))

plt.show()