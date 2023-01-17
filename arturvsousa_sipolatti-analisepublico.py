import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from mpl_toolkits.basemap import Basemap

%matplotlib inline  
products = pd.read_csv('../input/olist-compras-20162018/olist_products_dataset.csv')
products.head()
reviews = pd.read_csv('../input/olist-compras-20162018/olist_order_reviews_dataset.csv')
reviews_clean = reviews.dropna()

reviews_clean.head()
clientes = pd.read_csv('../input/olist-compras-20162018/olist_customers_dataset.csv')
clientes.head()
clientesES = clientes['customer_state'] == 'ES'
clientesES = clientes[clientesES]
clientesES
cidades_compras = clientesES['customer_city'].value_counts().head(15)
cidades_compras.plot(kind='barh')

plt.show()
pedidos = pd.read_csv('../input/olist-compras-20162018/olist_orders_dataset.csv')
pedidosES = pd.merge(on='customer_id', how='inner', left=clientesES, right=pedidos)
pedidosES
reviewsES = pd.merge(on='order_id', left=pedidosES, right=reviews_clean)
reviewsES
itens = pd.read_csv('../input/olist-compras-20162018/olist_order_items_dataset.csv')



reviews_produtos = pd.merge(left=reviewsES, right=itens, on='order_id')
reviews_produtos
reviews_produtos['price'].describe()
pivot_gasto_medio = pd.pivot_table(reviews_produtos, values='price', index='customer_city')
maiores_gastos_cidade = pivot_gasto_medio.sort_values(by='price', ascending=False).head(10)

maiores_gastos_cidade.plot(kind='barh')
maior_gasto = reviews_produtos[reviews_produtos['price'] > 181]
maior_gasto = maior_gasto.drop_duplicates(subset='order_id')

maior_gasto.head()
maiorgasto_produtos = pd.merge(left=maior_gasto, right=products, on='product_id')
maiorgasto_produtos = maiorgasto_produtos[['customer_id','price', 'freight_value', 'product_category_name', 'customer_city', 'review_comment_title', 'review_comment_message']]
maiorgasto_produtos.sort_values(by='price', ascending=False)
reviews_produtos_negativ = reviews_produtos[reviews_produtos['review_comment_message'].str.contains('errado') | reviews_produtos['review_comment_message'].str.contains('Nunca mais') | reviews_produtos['review_comment_message'].str.contains('não condiz') | reviews_produtos['review_comment_message'].str.contains('ruim') | reviews_produtos['review_comment_message'].str.contains('não recomendo') | reviews_produtos['review_comment_message'].str.contains('não recomendado') | reviews_produtos['review_comment_message'].str.contains('péssimo')]



reviews_produtos_negativ = reviews_produtos_negativ[['customer_city', 'order_id', 'review_comment_title', 'review_comment_message', 'seller_id', 'price']]



reviews_produtos_negativ
compras_cidades = reviews_produtos['customer_city'].value_counts().head(10)
compras_cidades.plot(kind='barh')
rev_pro = reviews_produtos[reviews_produtos['price']<2000]

sns.barplot(x=rev_pro['review_score'], y=rev_pro['price'], ci=None)
produtos_cats = pd.merge(left=reviews_produtos, right=pedidos, on='order_id')

produtos_cats = pd.merge(left=produtos_cats, right=products, on='product_id')



prod_cat_piv = pd.pivot_table(produtos_cats, values='review_score', index='product_category_name')
prod_cat_piv.sort_values(ascending=False, by='review_score')
pedidosES.info()
products.info()
itens.info()
produtos = pd.merge(left=products, right=itens, on='product_id')
produtos.info()
pedidosESok = pd.merge(left=pedidosES, right=produtos, on='order_id')
pedidosESok.info()
pedidosESok = pedidosESok[['customer_id', 'order_id', 'customer_city', 'product_category_name', 'price', 'customer_zip_code_prefix']]
pedidosESok.info()
vitoria = pedidosESok[pedidosESok['customer_city'] == 'vitoria']
vitoria.info()
vitoria['product_category_name'].value_counts().head()
vv = pedidosESok[pedidosESok['customer_city'] == 'vila velha']

serra = pedidosESok[pedidosESok['customer_city'] == 'serra']

cariacica = pedidosESok[pedidosESok['customer_city'] == 'cariacica']
vv['product_category_name'].value_counts().head()
serra['product_category_name'].value_counts().head()
cariacica['product_category_name'].value_counts().head()
geoloc = pd.read_csv('../input/olist-compras-20162018/olist_geolocation_dataset.csv')
geoloc.info()
ES_geoloc = geoloc[geoloc['geolocation_zip_code_prefix'].astype(str).str.contains('29...')]
ES_geoloc
cariacica_geo = ES_geoloc[ES_geoloc['geolocation_city'] == 'cariacica']
cariacica_geo
cariacica.info()
cariacica_geo = cariacica_geo.rename(columns={'geolocation_zip_code_prefix': 'customer_zip_code_prefix'})
cariacica_geo
cariacica_compras_loc = pd.merge(left=cariacica, right=cariacica_geo, on='customer_zip_code_prefix')
cariacica_compras_loc.info()
#fig, ax = plt.subplots(figsize=(15,20))

#m = Basemap(projection='merc', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180)

#m.drawcoastlines()
cariacica_compras_loc = cariacica_compras_loc.drop_duplicates(subset='order_id')
cariacica_compras_loc['customer_zip_code_prefix'].value_counts()