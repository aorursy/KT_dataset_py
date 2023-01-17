import pandas as pd

pd.options.display.float_format = '{:.2f}'.format

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from matplotlib.pyplot import figure

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')

import folium

from folium import plugins

from sklearn.cluster import DBSCAN

print('Bibliotecas preparadas!')
files = {'customers'    : '/kaggle/input/brazilian-ecommerce/olist_customers_dataset.csv',

         'geolocation'  : '/kaggle/input/brazilian-ecommerce/olist_geolocation_dataset.csv',

         'items'        : '/kaggle/input/brazilian-ecommerce/olist_order_items_dataset.csv',

         'payment'      : '/kaggle/input/brazilian-ecommerce/olist_order_payments_dataset.csv',

         'orders'       : '/kaggle/input/brazilian-ecommerce/olist_orders_dataset.csv',

         'products'     : '/kaggle/input/brazilian-ecommerce/olist_products_dataset.csv',

         'sellers'      : '/kaggle/input/brazilian-ecommerce/olist_sellers_dataset.csv',

         'review'       : '/kaggle/input/brazilian-ecommerce/olist_order_reviews_dataset.csv',

         }



dfs = {}

for key, value in files.items():

    dfs[key] = pd.read_csv(value)
# Cruzamento gradativo

customers_location = dfs['customers'].merge(dfs['geolocation'], how='inner', left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix').drop_duplicates('customer_id', keep='first')

cusloc_order = customers_location.merge(dfs['orders'], how='inner', on='customer_id')

cuslocord_item = cusloc_order.merge(dfs['items'], how='inner', on='order_id')

cuslocordite_prod = cuslocord_item.merge(dfs['products'], how='inner', on='product_id')

cuslocordite_rev= cuslocordite_prod.merge(dfs['review'], how='left', on='order_id')



# Selecionando as colunas de interesse

final = cuslocordite_rev[['customer_id', 'customer_unique_id', 'customer_zip_code_prefix',

       'customer_city', 'customer_state',

       'geolocation_lat', 'geolocation_lng','order_id', 'order_status',

       'order_purchase_timestamp', 'order_approved_at',

       'order_delivered_carrier_date', 'order_delivered_customer_date',

       'order_estimated_delivery_date', 'order_item_id', 'product_id',

       'seller_id', 'shipping_limit_date', 'price', 'freight_value',

       'product_category_name', 'product_photos_qty',

       'review_id', 'review_score', 'review_comment_title',

       'review_comment_message', 'review_creation_date',

       'review_answer_timestamp']]



# Convertendo para datetime

datas = ['order_purchase_timestamp'

        ,'order_purchase_timestamp'

        ,'order_delivered_carrier_date'

        ,'order_delivered_customer_date'

        ,'order_estimated_delivery_date'

        ,'shipping_limit_date'

        ,'review_creation_date'

        ,'review_answer_timestamp' 

        ]



for data in datas:

    final[data] = pd.to_datetime(final[data])



# Criando coluna de tempo de entrega e Hora da compra

final['delivery_time'] = (final['order_delivered_customer_date'].dt.date - final['order_purchase_timestamp'].dt.date).dt.days



# Seleção do período de interesse

final = final[(final['order_purchase_timestamp'].dt.year > 2016) 

              & 

              (final['order_purchase_timestamp'] < pd.to_datetime('20180901'))

             ]

final = final.reset_index(drop=True)

final.info()
meses_compras = pd.DataFrame()

meses_compras['mes'] = final['order_purchase_timestamp'].dt.month

meses_compras['ano'] = final['order_purchase_timestamp'].dt.year

meses_compras['count'] = final['customer_id']

meses_compras = meses_compras.groupby(['ano','mes'])['count'].count().reset_index()

meses_compras['ano_mes'] = meses_compras['ano'].astype(str) + ', ' + meses_compras['mes'].astype(str)



meses_compras.plot(x='ano_mes', y='count', figsize=(25,8))#, color='#42A5F5', alpha=0.9, ci=None)

plt.xlabel('Ano, Mês', size=20)

plt.ylabel('Qtd. de Pedidos', size=20)



print(r'Gráfico I')

plt.show()
estado_compras = final.groupby('customer_state', as_index=False)['price'].sum().sort_values(by='customer_state')

estado_compras_med = final.groupby('customer_state', as_index=False)['price'].mean().sort_values(by='customer_state')

frete_medio = final.groupby('customer_state', as_index=False)['freight_value'].mean().sort_values(by='customer_state')



print('Tabela 1')

estado_compras_med['price'].describe()
figure(num=None, figsize=(12, 8), dpi=80)



plt.subplot(2, 1, 1)

sns.barplot(x=estado_compras['customer_state'], y=estado_compras['price'], color='#42A5F5', alpha=0.9)

plt.xlabel(None)

plt.ylabel('Volume de Compras')



plt.subplot(2, 1, 2)

sns.lineplot(x=frete_medio['customer_state'], y=frete_medio['freight_value'], color='#28B463', alpha=0.9)

#ylim(top=3)  # adjust the top leaving bottom unchanged

plt.ylim(0,50)

plt.xlabel('Estado')

plt.ylabel('Frete Médio')



print(r'Gráfico II')

plt.show()
# Criação da coluna com o valor referente às Horas

final['purchase_hour'] = final['order_purchase_timestamp'].dt.hour



figure(num=None, figsize=(10, 3), dpi=100)

plt.hist(final['purchase_hour'], bins=24, facecolor='b', alpha=0.6)

plt.xticks(ticks=np.arange(24))

plt.xlabel('Hora')

plt.ylabel('Qtd. de Pedidos')



print(r'Gráfico III')

plt.show()
#Seleção de informações agrupadas pelo consumidor

cus_valor = final.groupby('customer_unique_id', as_index=False)['price'].sum() #price_x

cus_qtd = final.groupby('customer_unique_id', as_index=False)['price'].count() #price_y

cus_frete = final.groupby('customer_unique_id', as_index=False)['freight_value'].sum()

cus_loc = final[['customer_unique_id', 'geolocation_lat', 'geolocation_lng', 'customer_state']].drop_duplicates('customer_unique_id')

cus_review = final.groupby('customer_unique_id', as_index=False)['review_score'].mean()



#União das informações em um Dataframe

customer = cus_valor.merge(cus_qtd, on='customer_unique_id')

customer = customer.merge(cus_frete, on='customer_unique_id')

customer = customer.merge(cus_loc, on='customer_unique_id')

customer = customer.merge(cus_review, on='customer_unique_id')

customer = customer.rename(columns={'price_x':'price', 'price_y':'count_items'})



print('Média do valor de compra: R$ ' + str(round(customer['price'].mean(),2)) + '\nDesvio Padrão: R$ ' + str(round(customer['price'].std(),2)))

customer.sort_values(by='price', ascending=False).head(10)
a = final[['customer_unique_id','review_score', 'review_comment_message']][

    (final['customer_unique_id'] == '0a0a92112bd4c708ca5fde585afaa872')

    |(final['customer_unique_id'] == '763c8b1c9c68a0229c42c9fc6f662b93')

    |(final['customer_unique_id'] == '459bef486812aa25204be022145caa62')]

print('Comentários por parte dos consumidores que pontuarão a avaliação com nota 1.00')

print('')

print('CONSUMIDOR 1: ' + a.groupby('customer_unique_id')['review_comment_message'].min()[0])

print('CONSUMIDOR 2: ' + a.groupby('customer_unique_id')['review_comment_message'].min()[1])

print('CONSUMIDOR 3: ' + a.groupby('customer_unique_id')['review_comment_message'].min()[2])
# Selecionando dados do Distrito Federal e uma amostra de 1000 consumidores

customer_df = customer[customer['customer_state'] == 'DF']

customer_df = customer_df.sample(1000, random_state=1223)



Clus_dataSet = customer_df[['geolocation_lat','geolocation_lng']]



db = DBSCAN(eps=0.015, min_samples=50).fit(Clus_dataSet)

labels = db.labels_

customer_df["Clus_Db"]=labels



# A sample of clusters

print('Clusters formados:')

customer_df["Clus_Db"].value_counts()
# Visualização gráfica

map_clusters = folium.Map(location=[-15.89, -47.9], zoom_start=11)

rainbow = ['#CD5C5C','#7B68EE','#FF8C00','#8B4513','#008B8B','#FF69B4']

print(' Cluster -1: Ciano\n','Cluster  0: Rosa\n','Cluster  1: Vermelho\n','Cluster  2: Azul\n','Cluster  3: Laranja\n','Cluster  4: Marrom')

# add markers to the map

markers_colors = []

for lat, lon, price, cluster in zip(customer_df['geolocation_lat'], customer_df['geolocation_lng'], customer_df['price'], customer_df['Clus_Db']):

    label = folium.Popup('R$ ' + str(price) + ' \(Cluster ' + str(cluster) + '\)', parse_html=True, max_width=150,min_width=100)

    folium.CircleMarker(

        [lat, lon],

        radius=3,

        popup=label,

        color=rainbow[cluster-1],

        fill=True,

        fill_color=rainbow[cluster-1],

        fill_opacity=0.7).add_to(map_clusters)

map_clusters 
# Agrupamento por Região encontrada na clusterização

customer_df['Região'] = customer_df['Clus_Db'].replace({-1:'Outras Satélites'

                                                       ,0:'Águas Claras'

                                                       ,1:'Asa Norte'

                                                       ,2:'Asa Sul'

                                                       ,3:'Guará'

                                                       ,4:'Cruzeiro'

                                                       })



print('Valor médio de compras:')

customer_df.groupby('Região')['price'].mean().sort_values(ascending=False)