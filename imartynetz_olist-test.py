import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
%matplotlib inline
olist = pd.read_csv("../input/olist_public_dataset.csv")
olist.head()
olist.info()
olist.describe()
olist['order_status'].value_counts()
olist['product_category_name'].value_counts()
#Checagem dos NaN no dataframe.
olist.isna().sum()
#checando quais casos order_aproved_at tem valores NaN
olist[pd.isna(olist['order_aproved_at'])]
#convertendo as strings das colunas que tenham formato datetime
olist['order_purchase_timestamp'] = pd.to_datetime(olist['order_purchase_timestamp'])
olist['order_aproved_at'] = pd.to_datetime(olist['order_aproved_at'])
olist['order_estimated_delivery_date'] = pd.to_datetime(olist['order_estimated_delivery_date'])
olist['order_delivered_customer_date'] = pd.to_datetime(olist['order_delivered_customer_date'])
olist['review_creation_date'] = pd.to_datetime(olist['review_creation_date'])
olist['review_answer_timestamp'] = pd.to_datetime(olist['review_answer_timestamp'])
#vendo qual intervalo de tempo entre a pessoa fazer o pedido e o pedido ser aprovado em minutos.
olist['delta_purch_aprov'] = (olist['order_aproved_at'] - olist['order_purchase_timestamp']).astype('timedelta64[m]')
olist.describe()
sns.set_style('whitegrid')
plt.figure(figsize=(12,8))
plt.title('Intervalo de tempo em minutos entre o cliente comprar e o pedido ser aprovado')
olist['delta_purch_aprov'].hist(bins=100)
plt.figure(figsize=(12,8))
plt.title("Zoom na parte que tem maior quantidade de contagens")
olist['delta_purch_aprov'].hist(bins=100, range=[0,200])
olist['order_aproved_at'].fillna(olist['order_purchase_timestamp']+ pd.Timedelta(minutes=19),inplace=True)
olist['order_aproved_at'].isna().value_counts()
#com isso todos valores NaN da coluna *oder_aproved_at* foram substituidos.
olist['delta_esti_deliv'] = (olist['order_delivered_customer_date'] - olist['order_estimated_delivery_date']).astype('timedelta64[D]')
olist.describe()
plt.figure(figsize=(12,8))
plt.title('Intervalo de tempo em dias entre a estimativa e o dia que o cliente recebe o produto')
olist['delta_esti_deliv'].hist(bins=100)
plt.figure(figsize=(12,8))
olist['delta_esti_deliv'].hist(bins=100, range= [-50,50])
olist[olist['delta_esti_deliv'] > 0]['delta_esti_deliv'].describe()
g = sns.FacetGrid(olist,col = 'review_score',height=8)
g = g.map(plt.hist, 'delta_esti_deliv', bins = 100, range=[-50,50])
olist['delta_esti_deliv'].corr(olist['review_score'])

olist['order_delivered_customer_date'].fillna(olist['order_estimated_delivery_date']- pd.Timedelta(days=12),inplace=True)
olist['order_delivered_customer_date'].isna().value_counts()
olist['year_aproved'] = olist['order_aproved_at'].apply(lambda time : time.year)
#Valor total de vendas por ano de cada categoria.
Total_sell_by_year = olist.groupby(by=['product_category_name','year_aproved'],sort=False)['order_products_value'].sum()
Total_sell_by_year.sort_index(ascending=False)
Total_sell_by_year.head(20)
olist.groupby(by=['year_aproved','product_category_name'],sort=False)['order_products_value'].count().head(20)
Total_sell_2016 = olist[olist['year_aproved']==2016].groupby('product_category_name').agg({'order_products_value':sum})
Total_sell_2016 = Total_sell_2016['order_products_value'].groupby(level=0, group_keys=False)
Total_sell_2016 = Total_sell_2016.apply(lambda x: x.sort_values(ascending=False))
Total_sell_2016.nlargest(20)
plt.figure(figsize=(20,8))
plt.title('Valor total anual de vendas de 2016 dos 20 maiores categorias vendidas.')
plt.xlabel('Categoria de produtos')
plt.ylabel('Valor total anual de vendas.')
Total_sell_2016.nlargest(20).plot(kind='bar')
Total_sell_2017 = olist[olist['year_aproved']==2017].groupby('product_category_name').agg({'order_products_value':sum})
Total_sell_2017 = Total_sell_2017['order_products_value'].groupby(level=0, group_keys=False)
Total_sell_2017 = Total_sell_2017.apply(lambda x: x.sort_values(ascending=False))
Total_sell_2017.nlargest(20)
plt.figure(figsize=(20,8))
plt.title('Valor total anual de vendas de 2017 dos 20 maiores categorias vendidas.')
plt.xlabel('Categoria de produtos')
plt.ylabel('Valor total anual de vendas.')
Total_sell_2017.nlargest(20).plot(kind='bar')
Total_sell_2018 = olist[olist['year_aproved']==2018].groupby('product_category_name').agg({'order_products_value':sum})
Total_sell_2018 = Total_sell_2018['order_products_value'].groupby(level=0, group_keys=False)
Total_sell_2018 = Total_sell_2018.apply(lambda x: x.sort_values(ascending=False))
Total_sell_2018.nlargest(20)
plt.figure(figsize=(20,8))
plt.title('Valor total anual de vendas de 2018 dos 20 maiores categorias vendidas.')
plt.xlabel('Categoria de produtos')
plt.ylabel('Valor total anual de vendas.')
Total_sell_2018.nlargest(20).plot(kind='bar')
sell_2016 = Total_sell_2016.nlargest(20)
sell_2017 = Total_sell_2017.nlargest(20)
sell_2018 = Total_sell_2018.nlargest(20)
Total_sell = pd.DataFrame(data = [sell_2016,sell_2017,sell_2018])
Total_sell = Total_sell.T
Total_sell.columns = ['order_products_value_2016','order_products_value_2017','order_products_value_2018']
Total_sell
ax = Total_sell.plot(kind='bar',figsize=(20,12))
plt.ylabel('Valor total de vendas.')
ax.legend(['2016','2017','2018'])
Total_quant_2016 = olist[olist['year_aproved']==2016].groupby('product_category_name').agg({'order_sellers_qty':sum})
Total_quant_2016 = Total_quant_2016['order_sellers_qty'].groupby(level=0, group_keys=False)
Total_quant_2016 = Total_quant_2016.apply(lambda x: x.sort_values(ascending=False))
Total_quant_2016.nlargest(20)
plt.figure(figsize=(20,8))
plt.title('Quantidade total anual de produtos vendidos em 2016 dos 20 maiores categorias vendidas.')
plt.xlabel('Categoria de produtos')
plt.ylabel('Quantidade total de vendas.')
Total_quant_2016.nlargest(20).plot(kind='bar')
Total_quant_2017 = olist[olist['year_aproved']==2017].groupby('product_category_name').agg({'order_sellers_qty':sum})
Total_quant_2017 = Total_quant_2017['order_sellers_qty'].groupby(level=0, group_keys=False)
Total_quant_2017 = Total_quant_2017.apply(lambda x: x.sort_values(ascending=False))
Total_quant_2017.nlargest(20)
plt.figure(figsize=(20,8))
plt.title('Quantidade total anual de produtos vendidos em 2017 dos 20 maiores categorias vendidas.')
plt.xlabel('Categoria de produtos')
plt.ylabel('Quantidade total de vendas.')
Total_quant_2017.nlargest(20).plot(kind='bar')
Total_quant_2018 = olist[olist['year_aproved']==2018].groupby('product_category_name').agg({'order_sellers_qty':sum})
Total_quant_2018 = Total_quant_2018['order_sellers_qty'].groupby(level=0, group_keys=False)
Total_quant_2018 = Total_quant_2018.apply(lambda x: x.sort_values(ascending=False))
Total_quant_2018.nlargest(20)
plt.figure(figsize=(20,8))
plt.title('Quantidade total anual de produtos vendidos em 2018 dos 20 maiores categorias vendidas.')
plt.xlabel('Categoria de produtos')
plt.ylabel('Quantidade total de vendas.')
Total_quant_2017.nlargest(20).plot(kind='bar')
quant_2016 = Total_quant_2016.nlargest(20)
quant_2017 = Total_quant_2017.nlargest(20)
quant_2018 = Total_quant_2018.nlargest(20)
Total_quant = pd.DataFrame(data = [quant_2016*10,quant_2017,quant_2018])
Total_quant = Total_quant.T
Total_quant.columns= ['order_sellers_qty_2016','order_sellers_qty_2017','order_sellers_qty_2018']

Total_quant
ax = Total_quant.plot(kind='bar',figsize=(20,12))
plt.ylabel('Quantidade total de produtos vendidos.')
ax.legend(['Total produtos vendidos em 2016 (valor multiplicado por 10)','Total produtos vendidos em 2017','Total produtos vendidos em 2018'])
olist['month_aproved'] = olist['order_aproved_at'].apply(lambda time : time.month)
olist[olist['year_aproved']==2018]['month_aproved'].max()
olist[olist['year_aproved']==2016]['month_aproved'].max()
olist[olist['year_aproved']==2016]['month_aproved'].min()
olist.head()
Total_quant_2017_month = olist[(olist['month_aproved']<=9) & (olist['year_aproved']==2017)].groupby('product_category_name').agg({'order_sellers_qty':sum})
Total_quant_2017_month = Total_quant_2017_month['order_sellers_qty'].groupby(level=0, group_keys=False)
Total_quant_2017_month = Total_quant_2017_month.apply(lambda x: x.sort_values(ascending=False))
Total_quant_2017_month.nlargest(20)
quant_2017_month = Total_quant_2017_month.nlargest(20)
quant_2018 = Total_quant_2018.nlargest(20)
Total_quant_month = pd.DataFrame(data = [quant_2017_month,quant_2018])
Total_quant_month = Total_quant_month.T
Total_quant_month.columns= ['order_sellers_qty_2017','order_sellers_qty_2018']
Total_quant_month
ax = Total_quant_month.plot(kind='bar',figsize=(20,12))
plt.title('Quantidade total de produtos vendidos entre os meses de janeiro a setembro.')
plt.ylabel('Quantidade total de produtos vendidos.')
ax.legend(['Total produtos vendidos em 2017','Total produtos vendidos em 2018'])

olist_2017_quant = olist[olist['year_aproved']==2017]
Tq_2017= Total_quant['order_sellers_qty_2017'].dropna().reset_index()
olist_2017_quant = olist_2017_quant[olist_2017_quant['product_category_name'].isin(Tq_2017['index'])].groupby(by = ['product_category_name','month_aproved']).sum()
olist_2017_quant.reset_index(inplace=True)
olist_2017_quant.head(20)
plt.figure(figsize=(25,10))
plt.title('Quantidade de vendas por m??s dos produtos mais vendidos no ano de 2017')
plt.xticks(np.arange(1,13,1))
sns.lineplot(x = "month_aproved" , y = 'order_items_qty', data = olist_2017_quant, hue='product_category_name', palette= 'muted')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
olist_2017_sell = olist[olist['year_aproved']==2017]
Ts_2017 = Total_sell['order_products_value_2017'].dropna().reset_index()
olist_2017_sell = olist_2017_sell[olist_2017_sell['product_category_name'].isin(Ts_2017['index'])].groupby(by = ['product_category_name','month_aproved']).sum()
olist_2017_sell.reset_index(inplace=True)
olist_2017_sell.head(20)
plt.figure(figsize=(25,10))
plt.title('Valor total de vendas por m??s dos produtos mais vendidos no ano de 2017')
plt.xticks(np.arange(1,13,1))
sns.lineplot(x = "month_aproved" , y = 'order_products_value', data = olist_2017_sell, hue='product_category_name', palette= 'Paired')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
olist_2018_quant = olist[olist['year_aproved']==2018]
Tqm_2018 = Total_quant['order_sellers_qty_2018'].dropna().reset_index()
olist_2018_quant = olist_2018_quant[olist_2018_quant['product_category_name'].isin(Tqm_2018['index'])].groupby(by = ['product_category_name','month_aproved']).sum()
olist_2018_quant.reset_index(inplace=True)
olist_2018_quant.head(20)
plt.figure(figsize=(25,10))
plt.title('Quantidade de vendas por m??s dos produtos mais vendidos no ano de 2018')
plt.xticks(np.arange(1,13,1))
sns.lineplot(x = "month_aproved" , y = 'order_items_qty', data = olist_2018_quant, hue='product_category_name', palette= 'muted')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
olist_2018_sell = olist[olist['year_aproved']==2018]
Ts_2018 = Total_sell['order_products_value_2017'].dropna().reset_index()
olist_2018_sell = olist_2018_sell[olist_2018_sell['product_category_name'].isin(Ts_2018['index'])].groupby(by = ['product_category_name','month_aproved']).sum()
olist_2018_sell.reset_index(inplace=True)
olist_2018_sell.head(20)
plt.figure(figsize=(25,10))
plt.title('Valor total de vendas por m??s dos produtos mais vendidos no ano de 2018')
plt.xticks(np.arange(1,13,1))
sns.lineplot(x = "month_aproved" , y = 'order_products_value', data = olist_2018_sell, hue='product_category_name', palette= 'Paired')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
