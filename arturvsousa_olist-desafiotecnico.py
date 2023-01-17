print("Estas duas linhas são código em Python. A de cima é o comando feito, e a de baixo é o que retorna desse comando.")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from scipy.stats import pearsonr
import seaborn as sns
import random
import re
from sklearn import preprocessing
import operator
plt.rcParams.update({'font.size': 15})
plt.rcParams['figure.figsize'] = 18, 5
pd.set_option('display.max_colwidth', 50)
customers = pd.read_csv('../input/olist-compras-20162018/olist_customers_dataset.csv')
geoloc = pd.read_csv('../input/olist-compras-20162018/olist_geolocation_dataset.csv')
geoloc.head()
order_items = pd.read_csv('../input/olist-compras-20162018/olist_order_items_dataset.csv')
order_items.head()
order_payments = pd.read_csv('../input/olist-compras-20162018/olist_order_payments_dataset.csv')
order_payments.head()
order_reviews = pd.read_csv('../input/olist-compras-20162018/olist_order_reviews_dataset.csv')
order_reviews.head()
order_reviews.head()
orders = pd.read_csv('../input/olist-compras-20162018/olist_orders_dataset.csv')
orders.head()
products = pd.read_csv('../input/olist-compras-20162018/olist_products_dataset.csv')
products.head()
sellers = pd.read_csv('../input/olist-compras-20162018/olist_sellers_dataset.csv')
sellers.head()
product_category_name_translation = pd.read_csv('../input/olist-compras-20162018/product_category_name_translation.csv')
product_category_name_translation
del product_category_name_translation
order_reviews.head(10)
order_reviews.isnull().sum()
order_reviews_not_null = order_reviews.dropna()
order_reviews_not_null['review_comment_title'].isnull().sum()
order_reviews_not_null['review_comment_message'].isnull().sum()
mensagens = order_reviews_not_null['review_comment_message'].str.cat(sep=' ')
mensagens = mensagens.replace('que', '').replace('de', '').replace('porém', '').replace('para', ' ').replace('por', '').replace('uma', '').replace('pois', '').replace('um', '').replace('ma', '')
wordcloud = WordCloud(background_color='white', scale=2).generate(mensagens)
plt.figure(figsize=(18, 10), dpi=100)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()
order_reviews_not_null[order_reviews_not_null['review_comment_message'].str.contains('prazo')]['review_comment_message']
prazo_pontos = order_reviews_not_null[order_reviews_not_null['review_comment_message'].str.contains('prazo')]['review_score'].value_counts()
graph_prazo_pontos = prazo_pontos.plot(kind='bar', rot=0)
graph_prazo_pontos.set_xlabel('Nota')
graph_prazo_pontos.set_ylabel('Pessoas que usaram o termo "prazo"')
titulos = order_reviews_not_null['review_comment_title'].str.cat(sep=' ')
wordcloud = WordCloud(background_color="white", scale=2).generate(titulos)
plt.figure(figsize=(18, 10), dpi=100)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
del titulos
del wordcloud
del graph_prazo_pontos
del prazo_pontos
del order_reviews_not_null
del mensagens
order_reviews[order_reviews['review_comment_message'].isnull()]
sem_coment = order_reviews[order_reviews['review_comment_message'].isnull()]
graph_sem_coment = sem_coment['review_score'].value_counts().plot(kind='bar', rot=0)
graph_sem_coment.set_xlabel('Nota')
graph_sem_coment.set_ylabel('Número de clientes')
del graph_sem_coment
orders['order_id'] = pd.Series(orders['order_id'], dtype='string')
orders['order_id'].head()
sem_coment['order_id'] = pd.Series(sem_coment['order_id'], dtype='string')
sem_coment['order_id'].head()
sem_coment_full = sem_coment.merge(orders, how='inner', on='order_id' )
sem_coment_full = sem_coment_full.merge(order_items, how='inner', on='order_id')
sem_coment_full = sem_coment_full.merge(products, how='inner', on='product_id')
sem_coment_full.info()
sem_coment_full['product_category_name'].value_counts().head(10).plot(kind='bar', rot=45)
plt.ylabel('Quantidade de produtos comprados')
sem_coment_full[sem_coment_full['product_category_name'] == 'cama_mesa_banho']['price'].describe()
sem_coment_full[sem_coment_full['product_category_name'] == 'beleza_saude']['price'].describe()
sem_coment_full[sem_coment_full['product_category_name'] == 'esporte_lazer']['price'].describe()
sem_coment_full[sem_coment_full['product_category_name'] == 'moveis_decoracao']['price'].describe()
gastos_geral = order_items.merge(products, on='product_id', how='inner')
gastos_geral['price'].describe()
gastos_geral[gastos_geral['product_category_name'] == 'cama_mesa_banho']['price'].describe()
gastos_geral['product_category_name'].value_counts()
gastos_geral = gastos_geral.merge(order_reviews, on='order_id', how='inner')
com_coment_full = gastos_geral[gastos_geral['review_comment_message'].notnull()]
com_coment_full = com_coment_full.merge(orders, on='order_id', how='inner')
com_coment_full = com_coment_full.merge(customers, on='customer_id', how='inner')
com_coment_full['product_category_name'].value_counts()
com_coment_full_graph = com_coment_full['review_score'].value_counts().plot(kind='bar', rot=0)
com_coment_full_graph.set_xlabel('Nota')
com_coment_full_graph.set_ylabel('Número de clientes')
com_coment_full['price'].describe()
del sem_coment_full
del sem_coment
del gastos_geral
del com_coment_full_graph
compras = orders.merge(order_items, on='order_id', how='inner')
compras.head()
compras['order_approved_at'] = compras['order_approved_at'].str.replace('-\d\d \d\d:\d\d:\d\d', '')
compras['order_approved_at'] = pd.to_datetime(compras['order_approved_at'])
pd.pivot_table(compras,index='order_approved_at', values='price', aggfunc='sum').plot()
periodo_de_queda = compras[(compras['order_approved_at'] < '2018-10') & (compras['order_approved_at'] > '2018-07')]
pd.pivot_table(periodo_de_queda,index='order_approved_at', values='price', aggfunc='sum').plot()
compras_clean = compras[(compras['order_approved_at'] >= '2017-01') & (compras['order_approved_at'] < '2018-08-01')]
pd.pivot_table(compras_clean,index='order_approved_at', values='price', aggfunc='sum').plot()
compras_clean = orders.merge(order_items, on='order_id', how='inner')
compras_clean.head()
compras_clean['order_approved_at'] = compras_clean['order_approved_at'].str.replace(' \d\d:\d\d:\d\d', '')
compras_clean['order_approved_at']
compras_clean = compras_clean[(compras_clean['order_approved_at'] >= '2017-06-01') & (compras_clean['order_approved_at'] < '2018-08-01')]
compras_clean.info()
compras_clean['order_approved_at'].head()
compras_clean = compras_clean.groupby(by='order_approved_at').agg('sum')
compras_clean['data_compra'] = compras_clean.index
compras_clean.reset_index(drop=True, inplace=True)
compras_clean['dias'] = compras_clean.index
compras_clean.tail()
compras_clean.describe()
compras_clean.plot(kind='scatter', x='dias', y='price')
compras_clean = compras_clean[compras_clean['price'] < 60000]
corr, _ = pearsonr(compras_clean['dias'], compras_clean['price'])
corr
X = compras_clean['dias'].values.reshape(-1, 1)
y = compras_clean['price'].values.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.intercept_)
print(regressor.coef_)
y_pred = regressor.predict(X_test)
mostrar = pd.DataFrame({'Dados reais': y_test.flatten(), 'Previsao': y_pred.flatten()})
mostrar
mostrar = mostrar.head(25)
mostrar.plot(kind='bar')
plt.figure()
plt.scatter(X_test, y_test, color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()
futuro = []

inicio = 421
for i in range(730):
    inicio+=1
    futuro.append(inicio)

futuro = pd.DataFrame(futuro)
futuro = np.array(futuro)
predicoes = pd.DataFrame(regressor.predict((futuro)))
datelist = pd.date_range(start='2019-06-02', periods=730)
predicoes['data'] = datelist
predicoes.columns = 'vendas', 'data'
predicoes = predicoes[['data', 'vendas']]
predicoes
mostrar
series = compras_clean[['price']]
predicoes[predicoes['data'] == '2019-08-17']
predicoes[predicoes['data'] > '2018-12-31']['vendas'].sum()
RMSE = mean_squared_error(y_test, regressor.predict(X_test))**0.5
RMSE
entregas = customers.rename(columns={'customer_zip_code_prefix':'geolocation_zip_code_prefix'})
entregas = entregas[['customer_id', 'geolocation_zip_code_prefix', 'customer_city', 'customer_state']]
entregas = entregas.merge(orders, on='customer_id', how='inner')
entregas = entregas.merge(order_items, on='order_id', how='inner')
entregas = entregas[['customer_id', 'geolocation_zip_code_prefix', 'customer_city', 'customer_state', 'order_approved_at', 'order_delivered_carrier_date', 'order_delivered_customer_date', 'order_estimated_delivery_date', 'shipping_limit_date', 'freight_value']]
entregas['order_estimated_delivery_date'] = pd.to_datetime(entregas['order_estimated_delivery_date'])
entregas['order_delivered_customer_date'] = pd.to_datetime(entregas['order_delivered_customer_date'])
entregas['order_delivered_carrier_date'] = pd.to_datetime(entregas['order_delivered_carrier_date'])
entregas['order_approved_at'] = pd.to_datetime(entregas['order_approved_at'])
entregas['dif_entregaest_x_efetiva'] = entregas['order_estimated_delivery_date'] - entregas['order_delivered_customer_date']
entregas = entregas.dropna()
entregas[entregas['dif_entregaest_x_efetiva'] < '0 days']
entregas['horario_despachado'] = entregas['order_delivered_carrier_date'].dt.hour
entregas['dif_entregaest_x_efetiva'] = entregas['dif_entregaest_x_efetiva'].astype(str)
entregas['dif_entregaest_x_efetiva'] = entregas['dif_entregaest_x_efetiva'].str.replace(' days.*', '')
entregas['dif_entregaest_x_efetiva'] = entregas['dif_entregaest_x_efetiva'].astype(int)
entregas[entregas['horario_despachado'] > 18]
entregas[entregas['dif_entregaest_x_efetiva'] < 0]['horario_despachado'].value_counts().sort_index()
entregas[entregas['dif_entregaest_x_efetiva'] >= 0]['horario_despachado'].value_counts().sort_index()
percent_atraso_hora_despachado = {}

for i in range(24):
    hora = i+1
    if hora < 24:
        percentual_atraso = len(entregas[(entregas['horario_despachado'] == hora) & (entregas['dif_entregaest_x_efetiva'] < 0)])/len(entregas[(entregas['horario_despachado'] == hora)])
        percent_atraso_hora_despachado[hora] = percentual_atraso
    else:
        percentual_atraso = len(entregas[(entregas['horario_despachado'] == 0) & (entregas['dif_entregaest_x_efetiva'] < 0)])/len(entregas[(entregas['horario_despachado'] == 0)])
        percent_atraso_hora_despachado[hora] = percentual_atraso
percent_atraso_hora_despachado = pd.DataFrame.from_dict(percent_atraso_hora_despachado, orient='index')
percent_atraso_hora_despachado['hora'] = percent_atraso_hora_despachado.index
percent_atraso_hora_despachado.columns = 'percentual de atraso', 'hora de despacho'
percent_atraso_hora_despachado = percent_atraso_hora_despachado[['hora de despacho', 'percentual de atraso']]
percent_atraso_hora_despachado['percentual de atraso'] = percent_atraso_hora_despachado['percentual de atraso']*100
percent_atraso_hora_despachado.plot(x='hora de despacho', y='percentual de atraso', legend=False)
plt.xticks(percent_atraso_hora_despachado['hora de despacho'])
plt.ylabel('percentual de atraso')
entregas['horario_despachado'].value_counts().plot(kind='bar', rot=0)
plt.ylabel('Número de produtos despachados')
plt.xlabel('Hora despachada')
plt.title('Produtos despachados à transportadora x horário de despacho')
grupo_a = entregas[(entregas['dif_entregaest_x_efetiva'] < 0) & ((entregas['horario_despachado'] == 14) | (entregas['horario_despachado'] == 15) | (entregas['horario_despachado'] == 19))]
grupo_b = entregas[(entregas['dif_entregaest_x_efetiva'] < 0) & ((entregas['horario_despachado'] == 9) | (entregas['horario_despachado'] == 10) | (entregas['horario_despachado'] == 11))]
xa = grupo_a['dif_entregaest_x_efetiva'].mean()
xb = grupo_b['dif_entregaest_x_efetiva'].mean()
estatistica_teste = xb-xa
estatistica_teste
dois_grupos = pd.concat([grupo_a, grupo_b], ignore_index=True)
dois_grupos['horario_despachado'].unique()
todosgrupos = dois_grupos['dif_entregaest_x_efetiva'].to_list()
media_dif = []

for i in range(1000):
    grupo_a = []
    grupo_b = []
    for valor in todosgrupos:
        grupo = np.random.rand()
        if grupo >= 0.5:
            grupo_a.append(valor)
        else:
            grupo_b.append(valor)
    
    media_a = np.mean(grupo_a)
    media_b = np.mean(grupo_b)
    diferenca_media = media_b = media_a
    media_dif.append(diferenca_media)
sampling_dist = {}

for value in media_dif:
    if sampling_dist.get(value, False):
        newval = sampling_dist.get(value)
        newval += 1
    else:
        sampling_dist[value] = 1
freqs = []

for key in sampling_dist:
    if key >= estatistica_teste:
        freqs.append(key)

p_value = sum(freqs)
p_value /= 1000
p_value
del percent_atraso_hora_despachado
del entregas
del todosgrupos
del dois_grupos
del grupo_a
del grupo_b
del media_dif
del sampling_dist
customers = customers[['customer_id', 'customer_state']]
produtos = products.merge(order_items, on='product_id')
produtos = produtos.drop(columns=['product_weight_g', 'product_length_cm', 'product_height_cm', 'product_width_cm', 'order_item_id', 'shipping_limit_date'])
produtos = produtos.merge(sellers, on='seller_id', how='inner')
customers = customers.merge(orders, on='customer_id', how='inner')
customers = customers.merge(produtos, on='order_id', how='inner')
customers
sul = customers[(customers['customer_state'] == 'PR') | (customers['customer_state'] == 'SC') | (customers['customer_state'] == 'RS')]
sudeste = customers[(customers['customer_state'] == 'SP') | (customers['customer_state'] == 'MG') | (customers['customer_state'] == 'RJ') | (customers['customer_state'] == 'ES')]
nordeste = customers[(customers['customer_state'] == 'AL') | (customers['customer_state'] == 'BA') | (customers['customer_state'] == 'CE') | (customers['customer_state'] == 'MA') | (customers['customer_state'] == 'PB') | (customers['customer_state'] == 'PE') | (customers['customer_state'] == 'PI') | (customers['customer_state'] == 'RN') | (customers['customer_state'] == 'SE')]
centro_oeste = customers[(customers['customer_state'] == 'GO') | (customers['customer_state'] == 'MT') | (customers['customer_state'] == 'MS')]
norte = customers[(customers['customer_state'] == 'AC') | (customers['customer_state'] == 'AP') | (customers['customer_state'] == 'AM') | (customers['customer_state'] == 'PA') | (customers['customer_state'] == 'RO') | (customers['customer_state'] == 'RR') | (customers['customer_state'] == 'TO')]
customers.loc[(customers['customer_state'] == 'PR') | (customers['customer_state'] == 'SC') | (customers['customer_state'] == 'RS'), 'regiao'] = 'sul'
customers.loc[(customers['customer_state'] == 'SP') | (customers['customer_state'] == 'MG') | (customers['customer_state'] == 'RJ') | (customers['customer_state'] == 'ES'), 'regiao'] = 'sudeste'
customers.loc[(customers['customer_state'] == 'AL') | (customers['customer_state'] == 'BA') | (customers['customer_state'] == 'CE') | (customers['customer_state'] == 'MA') | (customers['customer_state'] == 'PB') | (customers['customer_state'] == 'PE') | (customers['customer_state'] == 'PI') | (customers['customer_state'] == 'RN') | (customers['customer_state'] == 'SE'), 'regiao'] = 'nordeste'
customers.loc[(customers['customer_state'] == 'GO') | (customers['customer_state'] == 'MT') | (customers['customer_state'] == 'MS'), 'regiao'] = 'centro-oeste'
customers.loc[(customers['customer_state'] == 'AC') | (customers['customer_state'] == 'AP') | (customers['customer_state'] == 'AM') | (customers['customer_state'] == 'PA') | (customers['customer_state'] == 'RO') | (customers['customer_state'] == 'RR') | (customers['customer_state'] == 'TO'), 'regiao'] = 'norte'

customers[customers['regiao'] == 'sul']['product_category_name'].value_counts()
customers[customers['regiao'] == 'sudeste']['product_category_name'].value_counts()
customers[customers['regiao'] == 'nordeste']['product_category_name'].value_counts()
customers[customers['regiao'] == 'centro-oeste']['product_category_name'].value_counts()
customers[customers['regiao'] == 'norte']['product_category_name'].value_counts()
customers.groupby(by='regiao').agg('sum').astype(int)[['price']].plot(kind='bar', legend=False)
plt.ylabel('Valor absoluto gasto em milhões')
plt.xlabel('')
customers.groupby(by='regiao').agg('mean').astype(int)[['price']].plot(kind='bar', legend=False, rot=45)
plt.xlabel('')
plt.ylabel('Gasto médio por compra em Reais')
b = sns.boxplot(showfliers=False, data = customers[['price', 'regiao']],
                x = 'regiao',
                y = 'price',
                order = ['sudeste', # custom order of boxplots
                         'sul',
                         'nordeste',
                         'norte',
                         'centro-oeste'])
plt.ylabel('Valor médio gasto por compra')
plt.xlabel('')
b = sns.boxplot(showfliers=False, data = customers,
                x = 'regiao',
                y = 'freight_value',
                order = ['sudeste', # custom order of boxplots
                         'sul',
                         'nordeste',
                         'norte',
                         'centro-oeste'])
plt.ylabel('Valor médio gasto por frete')
plt.xlabel('')
del b
del customers
order_reviews = order_reviews.merge(orders, on='order_id', how='inner')
order_reviews.info()
order_reviews = order_reviews.merge(order_items, on='order_id', how='inner')
order_reviews = order_reviews.merge(products, on='product_id', how='inner')
order_reviews
order_reviews.info()
order_reviews.drop_duplicates(subset ="order_id", keep = False, inplace = True) 
insatisfeitos = order_reviews[(order_reviews['review_score'] == 1) | (order_reviews['review_score'] == 2)]
titulos = insatisfeitos['review_comment_title'].str.cat(sep=' ')
wordcloud = WordCloud(background_color="white", scale=2).generate(titulos)
plt.figure(figsize=(18, 10), dpi=100)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
mensagens = insatisfeitos['review_comment_message'].str.cat(sep=' ')
wordcloud = WordCloud(background_color="white", scale=2).generate(mensagens)
plt.figure(figsize=(18, 10), dpi=100)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
len(insatisfeitos)
insatisfeitos['product_category_name'].value_counts()
order_reviews[order_reviews['review_score'] > 2 ]['product_category_name'].value_counts()
len(order_reviews[(order_reviews['product_category_name'] == 'cama_mesa_banho') & (order_reviews['review_score'] <= 2)]) / len(order_reviews[(order_reviews['product_category_name'] == 'cama_mesa_banho') & (order_reviews['review_score'] > 2)])
len(order_reviews[(order_reviews['product_category_name'] == 'beleza_saude') & (order_reviews['review_score'] <= 2)]) / len(order_reviews[(order_reviews['product_category_name'] == 'beleza_saude') & (order_reviews['review_score'] > 2)])
insatisfeitos['seller_id'].value_counts()
mensagens = insatisfeitos[insatisfeitos['seller_id'] == '4a3ca9315b744ce9f8e9374361493884']['review_comment_message'].str.cat(sep=' ').replace('de', '').replace('que', '').replace('um', '').replace('produto', '').replace('muito', '').replace('da', '').replace('na', '').replace('para', '').replace('ma', '').replace('meu', '')
wordcloud = WordCloud(background_color="white", scale=2).generate(mensagens)
plt.figure(figsize=(18, 10), dpi=100)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
mensagens = insatisfeitos[insatisfeitos['seller_id'] == '6560211a19b47992c3666cc44a7e94c0']['review_comment_message'].str.cat(sep=' ').replace('de', '').replace('que', '').replace('um', '').replace('produto', '').replace('muito', '').replace('da', '').replace('na', '').replace('para', '').replace('ma', '').replace('meu', '').replace('Produto', '').replace('veio', '').replace('veio', '')
wordcloud = WordCloud(background_color="white", scale=2).generate(mensagens)
plt.figure(figsize=(18, 10), dpi=100)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
mensagens = insatisfeitos[insatisfeitos['seller_id'] == 'cc419e0650a3c5ba77189a1882b7556a']['review_comment_message'].str.cat(sep=' ').replace('de', '').replace('que', '').replace('um', '').replace('produto', '').replace('muito', '').replace('da', '').replace('na', '').replace('para', '').replace('ma', '').replace('meu', '').replace('Produto', '').replace('veio', '').replace('veio', '')
wordcloud = WordCloud(background_color="white", scale=2).generate(mensagens)
plt.figure(figsize=(18, 10), dpi=100)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
mensagens = insatisfeitos[insatisfeitos['product_category_name'] == 'informatica_acessorios']['review_comment_message'].str.cat(sep=' ').replace('de', '').replace('que', '').replace('um', '').replace('produto', '').replace('muito', '').replace('da', '').replace('na', '').replace('para', '').replace('ma', '').replace('meu', '').replace('Produto', '').replace('veio', '').replace('veio', '')
wordcloud = WordCloud(background_color="white", scale=2).generate(mensagens)
plt.figure(figsize=(18, 10), dpi=100)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis('off')
plt.show()
insatisfeitos = insatisfeitos.dropna()
pd.set_option('display.max_colwidth', -1)
insatisfeitos[insatisfeitos['review_comment_message'].str.contains('targaryen')]['review_comment_message']
insatisfeitos[insatisfeitos['review_comment_message'].str.contains('lannister')]['review_comment_message']
insatisfeitos[insatisfeitos['review_comment_message'].str.contains('stark')]['review_comment_message']
del insatisfeitos
del order_reviews
del mensagens
pd.set_option('display.max_colwidth', 50)