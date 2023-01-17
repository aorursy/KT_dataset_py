# Primeiramente iremos importar todas as bibliotecas necessárias para começar a varredura no site do zap_imoveis
from urllib.request import urlretrieve, urlopen, Request
from urllib.error import URLError, HTTPError
import pandas as pd
from bs4 import BeautifulSoup
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import requests
import time
from shapely.geometry import Polygon, Point, LineString, MultiPolygon
import geopandas as gpd
import statsmodels.api as sm
def trata_html(input):
    html_tratada = ' '.join(input.split()).replace('> <', '><')
    return html_tratada
imoveis = []
for i in range(1, 11):
    url = 'https://www.zapimoveis.com.br/venda/imoveis/sp+sao-bernardo-do-campo/?__zt=spg%3Ac&pagina=' + str(i) + '&onde=,S%C3%A3o%20Paulo,S%C3%A3o%20Bernardo%20do%20Campo,,,,BR%3ESao%20Paulo%3ENULL%3ESao%20Bernardo%20do%20Campo,-23.6816587,-46.6203412&transacao=Venda&tipo=Im%C3%B3vel%20usado'
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36'}
    req = Request(url, headers = headers)
    response = urlopen(req)
    html = response.read()
    html = html.decode('utf-8')
    trata_html(html)
    soup = BeautifulSoup(html, 'html.parser')
    anuncios = soup.findAll('div', {'class': 'card-listing simple-card js-listing-card'})

    for anuncio in anuncios:
        dic_anuncio = {}

    # Requisitando o preço dos imóveis

        if anuncio.find('p', {'class':'simple-card__price'}) != None and anuncio.find('p', {'class':'simple-card__price'}).get_text().strip() != 'Sob consulta':
            dic_anuncio['Valor_Imovel'] = float(''.join(anuncio.find('p', {'class':'simple-card__price'}).get_text().split()[-1].split('.')))
        
        else:
            dic_anuncio['Valor_Imovel'] = 0

    #Requisitando o preço do condominio
        if anuncio.find('li', {'class':'condominium'}) != None:
            dic_anuncio['valor_Condominio'] = float(''.join(anuncio.find('li', {'class':'condominium'}).get_text().split()[-1].split('.')))
        else:
            dic_anuncio['valor_Condominio'] = 0


    #Requisitando IPTU
        if anuncio.find('li', {'class':'iptu'}) != None:
            dic_anuncio['IPTU'] = float(anuncio.find('li', {'class':'iptu'}).get_text().split()[-1])
        else:
            dic_anuncio['IPTU'] = 0


    #Requisitando Endereço
        if anuncio.find('p', {'class':'simple-card__address'}) != None:
            dic_anuncio['Endereco'] = anuncio.find('p', {'class':'simple-card__address'}).get_text()
        else:
            dic_anuncio['Endereco'] = 0

    #Requisitando Àrea do Imóvel
        if anuncio.find('li', {'class':'js-areas'}) != None:
            dic_anuncio['Area_m2'] = anuncio.find('li', {'class':'js-areas'}).get_text().split()[-2]
        else:
            dic_anuncio['Area_m2'] = 0

    #Requisitando Quantidade de Quartos
        if anuncio.find('li', {'class':'js-bedrooms'}) != None:
            dic_anuncio['Quantidade_Quartos'] = anuncio.find('li', {'class':'js-bedrooms'}).get_text().split()[-1]
        else:
            dic_anuncio['Quantidade_Quartos'] = 0

    #Requisitando Quantidade de Garagens
        if anuncio.find('li', {'class':'js-parking-spaces'}) != None:
            dic_anuncio['Quantidade_Garangens'] = anuncio.find('li', {'class':'js-parking-spaces'}).get_text().split()[-1]
        else:
            dic_anuncio['Quantidade_Garangens'] = 0

    #Requisitando quantidade de banheiros
        if anuncio.find('li', {'class':'js-bathrooms'})!= None:
            dic_anuncio['Quantidade_Banheiros'] = anuncio.find('li', {'class':'js-bathrooms'}).get_text().split()[-1]
        else:
            dic_anuncio['Quantidade_Banheiros'] = 0
        imoveis.append(dic_anuncio)
dataset = pd.DataFrame(imoveis)
dataset.head()
cep1 = pd.read_csv('../input/correios/sp.cepaberto_parte_1.csv', names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6'])
cep2 = pd.read_csv('../input/correios/sp.cepaberto_parte_2.csv', names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6'])
cep3 = pd.read_csv('../input/correios/sp.cepaberto_parte_3.csv', names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6'])
cep4 = pd.read_csv('../input/correios/sp.cepaberto_parte_4.csv', names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6'])
cep5 = pd.read_csv('../input/correios/sp.cepaberto_parte_5.csv', names=['col1', 'col2', 'col3', 'col4', 'col5', 'col6'])
cep_total = pd.concat([cep1, cep2, cep3, cep4, cep5])
cep_total = cep_total[(cep_total['col5'] == 8482) | (cep_total['col5'] == 8498)| (cep_total['col5'] == 2955) | (cep_total['col5'] == 8351) | (cep_total['col5'] == 8352) | (cep_total['col5'] == 5590)]
ceps = {}
for index, valor in dataset.iterrows():
    rua = valor['Endereco'].split(',')[0].strip()
    bairro = valor['Endereco'].split(',')[1].strip()
    valor = cep_total[(cep_total['col2'] == rua) & (cep_total['col4'] == bairro)]['col1'].values
    ceps[index] = list(valor)
for i in ceps:
    if len(ceps[i]) != 0:
        ceps[i] = str(0) + str(ceps[i][0])
ceps = pd.DataFrame(ceps.values(), index = ceps.keys())
dataset['CEP'] = ceps
dataset.head()
for index, data in dataset.iterrows():
    
    url = "https://www.cepaberto.com/api/v3/cep?cep="+ str(data['CEP'])
    # O seu token está visível apenas pra você
    headers = {'Authorization': 'Token token=5f71e3519466671e52b02de90aa733fb'}
    response = requests.get(url, headers=headers)
    try:
        if response.json() != {}:
            dataset.loc[index, 'Latitude'] = response.json()['latitude']
            dataset.loc[index, 'Longitude'] = response.json()['longitude']
        else:
            dataset.loc[index, 'Latitude'] = 0
            dataset.loc[index, 'Longitude'] = 0
        time.sleep(1)
    except ValueError:
        print("Response content is not valid JSON")
dataset.dropna(inplace = True)
print('Quantidade de nulos %s' %dataset.isnull().sum().sum())
dataset['Latitude'] = pd.to_numeric(dataset['Latitude'])
dataset['Longitude'] = pd.to_numeric(dataset['Longitude'])
dataset.dtypes
geometry = [Point(x) for x in zip(dataset.Longitude, dataset.Latitude)]
crs = {'proj': 'latlong', 'ellps': 'WGS84', 'datum': 'WGS84', 'no_defs': True}
geo_dataset = gpd.GeoDataFrame(dataset, crs = crs, geometry = geometry)
sbc = gpd.read_file('../input/cidade/SBC.shp')
sbc.crs
geo_dataset = geo_dataset.to_crs({'init': 'epsg:4674'})
geo_dataset.crs
geo_dataset = geo_dataset[geo_dataset['Latitude'] != 0]
ax = sbc.plot(color = 'white', edgecolor = 'black', figsize = (15,12))
geo_dataset.plot(ax = ax)
equipamento = gpd.read_file('../input/equipamento/EQUIPAMENTO.shp')
equipamento = equipamento.to_crs({'init': 'epsg:4674'})
estabelecimentos = list(equipamento.TIPO.value_counts().index)
ax = sbc.plot(color = 'white', edgecolor = 'black', figsize = (20,12))
geo_dataset.plot(ax = ax, color = 'yellow', alpha = 0.5, label = 'Casas', edgecolor = 'black')
equipamento[(equipamento.TIPO == 'Escola Municipal de Educação Básica - EMEB') |
            (equipamento.TIPO == 'Escola Particular') |
            (equipamento.TIPO == 'Escola Estadual')].plot(ax = ax, color = 'green', alpha = 0.4, Label = 'Escolas')
equipamento[(equipamento.TIPO == 'Unidade Básica de Saúde - UBS')].plot(ax = ax, color = 'red', alpha = 0.7, Label = 'UBS')
equipamento[(equipamento.TIPO == 'Teatro')].plot(ax = ax, color = 'orange', alpha = 0.7, Label = 'Teatro')
equipamento[(equipamento.TIPO == 'Hospital')].plot(ax = ax, color = 'cyan', alpha = 0.7, Label = 'Hospital')
ax.set_title('Mapa de São Bernardo', loc = 'left', fontsize = 16)
ax.legend()
print('Quantidade de nulos %s' %dataset.isnull().sum().sum())
print('Tamanho de DataFrame %s linhas %s colunas' %(dataset.shape[0], dataset.shape[1]))
dataset['Area_m2'] = pd.to_numeric(dataset['Area_m2'])
dataset['Quantidade_Quartos'] = pd.to_numeric(dataset['Quantidade_Quartos'])
dataset['Quantidade_Garangens'] = pd.to_numeric(dataset['Quantidade_Garangens'])
dataset['Quantidade_Banheiros'] = pd.to_numeric(dataset['Quantidade_Banheiros'])
dataset.info()
# Trabalhando com estatistica descritiva
Q1 = dataset['valor_Condominio'].quantile(0.25)
Q3 = dataset['valor_Condominio'].quantile(0.75)
IIQ =  Q3 - Q1
lim_inf = Q1 -(1.5 * IIQ)
lim_sup = Q3 +(1.5 * IIQ)
dataset.shape
dataset = dataset[(dataset['valor_Condominio'] >= lim_inf) & (dataset['valor_Condominio'] <= lim_sup) & (dataset['valor_Condominio'] != 0)]
dataset.shape
dataset = dataset[dataset['Latitude'] != 0]
dataset.shape
dataset.corr()
ax = sns.heatmap(dataset.corr(), annot= True, linewidths=.1, annot_kws={'size':12})
ax.figure.set_size_inches(16,12)
ax.set_title('Correlação das variáveis', fontsize = 16, loc = 'left')
ax = ax
ax = dataset.boxplot(column=['Valor_Imovel'], figsize = (16,12))
plt.title('Blox plot do Valor do Imóvel', loc = 'left', fontsize = 18)
dataset.describe()
ax = sns.distplot(dataset.Valor_Imovel)
ax.figure.set_size_inches(20, 6)
ax.set_title('Distribuição de Frequências', fontsize=20, loc = 'left')
ax = ax
sns.set_palette('Accent')
sns.set_style('darkgrid')
ax = sns.pairplot(dataset, x_vars= ['Area_m2','valor_Condominio', 'IPTU', 'Quantidade_Quartos',
                                    'Quantidade_Garangens', 'Quantidade_Banheiros'  ], y_vars='Valor_Imovel',
                                      height=5, kind = 'reg')
ax.fig.suptitle('Dispersão entre as Variáveis', fontsize=20, y=1.08)
ax.fig.set_size_inches(20, 6)
ax = ax
dataset_log = dataset.copy()
dataset_log.shape
dataset_log['Valor_Imovel'] = np.log(dataset_log['Valor_Imovel'])
dataset_log['valor_Condominio'] = np.log(dataset_log['valor_Condominio'])
dataset_log['IPTU'] = np.log(dataset_log['IPTU'] + 1)
dataset_log['Area_m2'] = np.log(dataset_log['Area_m2'])
dataset_log['Quantidade_Quartos'] = np.log(dataset_log['Quantidade_Quartos'] + 1)
dataset_log['Quantidade_Garangens'] = np.log(dataset_log['Quantidade_Garangens'] + 1)
dataset_log['Quantidade_Banheiros'] = np.log(dataset_log['Quantidade_Banheiros'] + 1)
ax = sns.distplot(dataset_log.Valor_Imovel)
ax.figure.set_size_inches(20, 6)
ax.set_title('Distribuição de Frequências', fontsize=20, loc = 'left')
ax = ax
sns.set_palette('Accent')
sns.set_style('darkgrid')
ax = sns.pairplot(dataset_log, x_vars= ['Area_m2','valor_Condominio', 'IPTU', 'Quantidade_Quartos',
                                    'Quantidade_Garangens', 'Quantidade_Banheiros'  ], y_vars='Valor_Imovel',
                                      height=5, kind = 'reg')
ax.fig.suptitle('Dispersão entre as Variáveis', fontsize=20, y=1.08)
ax.fig.set_size_inches(20, 6)
ax = ax
y = dataset_log['Valor_Imovel']
dataset.keys()
X = dataset_log[['Area_m2', 'Quantidade_Quartos', 'Quantidade_Banheiros', 'Quantidade_Garangens', 'valor_Condominio']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 500)
X_train_com_constante = sm.add_constant(X_train)
X_train_com_constante.head()
modelo_statsmodels = sm.OLS(y_train, X_train_com_constante, hasconst=True).fit()
print(modelo_statsmodels.summary())
X = dataset_log[['Area_m2', 'Quantidade_Garangens', 'valor_Condominio']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 500)
X_train_com_constante = sm.add_constant(X_train)
X_train_com_constante.head()
modelo_statsmodels = sm.OLS(y_train, X_train_com_constante, hasconst=True).fit()
print(modelo_statsmodels.summary())
modelo = LinearRegression()
modelo.fit(X_train, y_train)
print('R² = {:.03f}'.format(modelo.score(X_train, y_train)))
y_previsto = modelo.predict(X_test)
print('R² = {0:.3f}'.format(metrics.r2_score(y_test, y_previsto)))
entrada = X_test[5:6]
entrada
modelo.predict(entrada)[0]
np.exp(modelo.predict(entrada)[0]).round(2)
dataset.loc[62]['Valor_Imovel']
def simulador(Area, Garagens, Valor_Condominio):
    Area = np.log(Area)
    Garagens = np.log(Garagens)
    Valor_Condominio = np.log(Valor_Condominio)
    resultado = np.exp(modelo.predict([[Area, Garagens, Valor_Condominio ]])[0])
    print("R$ {:.2f}".format(resultado))
simulador(115, 4, 900)

