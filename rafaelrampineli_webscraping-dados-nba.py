# Versão da Linguagem Python
from platform import python_version
print('Versão Python:', python_version())
# Instala o pacote watermark. 
# Esse pacote é usado para gravar as versões de outros pacotes usados neste jupyter notebook.
!pip install -q -U watermark
# Pacote Pingouin
# Pacote para analise estatisticas
!pip install -q -U pingouin
# Imports

# Imports para Web Scraping
import bs4
import csv 
import requests 
from bs4 import BeautifulSoup

# Imports para manipulação, visualização e análise de dados
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pingouin as pg
import numpy as np
# Versões dos pacotes usados neste jupyter notebook
%reload_ext watermark
%watermark --iversions
# Etapa 1: Carregar os dados direto do website.
website = requests.get("https://www.basketball-reference.com/leagues/NBA_2020_per_game.html")
# Validando a conexao com a url {200} = OK
website.status_code
# Visualizando todo o documento extraído do website
website.text
# Obtendo o Código HTML da Página através da biblioteca BeatufilSoap
website_html = BeautifulSoup(website.text, 'html.parser')
#website_html
dados_extraidos = website_html.find("div", {"class": "overthrow table_container"})
#dados_extraidos
# A partir dos dados extraidos, queremos obter o cabeçalho da tabela. Cabeçalho esse que contém os nomes das colunas.
# Esses dados se encontram na TAG THEAD
header = dados_extraidos.find("thead")
header_elements = header.find_all("th")
# Para cada registro na lista, extraía somente os texto, que é onde está o nome das colunas. [1:] para ignorar a coluna Rk
header_elements = [head.text for head in header_elements[1:]]
# Salvando todo o resultado da extração em uma lista
full_data = []
full_data.append(header_elements)
full_data
# Obtendo as informações contidas nos dados extraídos.
Line_elements = dados_extraidos.find_all("tr", {"class": "full_table"})
# Extrai somente as informações da tag td e obtem o texto.  No fim, adiciona o resultado dentro de uma lista.
full_elements = []
for row in Line_elements:
    data_elements = row.find_all("td")
    data_elements = [data.text for data in data_elements]
    full_elements.append(data_elements)
# Transforma as 2 listas em dataframe
df = pd.DataFrame(full_data + full_elements)
# Renomeia o cabeçalho para os registros da posição 0
df = df.rename(columns=df.iloc[0])
# Remove a linha de posição 0
df = df.drop(df.index[0])
# Salva os dados obtidos em um arquivo CSV
#df.to_csv('/kaggle/input/scraping-nba/scraping_nba.csv', index=False, header=True)
# Carrega o arquivo csv com o conteúdo do web scraping
df_nba = pd.read_csv('/kaggle/input/scraping-nba/scraping_nba.csv')
# Shape
df_nba.shape
# Visualiza uma amostra dos dados
df_nba.head(10)
df_nba['Age'].mean()
df_nba['Age'].hist()
# hist = Plota a distribuição do histograma
# kde = plota a linha de densidade estimada
# rug = plota um gráfico de linhas ao pé do eixo de suporte
# hist_kwd = cor da divisão das linhas no histograma
ax = sns.distplot(df_nba['Age'], hist = True, kde = True, rug = False, color = 'blue', bins = 25, hist_kws = {'edgecolor':'black'})
plt.show()
# Teste de normalidade com Pingouin
# Valor alpha default: 0.05
pg.normality(df_nba['Age'])
# reset_index é utilizado para que a coluna Player não seja categorizada como INDEX do resultado.
df_nba_pts = df_nba.groupby(['Player'])['PTS'].sum().reset_index()
df_nba_pts
# Retorna o TOP 10 maiores registros na coluna PTS
df_nba_pts.nlargest(10, 'PTS')
df_nba_35Age = df_nba[(df_nba['Age'] >= 35) & (df_nba['GS'] > 0)]
df_nba_35Age[['Player','Age']]
plt.figure(figsize=[10,10])
plt.title('\nMinutos Jogados x Rebotes Ofensivos\n', fontsize = 20)
ax = sns.regplot(x = df_nba['MP'], y = df_nba['ORB'], marker = '+')
ax.set_xlabel('Minutos Jogados', fontsize=15)
ax.set_ylabel('Número Rebotes Ofensivos', fontsize=15);
plt.figure(figsize=[10,10])
plt.title('\nMinutos Jogados x Rebotes Defensivos\n', fontsize = 20)
ax = sns.regplot(x = df_nba['MP'], y = df_nba['DRB'], marker = '+')
ax.set_xlabel('Minutos Jogados', fontsize=15)
ax.set_ylabel('Número Rebotes Defensivos', fontsize=15);
# Obtendo somente os dados de tempo jogado e rebote ofensivo 
df_preditcion = df_nba[['MP', 'ORB']]
df_preditcion
# Importação dos pocates para dividir os dados em treino e teste, pacote do modelo de regressão e o pacote para avaliar o resultado.
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics
%matplotlib inline
df_preditcion.shape
(df_preditcion['MP'].values).shape
# Necessário aplicar a conversão utilizando o reshape para que as informações sejam representadas com 2 dimensões (linhaxcoluna)
x = df_preditcion['MP'].values.reshape(-1, 1)
target = df_preditcion['ORB'].values
x.shape
x_train, x_test, y_train, y_test = train_test_split(x, target, test_size=0.3)
regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
# flatten() : retorna os dados em uma única dimensão (semelhante ao ravel)
df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
df
df1 = df.head(25)
df1.plot(kind='bar',figsize=(16,10))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()
import numpy as np

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))