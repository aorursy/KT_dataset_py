import numpy as np 
import pandas as pd 
import os, datetime
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
%matplotlib inline
covid = pd.read_csv('../input/corona-virus-brazil/brazil_covid19.csv') # dados do covid dos estados
estados = pd.read_csv('../input/pip-e-populacao-estados/estados.csv',sep=';') # pib dos estados
brasil = pd.read_csv('../input/corona-virus-brazil/brazil_covid19_macro.csv') # dados do covid, macro
today = datetime.date.today()
print("Todos os dados estão atualizados até:", today)
covid.head()
estados.head()
brasil.head()
estados.rename(columns={'Estado':'state'}, inplace=True)
covid = pd.merge(covid,estados,how='inner',on='state')
covid.rename(columns={'População TCU 2019':"Populacao"},inplace=True)
brasil['letalidade'] = brasil['deaths']/brasil['cases'] * 100
brasil['date'] = pd.to_datetime(brasil['date'])
brasil['date'] = brasil['date'].dt.date
covid['morte por milhão'] = covid['deaths'] / covid['Populacao'] * 1000000
covid['date'] = pd.to_datetime(covid['date'])
covid['date'] = covid['date'].dt.date
covid['letalidade'] = covid['deaths'] / covid['cases'] * 100
covid.head()
casos_estados = covid[covid['date']==max(covid['date'])].sort_values('deaths', ascending=False)
casos_estados10 = casos_estados.head(10)
plt.figure(figsize=(15,15))
plt.title('Estados com mais mortes')
sns.barplot(y=casos_estados['Nome Estado'],x=casos_estados['deaths'])
plt.ylabel('Estados')
plt.xlabel('Nº de Mortes')
# Função para fazer o label na coluna
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = int(rect.get_height())
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2.5, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
labels = casos_estados10['state']
casos=casos_estados10['cases']
mortes=casos_estados10['deaths']
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(15,8))
rects1 = ax.bar(x - width/2, casos, width, label='Casos')
rects2 = ax.bar(x + width/2, mortes, width, label='Mortes')

ax.set_ylabel('Confirmados')
ax.set_title('Estados com mais mortes')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(['Casos    ', 'Mortes     '])

ax1 = ax.twinx()
ax1.plot('state','letalidade',data=casos_estados10,marker='8', color='red')
ax1.legend(loc='upper right', bbox_to_anchor=(1, 0.92))
ax1.set_ylabel('Letalidade')

autolabel(rects1)
autolabel(rects2)
# Definição de variáveis, para facilitar criei uma variável para cada estado que utilizarei depois
SP = covid[covid['Nome Estado']=='São Paulo']
RJ = covid[covid['Nome Estado']=='Rio de Janeiro']
CE = covid[covid['Nome Estado']=='Ceará']
PA = covid[covid['Nome Estado']=='Pará']
MG = covid[covid['Nome Estado']=='Minas Gerais']
PE = covid[covid['Nome Estado']=='Pernambuco']
MS = covid[covid['Nome Estado']=='Mato Grosso do Sul']
AM = covid[covid['Nome Estado']=='Amazonas']
lista = [n for n in range(len((PA['state'])))]
fig, ax = plt.subplots(figsize=(12,7))
plt.title('Evolução dos casos nos estados com mais mortes')
sns.lineplot(x=SP['date'],y=SP['cases'],label='São Paulo')
sns.lineplot(x=RJ['date'],y=RJ['cases'], label='Rio de Janeiro')
sns.lineplot(x=CE['date'],y=CE['cases'], label='Ceará')
sns.lineplot(x=PA['date'],y=PA['cases'], label='Pará')
# sns.lineplot(x=lista,y=brasil['cases'][:len(lista)], label='Brasil')
plt.xlabel('Data')

ax.set_ylim(ymin=10**0)
ax.set_yscale('log')

fig, ax = plt.subplots(figsize=(12,7))
plt.title('Evolução dos casos nos estados com mais mortes')
sns.lineplot(x=lista,y=SP['cases'][:len(lista)],label='São Paulo')
sns.lineplot(x=lista,y=RJ['cases'][:len(lista)], label='Rio de Janeiro')
sns.lineplot(x=lista,y=CE['cases'][:len(lista)], label='Ceará')
sns.lineplot(x=lista,y=PA['cases'][:len(lista)], label='Pará')
# sns.lineplot(x=lista,y=brasil['cases'][:len(lista)], label='Brasil')

plt.xlabel('Dias')
plt.ylabel('Casos')
ax.set_xlim(xmin=0)
ax.set_ylim(ymin=10**0)
ax.set_yscale('log')

fig,ax = plt.subplots(figsize=(12,7))
plt.title('Letalidade nos estados com mais mortes')
sns.lineplot(x=SP['date'],y=SP['letalidade'],label='São Paulo')
sns.lineplot(x=RJ['date'],y=RJ['letalidade'],label='Rio de Janeiro')
sns.lineplot(x=CE['date'],y=CE['letalidade'],label='Ceará')
sns.lineplot(x=PA['date'],y=PA['letalidade'],label='Pará')
sns.lineplot(x=brasil['date'], y=brasil['letalidade'], label='Brasil')
# sns.lineplot(x=brasil['date'],y=brasil['letalidade'], label='Brasil')
ax.set_xlim(xmin=brasil[brasil['deaths']==1]['date'])
ax.set_ylabel('Letalidade')
ax.set_xlabel('Data')
fig,ax = plt.subplots(figsize=(12,7))
plt.title('Morte por milhão nos estados com mais mortes')
sns.lineplot(x=SP['date'],y=SP['morte por milhão'],label='São Paulo')
sns.lineplot(x=RJ['date'],y=RJ['morte por milhão'],label='Rio de Janeiro')
sns.lineplot(x=CE['date'],y=CE['morte por milhão'],label='Ceará')
sns.lineplot(x=PA['date'],y=PA['morte por milhão'],label='Pará')

ax.set_xlim(xmin=brasil[brasil['deaths']==1]['date'])
# quais estados tem maior taxa de morte por milhão?
full_data = casos_estados.sort_values(by='morte por milhão',ascending = False)
full_data[:10]
fig,ax = plt.subplots(figsize=(12,7))
plt.title('Morte por milhão de habitantes')
sns.lineplot(x=SP['date'],y=SP['morte por milhão'],label='São Paulo')
sns.lineplot(x=RJ['date'],y=RJ['morte por milhão'],label='Rio de Janeiro')
sns.lineplot(x=CE['date'],y=CE['morte por milhão'],label='Ceará')
sns.lineplot(x=PA['date'],y=PA['morte por milhão'],label='Pará')
sns.lineplot(x=PE['date'],y=PE['morte por milhão'],label='Pernambuco')
sns.lineplot(x=AM['date'],y=AM['morte por milhão'],label='Amazonas')

ax.set_xlabel('Data')


ax.set_xlim(xmin=brasil[brasil['deaths']==1]['date'])
sns.pairplot(full_data, corner=True)
correlation = full_data.corr()
plt.figure(figsize=(8,8))
sns.heatmap(correlation, annot=True)

correlation
correlation['PIB Per Capita'].sort_values(ascending=False)
full_data
# tratamento de dados
brasilgpd = gpd.read_file("../input/majson/data/Brasil.json")
brasilgpd.rename(columns={'ESTADO':'Nome Estado'}, inplace=True)
brasil_total = pd.merge(brasilgpd,full_data,how='inner',on='Nome Estado')
brasil_total.drop('date', axis=1, inplace=True)
brasilgpd.head()
fig , ax = plt.subplots(figsize=(11,11))
plt.title('Morte por Milhão de Habitantes')
brasil_total.plot(column = 'morte por milhão',ax = ax, legend=True,
             legend_kwds={'label':'Morte por milhão', 'orientation':'horizontal'},
             cmap='OrRd')

fig , ax = plt.subplots(figsize=(11,11))
plt.title('PIB Per Capita')
brasil_total.plot(column = 'PIB Per Capita',ax = ax, legend=True,
             legend_kwds={'label':'PIB Per Capita (R$)', 'orientation':'horizontal'},
             cmap='OrRd', )
ax.set_xticklabels= ({'teste'})

fig , ax = plt.subplots(figsize=(11,11))
plt.title('Casos')
brasil_total.plot(column = 'cases',ax = ax, legend=True,
             legend_kwds={'label':'Casos', 'orientation':'horizontal'},
             cmap='OrRd', )

fig , ax = plt.subplots(figsize=(11,11))
plt.title('Letalidade')
brasil_total.plot(column = 'letalidade',ax = ax, legend=True,
             legend_kwds={'label':'letalidade', 'orientation':'horizontal'},
             cmap='OrRd', )


# tratamento de dados, início sobre as cidades
cities = pd.read_csv("../input/corona-virus-brazil/brazil_covid19_cities.csv")
population_cities = pd.read_csv("../input/corona-virus-brazil/brazil_population_2019.csv",error_bad_lines=False)

cities['date'] = pd.to_datetime(cities['date'])
cities['date'] = cities['date'].dt.date
population_cities.rename(columns={'city':'name'}, inplace=True)
population_cities.drop('state', axis=1, inplace=True)
cities.rename(columns={'code':'city_code'}, inplace=True)
cities.drop('name', axis=1, inplace=True)
cities = pd.merge(cities,population_cities, how='inner', on='city_code')
cities.set_index('city_code',inplace=True)
cities['Morte 100 mil'] = (cities['deaths']) / (cities['population']) * 100000
cities['letalidade'] = (cities['deaths']) / (cities['cases']) * 100
city = cities[cities['date']==max(cities['date'])]
city = city[city['state']=='SP']
city.drop(['state_code','region','state','date','health_region_code'], axis=1,inplace=True)
city_top_50 = city.sort_values(by='deaths', ascending=False).head(50)
city_top_50 = city_top_50.sort_values(by='Morte 100 mil', ascending=False)
plt.figure(figsize=(15,15))
plt.title('Cidades com maior taxa de morte p/ 100 mil habitantes')
sns.barplot(x='Morte 100 mil',y='name',data=city_top_50)
plt.ylabel('Cidades')
plt.xlabel('Morte p/ 100 mil habitantes')
city10 = city.sort_values(by='deaths', ascending=False).head(10)
labels = city10['name']
casos=city10['cases']
mortes=city10['deaths']
x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(20,8))
rects1 = ax.bar(x - width/2, casos, width, label='Casos')
rects2 = ax.bar(x + width/2, mortes, width, label='Mortes')
ax.legend()
ax.set_ylabel('Casos confirmados escala em log*')
ax.set_title('Cidades com mais mortes')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_yscale('log')

autolabel(rects1)
autolabel(rects2)

sns.jointplot(x='population',y='cases',data=city, kind='reg',dropna=True, xlim=(-0,500000), ylim=(-0,1700), height=8)
print(city['population'].corr(city['cases']))
# definição das variáveis , cidades da região
sp = cities[cities['state']=='SP']
rio_claro = sp[sp['name']== 'Rio Claro']
piracicaba = sp[sp['name']== 'Piracicaba']
araras = sp[sp['name']== 'Araras']
limeira = sp[sp['name']=='Limeira']

leme = sp[sp['name']=='Leme']
sao_carlos = sp[sp['name'] == 'São Carlos']
araraquara = sp[sp['name']== 'Araraquara']

dstartcases = datetime.datetime(2020, 3, 28)
dstartdeaths = datetime.datetime(2020, 4,7)
cities.head()
fig, ax = plt.subplots(2,2,figsize=(25,14))

ax[0,0].set_title('Casos de Covid-19 na Região')
ax[0,0] = sns.lineplot(x='date', y='cases', data=rio_claro,label='Rio Claro', ax=ax[0,0])
ax[0,0] = sns.lineplot(x='date', y='cases', data=piracicaba, label='Piracicaba', ax=ax[0,0])
ax[0,0] = sns.lineplot(x='date', y='cases', data=araras, label='Araras', ax=ax[0,0])
ax[0,0] = sns.lineplot(x='date', y='cases', data=limeira, label='Limeira', ax=ax[0,0])
ax[0,0].set_xlim(xmin=dstartcases)
ax[0,0].set_ylabel('Casos')
ax[0,0].set_xlabel('Data')
# plt.ylabel('Casos Confirmados')

ax[0,1].set_title('Mortes confirmadas pelo Covid-19 na Região')
ax[0,1] = sns.lineplot(x='date', y='deaths', data=rio_claro,label='Rio Claro', ax=ax[0,1])
ax[0,1] = sns.lineplot(x='date', y='deaths', data=piracicaba, label='Piracicaba', ax=ax[0,1])
ax[0,1] = sns.lineplot(x='date', y='deaths', data=araras, label='Araras',ax=ax[0,1])
ax[0,1] = sns.lineplot(x='date', y='deaths', data=limeira, label='Limeira',ax=ax[0,1])
ax[0,1].set_xlim(xmin=dstartdeaths)
ax[0,1].set_ylabel('Mortes')
ax[0,1].set_xlabel('Data')


ax[1,0].set_title('Mortes por 100 mil habitantes')
ax[1,0] = sns.lineplot(x='date', y='Morte 100 mil', data=rio_claro,label='Rio Claro', ax=ax[1,0])
ax[1,0] = sns.lineplot(x='date', y='Morte 100 mil', data=piracicaba, label='Piracicaba', ax=ax[1,0])
ax[1,0] = sns.lineplot(x='date', y='Morte 100 mil', data=araras, label='Araras',ax=ax[1,0])
ax[1,0] = sns.lineplot(x='date', y='Morte 100 mil', data=limeira, label='Limeira',ax=ax[1,0])
ax[1,0].set_xlim(xmin=dstartdeaths)
ax[1,0].set_ylabel('Mortes')
ax[1,0].set_xlabel('Data')

ax[1,1].set_title('Letalidade')
ax[1,1] = sns.lineplot(x='date', y='letalidade', data=rio_claro,label='Rio Claro', ax=ax[1,1])
ax[1,1] = sns.lineplot(x='date', y='letalidade', data=piracicaba, label='Piracicaba', ax=ax[1,1])
ax[1,1]= sns.lineplot(x='date', y='letalidade', data=araras, label='Araras',ax=ax[1,1])
ax[1,1] = sns.lineplot(x='date', y='letalidade', data=limeira, label='Limeira',ax=ax[1,1])
ax[1,1].set_xlim(xmin=dstartdeaths)
ax[1,1].set_ylabel('Letalidade em %')
ax[1,1].set_xlabel('Data')
fig, ax = plt.subplots(2,2,figsize=(25,14))

ax[0,0].set_title('Casos de Covid-19 na Região')
ax[0,0] = sns.lineplot(x='date', y='cases', data=rio_claro,label='Rio Claro', ax=ax[0,0])
ax[0,0] = sns.lineplot(x='date', y='cases', data=leme, label='Leme', ax=ax[0,0])
ax[0,0] = sns.lineplot(x='date', y='cases', data=araraquara, label='Araraquara', ax=ax[0,0])
ax[0,0] = sns.lineplot(x='date', y='cases', data=sao_carlos, label='São Carlos', ax=ax[0,0])
ax[0,0].set_xlim(xmin=dstartcases)
ax[0,0].set_ylabel('Casos')
ax[0,0].set_xlabel('Data')
# plt.ylabel('Casos Confirmados')

ax[0,1].set_title('Mortes confirmadas pelo Covid-19 na Região')
ax[0,1] = sns.lineplot(x='date', y='deaths', data=rio_claro,label='Rio Claro', ax=ax[0,1])
ax[0,1] = sns.lineplot(x='date', y='deaths', data=leme, label='Leme', ax=ax[0,1])
ax[0,1] = sns.lineplot(x='date', y='deaths', data=araraquara, label='Araraquara',ax=ax[0,1])
ax[0,1] = sns.lineplot(x='date', y='deaths', data=sao_carlos, label='São Carlos',ax=ax[0,1])
ax[0,1].set_xlim(xmin=dstartdeaths)
ax[0,1].set_ylabel('Mortes')
ax[0,1].set_xlabel('Data')


ax[1,0].set_title('Mortes por 100 mil habitantes')
ax[1,0] = sns.lineplot(x='date', y='Morte 100 mil', data=rio_claro,label='Rio Claro', ax=ax[1,0])
ax[1,0] = sns.lineplot(x='date', y='Morte 100 mil', data=leme, label='Leme', ax=ax[1,0])
ax[1,0] = sns.lineplot(x='date', y='Morte 100 mil', data=araraquara, label='Araraquara',ax=ax[1,0])
ax[1,0] = sns.lineplot(x='date', y='Morte 100 mil', data=sao_carlos, label='São Carlos',ax=ax[1,0])
ax[1,0].set_xlim(xmin=dstartdeaths)
ax[1,0].set_ylabel('Mortes')
ax[1,0].set_xlabel('Data')

ax[1,1].set_title('Letalidade')
ax[1,1] = sns.lineplot(x='date', y='letalidade', data=rio_claro,label='Rio Claro', ax=ax[1,1])
ax[1,1] = sns.lineplot(x='date', y='letalidade', data=leme, label='Leme', ax=ax[1,1])
ax[1,1]= sns.lineplot(x='date', y='letalidade', data=araraquara, label='Araraquara',ax=ax[1,1])
ax[1,1] = sns.lineplot(x='date', y='letalidade', data=sao_carlos, label='São Carlos',ax=ax[1,1])
ax[1,1].set_xlim(xmin=dstartdeaths)
ax[1,1].set_ylabel('Letalidade em %')
ax[1,1].set_xlabel('Data')