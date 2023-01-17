# Carregando a base de dados



# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



from sklearn.linear_model import LinearRegression

import statsmodels.api as sm

# fonte: # fonte: http://www.portalaction.com.br/analise-de-regressao/12-estimacao-dos-parametros-do-modelo





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

religion= pd.read_csv('/kaggle/input/world-religions/regional.csv')

religion1= pd.read_csv('/kaggle/input/world-religions/national.csv')

# Tratamento da base de dados das populações religiosas distribuídas pelas regiões mundiais



# 5 regiões



#Agrupar os grupos religiosos da base religion (regiões globais)



religion['outras'] = religion['zoroastrianism_all'] + religion['sikhism_all'] + religion['shinto_all'] + religion['baha’i_all'] + religion['jainism_all'] + religion['confucianism_all'] + religion['syncretism_all'] + religion['animism_all'] + religion['otherreligion_all']

religion['cristianismo'] = religion['christianity_all']

religion['judaismo'] = religion['judaism_all']

religion['islamismo'] = religion['islam_all']

religion['budismo'] = religion['buddhism_all']

religion['hinduismo'] = religion['hinduism_all']

religion['semreligiao'] = religion['noreligion_all']

religion['total'] = religion['religion_all']



#Criar as variáveis com as taxas populacionais regionais dos grupos religiosos

religion['cristianismo_pc'] = religion['cristianismo']/religion['population']

religion['judaismo_pc'] = religion['judaism_all']/religion['population']

religion['islamismo_pc'] = religion['islam_all']/religion['population']

religion['budismo_pc'] = religion['buddhism_all']/religion['population']

religion['hinduismo_pc'] = religion['hinduism_all']/religion['population']

religion['semreligiao_pc'] = religion['noreligion_all']/religion['population']

religion['total_pc'] = religion['religion_all']/religion['population']



#Criar as variáveis com as taxas populacionais dos grupos religiosos em relacão a população mundial



religion['cristianismo_wpc'] = religion['cristianismo']/religion['world_population']

religion['judaismo_wpc'] = religion['judaism_all']/religion['world_population']

religion['islamismo_wpc'] = religion['islam_all']/religion['world_population']

religion['budismo_wpc'] = religion['buddhism_all']/religion['world_population']

religion['hinduismo_wpc'] = religion['hinduism_all']/religion['world_population']

religion['semreligiao_wpc'] = religion['noreligion_all']/religion['world_population']

religion['total_wpc'] = religion['religion_all']/religion['world_population']
# Tratamento da base de dados das populações religiosas distribuídas pelos países



# 200 países



#Agrupar os grupos religiosos da base religion1 (paises)



religion1['outras'] = religion1['zoroastrianism_all'] + religion['sikhism_all'] + religion['shinto_all'] + religion['baha’i_all'] + religion['jainism_all'] + religion['confucianism_all'] + religion['syncretism_all'] + religion['animism_all'] + religion['otherreligion_all']

religion1['cristianismo'] = religion1['christianity_all']

religion1['judaismo'] = religion1['judaism_all']

religion1['islamismo'] = religion1['islam_all']

religion1['budismo'] = religion1['buddhism_all']

religion1['hinduismo'] = religion1['hinduism_all']

religion1['semreligiao'] = religion1['noreligion_all']

religion1['total'] = religion1['religion_all']



#Criar as variáveis com as taxas populacionais regionais dos grupos religiosos

religion1['cristianismo_pc'] = religion1['cristianismo']/religion1['population']

religion1['judaismo_pc'] = religion1['judaism_all']/religion1['population']

religion1['islamismo_pc'] = religion1['islam_all']/religion1['population']

religion1['budismo_pc'] = religion1['buddhism_all']/religion1['population']

religion1['hinduismo_pc'] = religion1['hinduism_all']/religion1['population']

religion1['semreligiao_pc'] = religion1['noreligion_all']/religion1['population']

religion1['total_pc'] = religion1['religion_all']/religion1['population']



#Criar as variáveis com as taxas populacionais dos grupos religiosos em relacão a população mundial



religion1['cristianismo_wpc'] = religion1['cristianismo']/religion['world_population']

religion1['judaismo_wpc'] = religion1['judaism_all']/religion['world_population']

religion1['islamismo_wpc'] = religion1['islam_all']/religion['world_population']

religion1['budismo_wpc'] = religion1['buddhism_all']/religion['world_population']

religion1['hinduismo_wpc'] = religion1['hinduism_all']/religion['world_population']

religion1['semreligiao_wpc'] = religion1['noreligion_all']/religion['world_population']

religion1['total_wpc'] = religion1['religion_all']/religion['world_population']
# Dropando as variaveis que foram não serão utilizadas nas bases religion (regiões mundiais) e religion1 (países)



religion.drop(columns=['christianity_protestant','christianity_romancatholic','christianity_easternorthodox','christianity_anglican','christianity_other','judaism_orthodox','judaism_conservative','judaism_reform','judaism_other','islam_sunni','islam_shi’a','islam_ibadhi','islam_nationofislam','islam_alawite','islam_ahmadiyya','islam_other','buddhism_mahayana','buddhism_theravada','buddhism_other','shinto_all','taoism_all','protestant_percent','romancatholic_percent','easternorthodox_percent','anglican_percent','otherchristianity_percent','christianity_percent','orthodox_percent','conservative_percent','reform_percent','otherjudaism_percent','judaism_percent','sunni_percent','shi’a_percent','ibadhi_percent','nationofislam_percent','alawite_percent','ahmadiyya_percent','otherislam_percent','islam_percent','mahayana_percent','theravada_percent','otherbuddhism_percent','buddhism_percent','zoroastrianism_percent','hinduism_percent','sikhism_percent','shinto_percent','baha’i_percent','taoism_percent','jainism_percent','confucianism_percent','syncretism_percent','animism_percent','noreligion_percent','otherreligion_percent','religion_sumpercent','total_percent','worldpopulation_percent'],inplace=True)

religion.drop(columns=['christianity_all','judaism_all','islam_all','buddhism_all','zoroastrianism_all','hinduism_all','sikhism_all','baha’i_all','jainism_all','confucianism_all','syncretism_all','animism_all','noreligion_all','otherreligion_all','religion_all'],inplace=True)



religion1.drop(columns=['christianity_protestant','christianity_romancatholic','christianity_easternorthodox','christianity_anglican','christianity_other','judaism_orthodox','judaism_conservative','judaism_reform','judaism_other','islam_sunni','islam_shi’a','islam_ibadhi','islam_nationofislam','islam_alawite','islam_ahmadiyya','islam_other','buddhism_mahayana','buddhism_theravada','buddhism_other','shinto_all','taoism_all','protestant_percent','romancatholic_percent','easternorthodox_percent','anglican_percent','otherchristianity_percent','christianity_percent','orthodox_percent','conservative_percent','reform_percent','otherjudaism_percent','judaism_percent','sunni_percent','shi’a_percent','ibadhi_percent','nationofislam_percent','alawite_percent','ahmadiyya_percent','otherislam_percent','islam_percent','mahayana_percent','theravada_percent','otherbuddhism_percent','buddhism_percent','zoroastrianism_percent','hinduism_percent','sikhism_percent','shinto_percent','baha’i_percent','taoism_percent','jainism_percent','confucianism_percent','syncretism_percent','animism_percent','noreligion_percent','otherreligion_percent','religion_sumpercent','total_percent'],inplace=True)

religion1.drop(columns=['christianity_all','judaism_all','islam_all','buddhism_all','zoroastrianism_all','hinduism_all','sikhism_all','baha’i_all','jainism_all','confucianism_all','syncretism_all','animism_all','noreligion_all','otherreligion_all','religion_all'],inplace=True)
# Criando as bases somente com os anos de 1980, 1990, 2000 e 2010



tx1 = religion[religion['year'] == 1980 ]

tx2= religion[religion['year'] == 1990 ]

tx3= religion[religion['year'] == 2000 ]

tx4 = religion[religion['year'] == 2010]



tx80= religion1[religion1['year'] == 1980 ]

tx90= religion1[religion1['year'] == 1990 ]

tx00= religion1[religion1['year'] == 2000 ]

tx10= religion1[religion1['year'] == 2010 ]



# Empilhando as bases com o corte de análise dos 4 anos (1980, 1990, 2000, 2010)

result = tx1.append([tx2, tx3,tx4])

result1 = tx80.append([tx90, tx00,tx10])
#bdcorr=religion1[religion1['year'] == 2010 ] 

temp1=religion1.groupby(['year', 'code'])['cristianismo'].sum()

temp2=religion1.groupby(['year', 'code'])['islamismo'].sum()

temp3=religion1.groupby(['year', 'code'])['outras'].sum()

temp4=religion1.groupby(['year', 'code'])['semreligiao'].sum()

temp5=religion1.groupby(['year', 'code'])['population'].sum()

temp6=religion1.groupby(['year', 'code'])['state'].sum()



bdcorr=pd.merge(temp1,temp2, on='code', how='left')

bdcorr=pd.merge(bdcorr,temp3, on='code', how='left')

bdcorr=pd.merge(bdcorr,temp4, on='code', how='left')

bdcorr=pd.merge(bdcorr,temp5, on='code', how='left')

#bdcorr=pd.merge(bdcorr,temp6, on='code', how='left')
# Base Religion (regiões globais)



religion.shape
religion.info()
religion.head().T
# Base Religion 1 (países)



religion1.shape
religion1.info()
religion1.head().T
religion1['state'].value_counts()
# Base result (regiões globais nos anos de 1990, 1980, 2000 e 2010)



result.shape
result.info()
result.head().T
# Base result1 (países nos anos de 1990, 1980, 2000 e 2010)



result1.shape
result1.info()
result1.head()
result1['state'].value_counts()
religion.groupby('year')['population'].describe()
religion.groupby('year')['world_population'].value_counts()
religion.groupby(['year', 'region'])['population'].sum()
# Gráfico 1. Evolução da população mundial de 1945 a 2010



plt.figure(figsize=(20,10))

sns.lineplot(x='year', y='world_population', data=religion)
# Gráfico 2. Evolução das populações nas regiões globais de 1945 a 2010



plt.figure(figsize=(20,10))

sns.lineplot(x='year', y='population', hue='region', data=religion)
# Gráfico 3. Distribuição da população mundial por regiões



#sns.set(font_scale=1)

sns.barplot(x='region',y='population',data=result,palette="Blues_d")

plt.xticks(rotation=25)

plt.ylabel('');

plt.xlabel('')

plt.tight_layout()
tx4.groupby(['year', 'region'])['population'].sum()/tx4['population'].sum()*100
# Gráfico 4. Distribuição populacional mundial por regiões globais (Box Plot)

sns.boxplot(y='population',x='region',data=religion)
# Gráfico 5. Evolução das populações nas regiões globais de 1945 a 2010



plt.figure(figsize=(10,10))

sns.lineplot(x='region',y='population',hue='year',data=religion)
religion[religion['region']=='Europe']['population']
# Gráfico 6. Evolução da distribuição populacional mundial por regiões globais de 1945 a 2010



plt.figure(figsize=(15,10))

sns.barplot(x = 'year', y='population', hue='region', data=religion.groupby(["year","region"]).sum().reset_index()).set_title('Evolução da População Mundial por Região: 1945 - 2010')

plt.xticks(rotation=90)

# Gráfico 7. A distribuição populacional mundial de 1945 e 2010

#plt.figure(figsize=(10,10))

#sns.distplot(religion[religion['year']== 1945]['population'].dropna(), label='1945')

#sns.distplot(religion[religion['year']== 2010]['population'].dropna(), label='2010')

#plt.legend()

# Gráfico 8.  

tx10_sum = pd.DataFrame(tx10['population'].groupby(tx10['state']).sum())

tx10_sum = tx10_sum.reset_index().sort_index(by='population',ascending=False)

most_cont = tx10_sum.head(8)

fig = plt.figure(figsize=(20,10))

plt.title('Países de menores .')

sns.set(font_scale=2)

sns.barplot(y='population',x='state',data=most_cont,palette="Blues_d")

plt.xticks(rotation=45)

plt.ylabel('');

plt.xlabel('')

plt.tight_layout()
tx10.nlargest(5,'population')
# Gráfico 9.  

tx10_sum = pd.DataFrame(tx10['population'].groupby(tx10['state']).sum())

tx10_sum = tx10_sum.reset_index().sort_index(by='population',ascending=True)

most_cont = tx10_sum.head(8)

fig = plt.figure(figsize=(20,10))

plt.title('População dos paises de menor população mundial')

sns.set(font_scale=2)

sns.barplot(y='population',x='state',data=most_cont,palette="Blues_d")

plt.xticks(rotation=45)

plt.ylabel('');

plt.xlabel('')

plt.tight_layout()
tx10.nsmallest(5,'population')
df=tx4.groupby(['cristianismo', 'judaismo','islamismo', 'budismo', 'hinduismo', 'outras', 'semreligiao']).sum()

df
# Total de fiéis de cada religião em 2010



tx4['cristianismo'].sum()

tx4['judaismo'].sum()

tx4['islamismo'].sum()

tx4['budismo'].sum()

tx4['hinduismo'].sum()

tx4['outras'].sum()

tx4['semreligiao'].sum()
# Gráfico 10. Total de fiéis de cada religião por região global em 2010

df=tx4.groupby(['cristianismo', 'judaismo','islamismo', 'budismo', 'hinduismo', 'outras', 'semreligiao'])['region'].sum()

df
# Dúvida em como plotar esse gráfico em Python. 



# Seria criar um dataframe a partir de uma tabela pivot e groupby (sum) da base tx4 (2010) para transformar as variáveis de cada religião em linhas e as regiões em colunas?



#Determine a tabela pivot

#pivot = religion.pivot_table(values=['cristianismo', 'judaismo', 'islamismo', 'budismo', 'hinduismo', 'outras', 'semreligiao'], index=['year', 'region'], aggfunc=np.sum)

# print(pivot)



#df1 = pd.pivot_table(religion,

#   index=['year'],

#   values=['budismo', 'hinduismo', 'outras', 'semreligiao'],

#   columns=['region'],

#   fill_value=''

#  )



# Dúvida em como plotar esse gráfico em Python. 



# Seria criar um dataframe a partir de uma tabela pivot e groupby (sum) da base tx4 (2010) para transformar as variáveis de cada religião em linhas e as regiões em colunas?
religion.groupby('year')['cristianismo'].sum()
# Crescimento da Percentagem da população cristianismo 

tx1['cristianismo'].sum()/tx1['population'].sum() #1980

tx2['cristianismo'].sum()/tx2['population'].sum() #1990

tx3['cristianismo'].sum()/tx3['population'].sum() #2000

tx4['cristianismo'].sum()/tx4['population'].sum() #2010
# Gráfico 11

plt.figure(figsize=(20,10))

sns.lineplot(x='year', y='cristianismo', data=religion)
# Gráfico 12

sns.set(font_scale=1)

sns.barplot(x='region',y='cristianismo',data=religion,palette="Blues_d")

plt.xticks(rotation=25)

plt.ylabel('');

plt.xlabel('')

plt.tight_layout()
#Gráfico 13

plt.figure(figsize=(9,6))

sns.distplot(religion[religion['year']== 1945]['cristianismo'].dropna(), label='1945')

sns.distplot(religion[religion['year']== 2010]['cristianismo'].dropna(), label='2010')

plt.legend()
religion.groupby('year')['cristianismo'].value_counts()
religion.groupby('region')['cristianismo'].describe()
religion.groupby('region')['cristianismo'].median()
# Gráfico 14

plt.figure(figsize=(20,6))

sns.lineplot(x='year', y='cristianismo_pc', data=religion)
religion.groupby('region')['cristianismo_pc'].describe()
religion.groupby('region')['cristianismo'].median()
#Gráfico 15

plt.figure(figsize=(9,6))

sns.distplot(religion[religion['year']== 1945]['cristianismo_pc'].dropna(), label='1945')

sns.distplot(religion[religion['year']== 2010]['cristianismo_pc'].dropna(), label='2010')

plt.legend()
religion.groupby('region')['cristianismo_pc'].median()
# Gráfico 16

sns.set(font_scale=1)

sns.barplot(x='region',y='cristianismo_pc',data=religion,palette="Blues_d")

plt.xticks(rotation=25)

plt.ylabel('');

plt.xlabel('')

plt.tight_layout()
religion.groupby('region')['cristianismo_pc'].mean()
religion.groupby('region')['cristianismo'].mean()/religion.groupby('region')['cristianismo'].mean().sum()
# Gráfico 17 - Distribuião da população cristã em 2010

plt.figure(figsize=(9,6))

sns.barplot(x = 'year', y='cristianismo', hue='region', data=tx4.groupby(["year","region"]).sum().reset_index()).set_title('População Cristianismo por região global - 2010')

plt.xticks(rotation=90)
tx4.groupby('region')['cristianismo'].sum()
#religion[religion['cristianismo']== religion['cristianismo'].max()]
tx4.groupby('region')['cristianismo'].sum()/tx4.groupby('region')['population'].sum()
# Gráfico 18 - Box Plot da população cristã de 1945 a 2010

# 1945 a 2010

#plt.figure(figsize=(9,6))

#sns.boxplot(y='cristianismo',x='region',data=religion)

plt.figure(figsize=(9,10))

sns.boxplot(x='region', y='cristianismo', hue=None, data=religion, order=None, hue_order=None, orient=None, color=None, palette=None, saturation=0.75, width=0.8, dodge=True, fliersize=5, linewidth=None, whis=1.5)
# Em 2010

tx10.nlargest(5,'cristianismo')
# Gráfico 19

tx10_sum = pd.DataFrame(tx10['cristianismo'].groupby(tx10['state']).sum())

tx10_sum = tx10_sum.reset_index().sort_index(by='cristianismo',ascending=False)

most_cont = tx10_sum.head(8)

fig = plt.figure(figsize=(20,10))

plt.title('População Cristianismo x paises.')

sns.set(font_scale=2)

sns.barplot(y='cristianismo',x='state',data=most_cont,palette="Blues_d")

plt.xticks(rotation=45)

plt.ylabel('');

plt.xlabel('')

plt.tight_layout()
tx10.nsmallest(5,'cristianismo')
tx4.nsmallest(5,'cristianismo')
# Estatística populacionais das religiões

df=tx4.groupby(['cristianismo', 'judaismo','islamismo', 'budismo', 'hinduismo', 'outras', 'semreligiao'])['region'].sum()

df
# Gráfico 20

tx10_sum = pd.DataFrame(tx10['cristianismo'].groupby(tx10['state']).sum())

tx10_sum = tx10_sum.reset_index().sort_index(by='cristianismo',ascending=True)

most_cont = tx10_sum.head(8)

fig = plt.figure(figsize=(20,10))

plt.title('População Cristianismo x paises.')

sns.set(font_scale=2)

sns.barplot(y='cristianismo',x='state',data=most_cont,palette="Blues_d")

plt.xticks(rotation=45)

plt.ylabel('');

plt.xlabel('')

plt.tight_layout()
# Gráfico 21

plt.figure(figsize=(16,6))

sns.lineplot(x='year', y='cristianismo', hue='region', data=religion)
# Gráfico 22

plt.figure(figsize=(16,6))

sns.lineplot(x='year', y='cristianismo_pc', hue='region', data=religion)
# Gráfico 23



#sns.barplot(x='region', y='cristianismo', data=result, hue='year')



g = sns.catplot(x="region", y="cristianismo", hue="year", data=result,

                height=10, kind="bar", palette="muted")

g.despine(left=True)

g.set_ylabels("Cristianimo (pop.)")
# Gráfico 24

plt.figure(figsize=(13,10))

sns.barplot(x='region', y='cristianismo_pc', data=result, hue='year')
# Gráfico 25 - Correlação entre o cristianismo, o islamismo, as outras religões e os sem religião.



f, ax = plt.subplots(figsize=(10,8))

sns.heatmap(bdcorr.corr(), vmin=-1, vmax=1,ax=ax,cmap='coolwarm',fmt='.2f',

            annot=True)



# Gráfico 26 

sns.set(font_scale=1)

sns.barplot(x='region',y='outras',data=religion,palette="Blues_d")

plt.xticks(rotation=25)

plt.ylabel('');

plt.xlabel('')

plt.tight_layout()
# Gráfico 27

sns.set(font_scale=1)

sns.barplot(x='region',y='semreligiao',data=religion,palette="Blues_d")

plt.xticks(rotation=25)

plt.ylabel('');

plt.xlabel('')

plt.tight_layout()
(1345174272+1195000000+312750000+239960000+190755800)/tx10['population'].sum()
tx10['population'].sum()
# Análise de regresão necessita de outra base (human rights)



#joint das bases result e result1 com a base human rights



# result3 = pd.merge(religion1,hrights, on='region', how='left')
# importando as libs

import numpy as np

import pandas as pd

from sklearn.linear_model import LinearRegression

import statsmodels.api as sm
# é necessário adicionar uma constante a matriz X

#population_sm = sm.add_constant("population")

# OLS vem de Ordinary Least Squares e o método fit irá treinar o modelo

#results = sm.OLS("cristianismo", population_sm).fit()

# mostrando as estatísticas do modelo

#results.summary()

# mostrando as previsões para o mesmo conjunto passado

#results.predict(population_sm)
#x_v = religion[['population']]#

#y_v = religion[['cristianismo']]

# criando e treinando o modelo

#model = LinearRegression()

#model.fit(x_v, y_v)

# para visualizar os coeficientes encontrados

#model.coef_

# para visualizar o R²

#model.score()

# mostrando as previsões para o mesmo conjunto passado

#model.predict(X_sm)
#religion['gr_religiosos'] = 0

#if religion['gcristianismo'] >0:

#    religion['gr_religiosos'] = 'cristianismo'

#elif religion['gjudaismo'] 
sns.set(font_scale=1)

sns.barplot(x='region',y='cristianismo',data=religion,palette="Blues_d")

plt.xticks(rotation=25)

plt.ylabel('');

plt.xlabel('')

plt.tight_layout()
sns.set(font_scale=1)

sns.barplot(x='region',y='islamismo',data=religion,palette="Blues_d")

plt.xticks(rotation=25)

plt.ylabel('');

plt.xlabel('')

plt.tight_layout()