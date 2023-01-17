# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns

import folium

from folium.plugins import HeatMap

from wordcloud import WordCloud

%matplotlib inline



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/crime-incident-reports-august-2015-to-date/tmpkrh6apk1.csv')

df_raw = df.copy()
# Primeiro contato de fato com a base, olhando brevemente as colunas e seus respectivos valores

df.sample(5).T
# Informações sobre as colunas

df.info()
total_missing_data = df.isnull().sum().sort_values(ascending=False)

percent_missing_data = np.round(df.isnull().sum() / df.isnull().count().sort_values(ascending=False) * 100, 2)

missing_data = pd.concat([total_missing_data, percent_missing_data], axis=1, keys=['Total', 'Porcentagem'], sort=False)

display(missing_data)
#Plotando gráfico que mostra a diferença na quantidade entre os campos com valores faltando e os campos com valores preenchidos



shooting_missing_values = total_missing_data['SHOOTING']

shooting_non_missing_values = df['SHOOTING'].value_counts()[0]



f, ax = plt.subplots(figsize=(6,10))

sns.barplot(x=['Valores Ausentes', 'Valores Preenchidos'], y=[shooting_missing_values, shooting_non_missing_values])

plt.title('Quantidade de missing values x Quantidade de valores preenchidos.')

plt.show()
# Removendo a coluna SHOOTING do datafram

df.drop('SHOOTING', axis=1, inplace=True)
# Uma amostra do dataframe após a remoção da coluna SHOOTING

df.sample(10).T
total_missing_data = df.isnull().sum().sort_values(ascending=False)

percent_missing_data = np.round(df.isnull().sum() / df.isnull().count().sort_values(ascending=False) * 100, 2)

missing_data = pd.concat([total_missing_data, percent_missing_data], axis=1, keys=['Total', 'Porcentagem'], sort=False)

display(missing_data)
df[df['Location'] == '(0.00000000, 0.00000000)'].sample(10).T
# Quantidade de registros com valores zerados na variável location e a quantiade de valores que estão faltando nas variáveis Lat e Long

display(df[df['Location'] == '(0.00000000, 0.00000000)'].count()['Location'])

display(missing_data[1:3])
df[df['Location'] == '(-1.00000000, -1.00000000)'].sample(10).T
# Total de valores faltosos (0.00000000, 0.00000000) em relação a quantidade geral de registros por tipo de ocorrido.

ocg_with_locat0 = df[df['Location'] == '(0.00000000, 0.00000000)']['OFFENSE_CODE_GROUP'].value_counts().sort_values(ascending=False)

percent_ocg_with_locat0 = ocg_with_locat0 / df['OFFENSE_CODE_GROUP'].value_counts().sort_values(ascending=False)



df_ocg_with_locat0 = pd.concat([ocg_with_locat0, percent_ocg_with_locat0 * 100], axis=1, keys=['Total', 'Porcentagem'], sort=True)



display(df_ocg_with_locat0.sort_values('Porcentagem', ascending=False).head(15))
# Total de valores faltosos (-1.00000000, -1.00000000) em relação a quantidade geral de registros por tipo de ocorrido.

ocg_with_locatm1 = df[df['Location'] == '(-1.00000000, -1.00000000)']['OFFENSE_CODE_GROUP'].value_counts().sort_values(ascending=False)

percent1_ocg_with_locatm1 = ocg_with_locatm1 / df['OFFENSE_CODE_GROUP'].value_counts().sort_values(ascending=False)

df_ocg_with_locatm1 = pd.concat([ocg_with_locatm1, percent1_ocg_with_locatm1 * 100], axis=1, keys=['Total', 'Porcentagem'], sort=False)



display(df_ocg_with_locatm1.sort_values('Porcentagem', ascending=False).head(15))
# Verficando a quantidade de ocorrências com valores faltosos, agrupados por tipos de ocorrências e distrito

qtd_ocg_with_locat0 = df[df['Location'] == '(0.00000000, 0.00000000)'].groupby('DISTRICT')['OFFENSE_CODE_GROUP'].value_counts().sort_values(ascending=False)

qtd_ocg_with_locatm1 = df[df['Location'] == '(-1.00000000, -1.00000000)'].groupby('DISTRICT')['OFFENSE_CODE_GROUP'].value_counts().sort_values(ascending=False)

total_qtd_ocg = df.groupby('DISTRICT')['OFFENSE_CODE_GROUP'].value_counts().sort_values(ascending=False)



np_percent_ocg_with_locat0 = np.round(qtd_ocg_with_locat0 / total_qtd_ocg * 100, 2)

np_percent_ocg_with_locatm1 = np.round(qtd_ocg_with_locatm1 / total_qtd_ocg * 100, 2)

df_ocg_with_locat0 = pd.concat([qtd_ocg_with_locat0, np_percent_ocg_with_locat0], axis=1, keys=('Total', 'Porcentagem'), sort=False)

df_ocg_with_locatm1 = pd.concat([qtd_ocg_with_locatm1, np_percent_ocg_with_locatm1], axis=1, keys=('Total', 'Porcentagem'), sort=False)



display(df_ocg_with_locat0.sort_values('Porcentagem', ascending=False).head(15))

display(df_ocg_with_locatm1.sort_values('Porcentagem', ascending=False).head(15))

# Criando um dataframe com a quantidade de registros com valores faltosos e a porcentagem

qtd_od_with_locat0 = df[df['Location'] == '(0.00000000, 0.00000000)'].groupby('DISTRICT')['OFFENSE_DESCRIPTION'].value_counts().sort_values(ascending=False)

qtd_od_with_locatm1 = df[df['Location'] == '(-1.00000000, -1.00000000)'].groupby('DISTRICT')['OFFENSE_DESCRIPTION'].value_counts().sort_values(ascending=False)

total_od = df.groupby('DISTRICT')['OFFENSE_DESCRIPTION'].value_counts().sort_values(ascending=False)

percent_od_with_locat0 = np.round(qtd_od_with_locat0 / total_od * 100, 2)

percent_od_with_locatm1 = np.round(qtd_od_with_locatm1 / total_od * 100, 2)

df_qtd_od_with_locat0 = pd.concat([qtd_od_with_locat0, percent_od_with_locat0], axis=1, keys=('Total', 'Porcentagem'), sort=False)

df_qtd_od_with_locatm1 = pd.concat([qtd_od_with_locatm1, percent_od_with_locatm1], axis=1, keys=('Total', 'Porcentagem'), sort=False)

display(df_qtd_od_with_locat0.sort_values('Porcentagem', ascending=False).head(15))

display(df_qtd_od_with_locatm1.sort_values('Porcentagem', ascending=False).head(15))
display(missing_data)
df[df['STREET'].isnull()].sample(10).T
df[df['STREET'].isnull() & ~df['Lat'].isnull()].T
coordinates = df[df['STREET'].isnull() & np.logical_and(~df['Lat'].isnull(), df['Lat'] != (-1))][['Lat', 'Long']].sample(100).reset_index()
coordinates
# alguns pontos no mapa que comprovam o informado acima: Alguns dos pontos onde temos a localidade, mas não temos o nome da rua, resultando assim missing values.

mapa_unamed_unknown_streets = folium.Map(location=[42.3567081,-71.0623494], zoom_start=14)

for i, lat, long in zip(coordinates['index'], coordinates['Lat'], coordinates['Long']):

    folium.Marker([lat, long], popup='{}'.format(i)).add_to(mapa_unamed_unknown_streets)

display(mapa_unamed_unknown_streets)
display(missing_data)
# Uma amostra dos registros que possuem valor faltando na variável DISTRICT

df[df['DISTRICT'].isnull()].sample(10).T
#Verificando os dados que possuem dados faltando na variável DISTRICT, REPORTING_AREA, Lat, Long e Location.

missing_values_district = df[np.logical_and(df['DISTRICT'].isnull(), 

                                            np.logical_or(df['REPORTING_AREA'] == ' ', 

                                                          np.logical_or(np.logical_or(df['Location'] == '(-1.00000000, -1.00000000)', 

                                                                                      df['Location'] == '(0.00000000, 0.00000000)'), df['STREET'].isnull())))]

missing_values_district.T
#Criando o dataframe com a quantidade de valores missing respeitando as proposições informadas anteriormente e porcentagem que ela representa do total de valores missings.

tese_explanation = np.round(pd.concat([missing_values_district[['DISTRICT']].isnull().sum(), missing_values_district[['DISTRICT']].isnull().sum() / df[df['DISTRICT'].isnull()]['DISTRICT'].isnull().count() * 100],

          axis=1, keys=('Total', 'Porcentagem'), sort=False),2)

display(missing_data)
display(tese_explanation)
#Verificando quais registros possuem valores missing na variável DISTRICT, entretanto não foram explicados pela proposição

df.loc[set(df[df['DISTRICT'].isnull()].index) - set(missing_values_district.index.values.tolist())].T
df[df['STREET'] == 'WASHINGTON ST']['DISTRICT'].value_counts()
display(df[df['REPORTING_AREA'] == '714']['DISTRICT'].value_counts())
display(df[df['STREET'] == 'GARFIELD AVE']['DISTRICT'].value_counts())

display(df[df['STREET'] == 'TOMAHAWK DR']['DISTRICT'].value_counts())

display(df[df['STREET'] == 'COMMONWEALTH AVE']['DISTRICT'].value_counts())

display(df[df['STREET'] == 'CONDOR ST']['DISTRICT'].value_counts())
display(df[df['REPORTING_AREA'] == '796']['DISTRICT'].value_counts())
display(missing_data)
df[df['UCR_PART'].isnull()]['OFFENSE_DESCRIPTION'].value_counts()
df[np.logical_and(df['OFFENSE_DESCRIPTION'] == 'HOME INVASION', ~df['UCR_PART'].isnull())]['UCR_PART'].value_counts()
df[np.logical_and(df['OFFENSE_DESCRIPTION'] == 'HUMAN TRAFFICKING - COMMERCIAL SEX ACTS', ~df['UCR_PART'].isnull())]['UCR_PART'].value_counts()
df[np.logical_and(df['OFFENSE_DESCRIPTION'] == 'INVESTIGATE PERSON', ~df['UCR_PART'].isnull())]['UCR_PART'].value_counts()
df[np.logical_and(df['OFFENSE_DESCRIPTION'] == 'HUMAN TRAFFICKING - INVOLUNTARY SERVITUDE', ~df['UCR_PART'].isnull())]['UCR_PART'].value_counts()
df.sample(10).T
df.info()
f, ax = plt.subplots(figsize=(20,4))

df.groupby('YEAR')['INCIDENT_NUMBER'].count().plot()

plt.ylabel('Quantidade de Ocorrências registradas')

plt.xlabel('Ano')

plt.grid()
df_2015 = df.where(df['YEAR'] == 2015)

df_2016 = df.where(df['YEAR'] == 2016)

df_2017 = df.where(df['YEAR'] == 2017)

df_2018 = df.where(df['YEAR'] == 2018)

df_2019 = df.where(df['YEAR'] == 2019)
df_2015_month = df_2015['MONTH'].value_counts()

df_2016_month = df_2016['MONTH'].value_counts()

df_2017_month = df_2017['MONTH'].value_counts()

df_2018_month = df_2018['MONTH'].value_counts()

df_2019_month = df_2019['MONTH'].value_counts()
display(np.round(df_2015_month.describe(),2))

display(np.round(df_2016_month.describe(),2))

display(np.round(df_2017_month.describe(),2))

display(np.round(df_2018_month.describe(),2))

display(np.round(df_2019_month.describe(),2))
f, ax = plt.subplots(figsize=(15,15))



plt.subplot(5,2,1)

df_2015_month.describe().plot(kind='bar')

plt.title('2015')

plt.grid()



plt.subplot(5,2,2)

df_2016_month.describe().plot(kind='bar')

plt.title('2016')

plt.grid()



plt.subplot(5,2,3)

df_2017_month.describe().plot(kind='bar')

plt.title('2017')

plt.grid()



plt.subplot(5,2,4)

df_2018_month.describe().plot(kind='bar')

plt.title('2018')

plt.grid()



plt.subplot(5,2,5)

df_2019_month.describe().plot(kind='bar')

plt.title('2019')

plt.grid()



plt.show()
f, ax = plt.subplots(figsize=(20,10))



plt.subplot(5,1,1)

sns.lineplot(x=df_2015['MONTH'].dropna().unique(), y=df_2015['MONTH'].value_counts(), color='r')

plt.title('2015')

plt.xlabel('Mês')

plt.ylabel('Quantidade')



plt.subplot(5,1,2)

sns.lineplot(x=df_2016['MONTH'].dropna().unique(), y=df_2016['MONTH'].value_counts(), color='b')

plt.title('2016')

plt.xlabel('Mês')

plt.ylabel('Quantidade')



plt.subplot(5,1,3)

sns.lineplot(x=df_2017['MONTH'].dropna().unique(), y=df_2017['MONTH'].value_counts(), color='g')

plt.title('2017')

plt.xlabel('Mês')

plt.ylabel('Quantidade')



plt.subplot(5,1,4)

sns.lineplot(x=df_2018['MONTH'].dropna().unique(), y=df_2018['MONTH'].value_counts(), color='y')

plt.title('2018')

plt.xlabel('Mês')

plt.ylabel('Quantidade')



plt.subplot(5,1,5)

sns.lineplot(x=df_2019['MONTH'].dropna().unique(), y=df_2019['MONTH'].value_counts(), color='c')

plt.title('2019')

plt.xlabel('Mês')

plt.ylabel('Quantidade')



plt.show()

df.info()
df['DAY_OF_WEEK'].value_counts().plot(kind='barh')
df['HOUR'].value_counts()[:10]
df['DISTRICT'].value_counts().plot(kind='barh')
df_2015_nona = df_2015.dropna()

df_2016_nona = df_2016.dropna().sample(int(50/100* df_2016['INCIDENT_NUMBER'].count()))

df_2017_nona = df_2017.dropna().sample(int(50/100* df_2017['INCIDENT_NUMBER'].count()))

df_2018_nona = df_2018.dropna().sample(int(50/100* df_2018['INCIDENT_NUMBER'].count()))

df_2019_nona = df_2019.dropna().sample(int(70/100* df_2019['INCIDENT_NUMBER'].count()))
places = []

mapa_offenses_2015 = folium.Map(location=[42.3546817,-70.9768088], zoom_start=11.5)

for i, line in df_2015_nona.iterrows():

    places.append((line['Lat'], line['Long']))



HeatMap(places, radius=6).add_to(mapa_offenses_2015)

display(mapa_offenses_2015)
places = []

mapa_offenses_2016 = folium.Map(location=[42.3546817,-70.9768088], zoom_start=11.5)

for i, line in df_2016_nona.iterrows():

    places.append((line['Lat'], line['Long']))



HeatMap(places, radius=6).add_to(mapa_offenses_2016)

display(mapa_offenses_2016)
places = []

mapa_offenses_2017 = folium.Map(location=[42.3546817,-70.9768088], zoom_start=11.5)

for i, line in df_2017_nona.iterrows():

    places.append((line['Lat'], line['Long']))



HeatMap(places, radius=6).add_to(mapa_offenses_2017)

display(mapa_offenses_2017)
places = []

mapa_offenses_2018 = folium.Map(location=[42.3546817,-70.9768088], zoom_start=11.5)

for i, line in df_2018_nona.iterrows():

    places.append((line['Lat'], line['Long']))



HeatMap(places, radius=6).add_to(mapa_offenses_2018)

display(mapa_offenses_2018)


places = []

mapa_offenses_2019 = folium.Map(location=[42.3546817,-70.9768088], zoom_start=11.5)

for i, line in df_2019_nona.iterrows():

    places.append((line['Lat'], line['Long']))



HeatMap(places, radius=6).add_to(mapa_offenses_2019)

display(mapa_offenses_2019)

text_2015 = str(df_2015['OFFENSE_CODE_GROUP'].dropna().tolist())

text_2016 = str(df_2016['OFFENSE_CODE_GROUP'].dropna().tolist())

text_2017 = str(df_2017['OFFENSE_CODE_GROUP'].dropna().tolist())

text_2018 = str(df_2018['OFFENSE_CODE_GROUP'].dropna().tolist())

text_2019 = str(df_2019['OFFENSE_CODE_GROUP'].dropna().tolist())
wordcloud = WordCloud(max_font_size=100, 

                      width = 1520, 

                      height = 535).generate(text_2015)

plt.figure(figsize=(16,9))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
wordcloud = WordCloud(max_font_size=100, 

                      width = 1520, 

                      height = 535).generate(text_2016)

plt.figure(figsize=(16,9))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
wordcloud = WordCloud(max_font_size=100, 

                      width = 1520, 

                      height = 535).generate(text_2017)

plt.figure(figsize=(16,9))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
wordcloud = WordCloud(max_font_size=100, 

                      width = 1520, 

                      height = 535).generate(text_2018)

plt.figure(figsize=(16,9))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()
wordcloud = WordCloud(max_font_size=100, 

                      width = 1520, 

                      height = 535).generate(text_2019)

plt.figure(figsize=(16,9))

plt.imshow(wordcloud)

plt.axis("off")

plt.show()