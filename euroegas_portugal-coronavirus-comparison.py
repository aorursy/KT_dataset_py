import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

        






import pandas as pd

from matplotlib import style

style.use('ggplot')

novo = pd.read_excel("/kaggle/input/bd-hopkins-corrected-and-updated/bdHopkins.xlsx")

novo = novo.sort_values(by='date')

novo[novo['Country/Region'] == 'Portugal'].tail()
ultimaData = novo[novo["Country/Region"] == "Portugal"]["date"].max()

ultimoDia = ultimaData.date()

print("Última data em que há dados de Portugal: ", ultimoDia)
"e da Austria {}".format(novo[novo["Country/Region"] == "Austria"]["date"].max().date())
# Constrói o gráfico comparativo da evolução nos 20 dias a partir dos 100 casos

tabela = pd.pivot_table(novo, values='CumConfirmed', index=['date', 'Country/Region'], aggfunc=np.sum)

tabela.reset_index(inplace = True)

tabela['Casos'] = tabela.CumConfirmed.astype('int32')

tabela['Data'] = pd.to_datetime(tabela['date'])

del tabela['date']

del tabela['CumConfirmed']

maximocasos = 1000

tabela = tabela[tabela['Casos'] > maximocasos]

tabela.rename(columns = {'Country/Region' : 'País'}, inplace = True)

paises = tabela['País'].unique()

tudo = pd.DataFrame()

for pais in paises:

    pais = tabela[tabela['País'] == pais]

    pais.reset_index(inplace = True, drop = True)

    pais.reset_index(inplace = True)

    tudo = pd.concat([tudo, pais], axis = 0)

tudo.rename(columns = {'index' : 'Dias'}, inplace = True)

tudo = tudo[tudo['País'] != 'Others']

tudo1 = tudo[tudo['País'] != 'China']

#selecao = ['Portugal', 'Switzerland','Korea, South', 'Italy', 'Spain', 'France', 'Germany', 'United Kingdom', 'US']

selecao = ['Portugal', 'Austria', 'Czechia', 'Germany', 'Denmark', 'Norway']

tudo2 = tudo1[tudo1['País'].isin(selecao)]

tabela2 = pd.pivot_table(tudo2, values='Casos', index=['Dias'],

                    columns=['País'], aggfunc=np.sum)

#tabela2 = tabela2[tabela2.index <= 50]

resumo = tabela2

import matplotlib.pyplot as plt 

%matplotlib inline

plt.figure()

resumo.plot(style = {'Portugal' : 'k-o'}, \

            title = 'Casos a seguir ao caso {} - situação em: {}'.format(maximocasos, ultimoDia), \

            figsize=(20,10), linewidth = 5);

plt.xlabel('Dias desde o caso {}'.format(maximocasos))

plt.ylabel('Número de Casos');
resumo.tail()


percentual = resumo.pct_change()[1:]*100

percentual = percentual[percentual["Portugal"] != 0 ]

plt.figure()

percentual.plot(style = {'Portugal' : 'k'}, \

            title = 'Variação percentual de casos nos dias a seguir ao caso {} - situação em: {}'.format(maximocasos, ultimoDia), \

            figsize=(20,10), kind='bar');

plt.xlabel('Dias desde o centésimo caso')

plt.ylabel('Variação percentual');
percentual.tail()
percentual2 = percentual

percentual2["Média"] = percentual2.mean(axis = 1)

compara2 = percentual2[['Portugal', 'Média']]

print(selecao)

plt.figure()

compara2[-21:].plot(title = 'Variação comparada com a média de {} - situação em: {}'.format(selecao, ultimoDia), \

            figsize=(20,10), kind='bar');

plt.xlabel('Dias desde o caso {}'.format(maximocasos))

plt.ylabel('Variação percentual dos novos casos');
compara2.tail()
compara2.tail()
mediaMovel = compara2.rolling(window=3).mean()[2:]

mediaMovel[-21:].plot(title = 'Média móvel de 3 dias da variação comparada com a média de {} - situação em: {}'.format(selecao, ultimoDia), \

            figsize=(20,10), kind='bar');

plt.xlabel('Dias desde o caso {}'.format(maximocasos))

plt.ylabel('Variação percentual dos novos casos')
# Constrói o gráfico comparativo da evolução os óbitos



tabela2 = pd.pivot_table(novo, values='CumDeaths', index=['date', 'Country/Region'], aggfunc=np.sum)

tabela2.reset_index(inplace = True)

tabela2['Óbitos'] = tabela2.CumDeaths.astype('int32')

tabela2['Data'] = pd.to_datetime(tabela2['date'])

del tabela2['date']

del tabela2['CumDeaths']

tabela2 = tabela2[tabela2['Óbitos'] > 10]

tabela2.rename(columns = {'Country/Region' : 'País'}, inplace = True)

paises2 = tabela2['País'].unique()

tudo2 = pd.DataFrame()

for pais2 in paises2:

    pais2 = tabela2[tabela2['País'] == pais2]

    pais2.reset_index(inplace = True, drop = True)

    pais2.reset_index(inplace = True)

    tudo2 = pd.concat([tudo2, pais2], axis = 0)

tudo2.rename(columns = {'index' : 'Dias'}, inplace = True)

tudo2 = tudo2[tudo2['País'] != 'Others']

tudo2 = tudo2[tudo2['País'] != 'China']

tudo2 = tudo2[tudo2['País'].isin(selecao)]

tabela2 = pd.pivot_table(tudo2, values='Óbitos', index=['Dias'],

                    columns=['País'], aggfunc=np.sum)

tabela2 = tabela2[tabela2.index <= 90]

resumo = tabela2 

%matplotlib inline

plt.figure()

ax = resumo.plot(style = {'Portugal' : 'k-o'}, \

            title = 'Óbitos nos 30 dias a seguir ao décimo óbito - situação em: {}'.format(ultimoDia), \

            figsize=(20,10), linewidth = 5)

ax.locator_params(integer=True)

plt.xlabel('Dias desde o décimo óbito')

plt.ylabel('Número de Óbitos')

plt.show();
#allData.to_excel('bdHopkins.xlsx', index = False) # Reset da BD
hopkins = novo.copy()

tabela = pd.pivot_table(hopkins, values='CumDeaths', index=['date', 'Country/Region'], aggfunc=np.sum)

tabela.reset_index(inplace = True)

mm = 1000 # mínimo de mortos a considerar, para excluír os microestados

aux = tabela[tabela['CumDeaths'] > mm]

lista = list(aux['Country/Region'].unique())

maiores = tabela[tabela['Country/Region'].isin(lista)]

populacao = pd.read_excel("/kaggle/input/bd-hopkins-corrected-and-updated/populao.xlsx")[['Country/Region','População']]

maiores = maiores.merge(populacao, on = 'Country/Region',how= 'inner')

maiores['por Milhão'] = 1000000*maiores['CumDeaths'] / maiores['População']

paises = list(maiores['Country/Region'].unique())

diarios = pd.DataFrame()

mmpm = 100 # mortos por milhão no dia 0

for pais in paises:

    df = maiores[maiores['Country/Region'] == pais].copy()

    df = df[df['por Milhão'] > mmpm]

    df.reset_index(drop = True, inplace = True)

    df.reset_index(inplace = True)

    diarios = pd.concat([diarios, df])

    del df

diarios = diarios[diarios['index'] < 90]

diarios.tail()
tabela2 = pd.pivot_table(diarios, values='por Milhão', 

                         index=['index'], columns=['Country/Region'], aggfunc=np.mean)

#tabela2['Média'] = tabela2.mean(axis = 1)

%matplotlib inline

#del tabela2['Diamond Princess']

tabela2 = tabela2.sort_values(by = tabela2.index[-1], ascending = False, axis = 1,  kind='quicksort', na_position='last')

ax = tabela2.plot(style = {'Portugal' : 'k-o', 'Média': 'r-o'}, figsize=(24,10), linewidth = 5)

ax.set_xlabel('Dias decorridos'.format(mmpm), fontsize=20, labelpad = 20)

ax.set_ylabel("Óbitos por milhão", fontsize=20, labelpad = 20)

ax.locator_params(integer=True)

ax.set_title('Óbitos por milhão desde o dia em que houve {} óbitos por milhão'.format(mmpm), pad = 30)

ax.title.set_size(40);
tabela2.tail()
selecao = [ 'Italy', 'Portugal', 'Spain','France', 'Germany']

tabela3 = tabela2[selecao]

ax = tabela3.plot(style = {'Portugal' : 'k-o'}, figsize=(20,10), linewidth = 5);

ax.set_xlabel('Óbitos por milhão desde o dia em que houve {} mortos por milhão'.format(mmpm), fontsize=20, labelpad = 20)

ax.set_ylabel("Óbitos por milhão", fontsize=20, labelpad = 20);

ax.set_title("Evolução dos Óbitos por Milhão", pad = 30)

ax.locator_params(integer=True)

ax.title.set_size(40)
#selecao = ['Belgium', 'China', 'France', 'Germany', 'Iran', 'Italy', 'Korea, South', 'Netherlands', 'Portugal', 'Spain', 'Sweden','Switzerland', 'US', 'United Kingdom']

selecao = ['Belgium', 'Netherlands', 'Portugal', 'Sweden','Switzerland']

tabela3 = tabela2[selecao]

ax = tabela3.plot(style = {'Portugal' : 'k-o'}, figsize=(20,10), linewidth = 5);

ax.set_xlabel('Óbitos por milhão desde o dia em que houve {} óbitos por milhão'.format(mmpm), fontsize=20, labelpad = 20)

ax.set_ylabel("Óbitos por milhão", fontsize=20, labelpad = 20);

ax.set_title("Evolução dos Óbitos por Milhão", pad = 30)

ax.locator_params(integer=True)

ax.title.set_size(40)
tabela3.tail()
selecao = ['Portugal','US', 'United Kingdom']

tabela3 = tabela2[selecao]

ax = tabela3.plot(style = {'Portugal' : 'k-o'}, figsize=(20,10), linewidth = 5);

ax.set_xlabel('Óbitos por milhão desde o dia em que houve {} mortos por milhão'.format(mmpm), fontsize=20, labelpad = 20)

ax.set_ylabel("Óbitos por milhão", fontsize=20, labelpad = 20);

ax.set_title("Evolução dos Óbitos por Milhão", pad = 30)

ax.locator_params(integer=True)

ax.title.set_size(40)
tabela3.tail(10)
selecao = ['Sweden', 'Portugal', 'Germany', 'Netherlands']

tabela3 = tabela2[selecao]

tabela3 = tabela3.sort_values(by=tabela3.index[-1], ascending = False, axis = 1)

ax = tabela3.plot(style = {'Sweden' : 'k-o', 'Netherlands' : 'r-o', 'Portugal' : 'b-o'}, figsize=(20,11), linewidth = 5);

ax.set_xlabel('Dias desde que houve {} óbitos por milhão'.format(mmpm), fontsize=20, labelpad = 20)

ax.set_ylabel("Óbitos por milhão", fontsize=20, labelpad = 20);

ax.set_title("Evolução dos Óbitos por Milhão", pad = 30)

ax.locator_params(integer=True)

ax.title.set_size(40)
confirm = pd.pivot_table(novo, values='CumConfirmed', index=['date'], columns=['Country/Region'], aggfunc=np.sum).diff()[1:]

mm3 = confirm.rolling(window=3).mean()[5:]

mm6 = confirm.rolling(window=6).mean()[5:]
tuplas = []

for pais in mm3.columns:

    m3 = mm3.loc[mm3[~ mm3[pais].isnull()].index[-1]][pais]

    m6 = mm6.loc[mm6[~ mm6[pais].isnull()].index[-1]][pais]

    tuplas.append((pais,m3,m6))

medias = pd.DataFrame(tuplas, columns = ['País', 'Média de 3 dias', 'Média de 6 dias'])



medias['A descer'] = medias['Média de 3 dias'] < medias['Média de 6 dias']

aDescer = medias[medias['A descer']]

print("Países a descer")

limite = 100

aDescer[aDescer['Média de 6 dias'] > limite]
confirmd = pd.pivot_table(novo, values='CumDeaths', index=['date'], columns=['Country/Region'], aggfunc=np.sum).diff()[1:]

mmd3 = confirmd.rolling(window=3).mean()[5:]

mmd6 = confirmd.rolling(window=6).mean()[5:]

tuplasd = []

for pais in mm3.columns:

    md3 = mmd3.loc[mmd3[~ mmd3[pais].isnull()].index[-1]][pais]

    md6 = mmd6.loc[mmd6[~ mmd6[pais].isnull()].index[-1]][pais]

    tuplasd.append((pais,md3,md6))

mediasd = pd.DataFrame(tuplasd, columns = ['País', 'Média de 3 dias', 'Média de 6 dias'])



mediasd['A descer'] = mediasd['Média de 3 dias'] < mediasd['Média de 6 dias']

aDescerd = mediasd[mediasd['A descer']]

print("Países a descer")

limite = 5

aDescerd[aDescerd['Média de 6 dias'] > limite]
obitos = pd.pivot_table(novo, values='CumDeaths', index=['date'], columns = ['Country/Region'], aggfunc=np.sum)

obitos.reset_index(inplace=True)

obitos['date'] = obitos['date'].dt.date

obitos.set_index('date', inplace = True)

selecao = ['Portugal', 'Switzerland', 'Italy', 'Spain', 'France', 'Germany', 'Belgium', 'US']

obitos = obitos[selecao]

obpct = obitos[-5:-1].pct_change()[1:]*100

ax = obpct.plot( figsize=(20,10), kind = 'bar');

ax.set_xlabel('Data', fontsize=20, labelpad = 20)

ax.set_ylabel("Percentagem", fontsize=20, labelpad = 20);

ax.set_title("Aumento percentual do números de Óbitos", pad = 30)

ax.title.set_size(40)
porpais = pd.pivot_table(novo, values='CumConfirmed', index=['date', 'Country/Region'], aggfunc=np.sum)

porpais.reset_index(inplace = True)

porpais['Casos'] = porpais.CumConfirmed.astype('int32')

porpais['Data'] = pd.to_datetime(porpais['date'])

del porpais['date']

del porpais['CumConfirmed']

porpais.rename(columns = {'Country/Region' : 'País'}, inplace = True)

tabela = pd.pivot_table(porpais, values='Casos', index=['Data'], columns = ['País'], aggfunc=np.sum)

tabela2 = tabela[tabela.columns[tabela.max() > 10000]]

tabela2 = tabela2[-30:]

ax = tabela2.plot(style = {'Portugal' : 'k-o'}, figsize=(20,10))

ax.set_xlabel('Data', fontsize=20, labelpad = 20)

ax.set_ylabel("Casos", fontsize=20, labelpad = 20);

ax.set_title("Evolução do Número de Casos nos países com mais de 10.000", pad = 30)

ax.title.set_size(40)
novoscasos = tabela.diff()[-20:]

novoscasos.drop(columns = ['US'], inplace = True)

novoscasos = novoscasos[novoscasos.columns[novoscasos.max() > 1000]]

ax = novoscasos.plot(style = {'Portugal' : 'k-o'}, figsize=(20,10))

ax.set_xlabel('Data', fontsize=20, labelpad = 20)

ax.set_ylabel("Casos", fontsize=20, labelpad = 20);

ax.set_title("Evolução do Novos Casos", pad = 30)

ax.title.set_size(40)
novoscasos.tail()
casos = novo.copy()

casos = pd.pivot_table(casos, values='CumConfirmed', index=['date', 'Country/Region'], aggfunc=np.sum)

casos.reset_index(inplace = True)

casos = casos.merge(populacao, on = 'Country/Region',how= 'inner')

casos['Casos por Milhão'] = 1000000*casos['CumConfirmed'] / casos['População']

casos.rename(columns={'Country/Region': 'País'}, inplace = True)

interessa = casos[casos['País'].isin(['Austria', 'Czechia', 'Norway', 'Germany', 'Portugal', 'Denmark'])]

interessa = pd.pivot_table(interessa, values='Casos por Milhão', index=['date'], columns =['País'], aggfunc=np.sum)

interessa = interessa.diff().rolling(window=7).mean()[-35:]

ax = interessa.plot(style = {'Portugal' : 'k-o'}, figsize=(20,10), linewidth = 5)

ax.set_xlabel('Data', fontsize=20, labelpad = 20)

ax.set_ylabel("Novos Casos por Milhão", fontsize=20, labelpad = 20)

ax.set_title("Casos diários por Milhão nos países em Abertura (média de 7 dias)", pad = 30)

ax.title.set_size(30);


interessa = casos[casos['País'].isin(['Sweden', 'Finland', 'Norway', 'Germany', 'Denmark'])]

interessa = pd.pivot_table(interessa, values='Casos por Milhão', index=['date'], columns =['País'], aggfunc=np.sum)

interessa = interessa.diff().rolling(window=7).mean()[-35:]

ax = interessa.plot( figsize=(20,10), linewidth = 5)

ax.set_xlabel('Data', fontsize=20, labelpad = 20)

ax.set_ylabel("Novos Casos por Milhão", fontsize=20, labelpad = 20)

ax.set_title("Casos diários por Milhão nos países nórdicos", pad = 30)

ax.title.set_size(30);
singapura = casos[casos['País'].isin(['Portugal', 'Singapore'])]

singapura = pd.pivot_table(singapura, values='Casos por Milhão', index=['date'], columns =['País'], aggfunc=np.sum)

singapura = singapura.diff()[-31:]

ax = singapura.plot(style = {'Portugal' : 'k-o'}, figsize=(20,10), linewidth = 5)

ax.set_xlabel('Data', fontsize=20, labelpad = 20)

ax.set_ylabel("Casos por Dia", fontsize=20, labelpad = 20)

ax.set_title("Casos diários por Milhão -  Singapura", pad = 30)

ax.title.set_size(30);
obitos = novo.copy()

obitos = pd.pivot_table(obitos, values='CumDeaths', index=['date', 'Country/Region'], aggfunc=np.sum)

obitos.reset_index(inplace = True)

obitos = obitos.merge(populacao, on = 'Country/Region',how= 'inner')

obitos['Óbitos por Milhão'] = 1000000*obitos['CumDeaths'] / obitos['População']

obitos.rename(columns={'Country/Region': 'País'}, inplace = True)

ointeressa = obitos[obitos['País'].isin(['Austria', 'Czechia', 'Norway', 'Germany', 'Portugal', 'Denmark'])]

ointeressa = pd.pivot_table(ointeressa, values='Óbitos por Milhão', index=['date'], columns =['País'], aggfunc=np.sum)

ointeressa = ointeressa.diff().rolling(window=7).mean()[-35:]

ax = ointeressa.plot(style = {'Portugal' : 'k-o'}, figsize=(20,10), linewidth = 5, colormap='Paired')

ax.set_xlabel('Data', fontsize=15, labelpad = 20)

ax.set_ylabel("Óbitos por Milhão", fontsize=15, labelpad = 20)

ax.set_title("Evolução do Número de Óbitos diários por Milhão nos países em Abertura (média de 7 dias)", pad = 30)

ax.title.set_size(20);
obitos = novo.copy()

obitos = pd.pivot_table(obitos, values='CumDeaths', index=['date', 'Country/Region'], aggfunc=np.sum)

obitos.reset_index(inplace = True)

obitos = obitos.merge(populacao, on = 'Country/Region',how= 'inner')

obitos['Óbitos por Milhão'] = 1000000*obitos['CumDeaths'] / obitos['População']

obitos.rename(columns={'Country/Region': 'País'}, inplace = True)

ointeressa = obitos[obitos['País'].isin(['Sweden', 'Finland', 'Norway', 'Germany', 'Denmark'])]

ointeressa = pd.pivot_table(ointeressa, values='Óbitos por Milhão', index=['date'], columns =['País'], aggfunc=np.sum)

ointeressa = ointeressa.diff().rolling(window=7).mean()[-35:]

ax = ointeressa.plot(style = {'Portugal' : 'k-o'}, figsize=(20,10), linewidth = 5, colormap='Paired')

ax.set_xlabel('Data', fontsize=15, labelpad = 20)

ax.set_ylabel("Óbitos por Milhão", fontsize=15, labelpad = 20)

ax.set_title("Evolução do Número de Óbitos diários por Milhão nos países nórdicos (média de 7 dias)", pad = 30)

ax.title.set_size(20);
obitos = novo.copy()

obitos = pd.pivot_table(obitos, values='CumDeaths', index=['date', 'Country/Region'], aggfunc=np.sum)

obitos.reset_index(inplace = True)

obitos = obitos.merge(populacao, on = 'Country/Region',how= 'inner')

obitos['Óbitos por Milhão'] = 1000000*obitos['CumDeaths'] / obitos['População']

obitos.rename(columns={'Country/Region': 'País'}, inplace = True)

ointeressa = obitos[obitos['País'].isin(['Sweden', 'Finland', 'Norway', 'Germany', 'Denmark'])]

ointeressa = pd.pivot_table(ointeressa, values='Óbitos por Milhão', index=['date'], columns =['País'], aggfunc=np.sum)

ointeressa.sort_values(by = ointeressa.index[-1], axis = 1, inplace= True, ascending = False)

ax = ointeressa[-147:].plot(style = {'Portugal' : 'k-o'}, figsize=(20,10), linewidth = 5, colormap='Paired')

ax.set_xlabel('Data', fontsize=15, labelpad = 20)

ax.set_ylabel("Óbitos por Milhão", fontsize=15, labelpad = 20)

ax.set_title("Óbitos por Milhão na Suécia e países vizinhos", pad = 30)

ax.title.set_size(20);
obitos = novo.copy()

obitos = pd.pivot_table(obitos, values='CumDeaths', index=['date', 'Country/Region'], aggfunc=np.sum)

obitos.reset_index(inplace = True)

obitos = obitos.merge(populacao, on = 'Country/Region',how= 'inner')

obitos['Óbitos por Milhão'] = 1000000*obitos['CumDeaths'] / obitos['População']

obitos.rename(columns={'Country/Region': 'País'}, inplace = True)

select = list(obitos[(obitos['Óbitos por Milhão'] > 1) & (obitos['CumDeaths'] > 500)]['País'].unique())

ointeressa = obitos[obitos['País'].isin(select)]

ointeressa = pd.pivot_table(ointeressa, values='Óbitos por Milhão', index=['date'], columns =['País'], aggfunc=np.sum)

ointeressa = ointeressa.diff().rolling(window=7).mean()[-7:]

ointeressa.sort_values(by = ointeressa.index[-2], axis = 1, inplace= True, ascending = False)

ointeressa = ointeressa[ointeressa.columns[0:25]]

ax = ointeressa.plot(style = {'Portugal' : 'k-o'}, figsize=(20,10), linewidth = 5, colormap='Paired')

ax.set_xlabel('Data', fontsize=15, labelpad = 20)

ax.set_ylabel("Óbitos por Dia por milhão", fontsize=15, labelpad = 20)

ax.set_title("Os piores em óbitos diários por milhão (média dos últimos 7 dias)", pad = 30)

ax.legend(loc = 'upper left')

ax.title.set_size(20);
casos = novo.copy()

casos = pd.pivot_table(casos, values='CumConfirmed', index=['date', 'Country/Region'], aggfunc=np.sum)

casos.reset_index(inplace = True)

casos = casos.merge(populacao, on = 'Country/Region',how= 'inner')

casos['Casos por Milhão'] = 1000000*casos['CumConfirmed'] / casos['População']

casos.rename(columns={'Country/Region': 'País'}, inplace = True)

interessa = casos[casos['País'].isin(['Spain', 'Italy', 'France', 'United Kingdom', 'Germany', 'Portugal', 'Belgium', 'Netherlands','Ireland','Luxembourg','Denmark', 'Switzerland', 'Sweden', 'Norway', 'Finland', 'Austria'])]

interessa = pd.pivot_table(interessa, values='Casos por Milhão', index=['date'], columns =['País'], aggfunc=np.sum)

interessa = interessa.diff().rolling(window=7).mean()[-35:]

if ~(interessa.at[interessa.index[-1],'Austria'] > 0):

    maxind = -2

else:

    maxind = -1

interessa.sort_values(by = interessa.index[maxind], axis=1, ascending=False, inplace=True, kind='quicksort', na_position='last')

ax = interessa[-21:].plot(style = {'Portugal' : 'k-o'}, figsize=(25,10), linewidth = 5, colormap='Paired')

ax.set_xlabel('Data', fontsize=20, labelpad = 20)

ax.set_ylabel("Novos Casos por Dia", fontsize=20, labelpad = 20)

ax.set_title("Casos diários por Milhão na Europa Ocidental (média de 7 dias)", pad = 30)

ax.title.set_size(30);
souk = interessa[['Portugal','United Kingdom']]

ax = souk.plot(style = {'Portugal' : 'k-o'}, figsize=(25,10), linewidth = 5, colormap='Paired')

ax.set_xlabel('Data', fontsize=20, labelpad = 20)

ax.set_ylabel("Novos Casos por Dia", fontsize=20, labelpad = 20)

ax.set_title("Casos diários por Milhão em Portugal e UK (média de 7 dias)", pad = 30)

ax.title.set_size(30);
obitos = novo.copy()

obitos = pd.pivot_table(obitos, values='CumDeaths', index=['date', 'Country/Region'], aggfunc=np.sum)

obitos.reset_index(inplace = True)

obitos = obitos.merge(populacao, on = 'Country/Region',how= 'inner')

obitos['Óbitos por Milhão'] = 1000000*obitos['CumDeaths'] / obitos['População']

obitos.rename(columns={'Country/Region': 'País'}, inplace = True)

select = ['Spain', 'Italy', 'France', 'United Kingdom', 'Germany', 'Portugal', 'Belgium', 'Netherlands','Ireland','Luxembourg','Denmark', 'Switzerland', 'Sweden', 'Norway', 'Finland', 'Austria']

ointeressa = obitos[obitos['País'].isin(select)]

ointeressa = pd.pivot_table(ointeressa, values='Óbitos por Milhão', index=['date'], columns =['País'], aggfunc=np.sum)

ointeressa = ointeressa.diff().rolling(window=7).mean()[-35:]

if ~(ointeressa.at[ointeressa.index[-1],'Austria'] > 0):

    maxind = -2

else:

    maxind = -1

ointeressa.sort_values(by = ointeressa.index[maxind], axis=1, ascending=False, inplace=True, kind='quicksort', na_position='last')

ax = ointeressa[-14:].plot(style = {'Portugal' : 'k-o'}, figsize=(25,10), linewidth = 5, colormap='Paired')

ax.set_xlabel('Data', fontsize=15, labelpad = 20)

ax.set_ylabel("Óbitos por Dia", fontsize=15, labelpad = 20)

ax.legend(loc='upper left')

ax.set_title("Evolução do Número de Óbitos diários por Milhão na Europa Ocidental (média de 7 dias)", pad = 30)

ax.title.set_size(20);
ointeressa.tail()
souk = ointeressa[['Portugal','United Kingdom']]

ax = souk.plot(style = {'Portugal' : 'k-o'}, figsize=(25,10), linewidth = 5, colormap='Paired')

ax.set_xlabel('Data', fontsize=20, labelpad = 20)

ax.set_ylabel("Óbitos por Dia", fontsize=20, labelpad = 20)

ax.set_title("Óbitos diários por Milhão em Portugal e UK (média de 7 dias)", pad = 30)

ax.title.set_size(30);
ue = ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia', 'Denmark', 'Estonia', 'Finland', 'France', 

      'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Luxembourg', 'Malta', 'Netherlands', 'Poland',

      'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden']

casos = novo.copy()

casos = pd.pivot_table(casos, values='CumConfirmed', index=['date', 'Country/Region'], aggfunc=np.sum)

casos.reset_index(inplace = True)

casos = casos.merge(populacao, on = 'Country/Region',how= 'inner')

casos['Casos por Milhão'] = 1000000*casos['CumConfirmed'] / casos['População']

casos.rename(columns={'Country/Region': 'País'}, inplace = True)

interessa = casos[casos['País'].isin(ue)]

interessa = pd.pivot_table(interessa, values='Casos por Milhão', index=['date'], columns =['País'], aggfunc=np.sum)

if ~(interessa.at[interessa.index[-1],'Austria'] > 0):

    maxind = -2

else:

    maxind = -1

#interessa.sort_values(by = interessa.index[maxind], axis=1, ascending=False, inplace=True, kind='quicksort', na_position='last')

interessa = interessa.diff().rolling(window=7).mean()[-35:]

interessa.sort_values(by = interessa.index[maxind], axis=1, ascending=False, inplace=True)

ax = interessa[-14:].plot(style = {'Portugal' : 'k-o'}, figsize=(20,12), linewidth = 5, colormap='Paired')

ax.set_xlabel('Data', fontsize=20, labelpad = 20)

ax.set_ylabel("Novos Casos por Dia", fontsize=20, labelpad = 20)

ax.set_title("Casos diários por Milhão na União Europeia (média de 7 dias)", pad = 30)

ax.title.set_size(30);
obitos = novo.copy()

obitos = pd.pivot_table(obitos, values='CumDeaths', index=['date', 'Country/Region'], aggfunc=np.sum)

obitos.reset_index(inplace = True)

obitos = obitos.merge(populacao, on = 'Country/Region',how= 'inner')

obitos['Óbitos por Milhão'] = 1000000*obitos['CumDeaths'] / obitos['População']

obitos.rename(columns={'Country/Region': 'País'}, inplace = True)

ointeressa = obitos[obitos['País'].isin(ue)]

ointeressa = pd.pivot_table(ointeressa, values='Óbitos por Milhão', index=['date'], columns =['País'], aggfunc=np.sum)

ointeressa = ointeressa.diff().rolling(window=7).mean()[-35:]

if ~(ointeressa.at[ointeressa.index[-1],'Austria'] > 0):

    maxind = -2

else:

    maxind = -1

ointeressa.sort_values(by = ointeressa.index[maxind], axis=1, ascending=False, inplace=True, kind='quicksort', na_position='last')

ax = ointeressa[-14:].plot(style = {'Portugal' : 'k-o'}, figsize=(25,10), linewidth = 5, colormap='Paired')

ax.set_xlabel('Data', fontsize=15, labelpad = 20)

ax.set_ylabel("Óbitos por Dia", fontsize=15, labelpad = 20)

ax.set_title("Evolução do Número de Óbitos diários por Milhão na União Europeia (média de 7 dias)", pad = 30)

ax.legend(loc="upper left")

ax.title.set_size(20);
popue = populacao[populacao['Country/Region'].isin(ue)]['População'].sum()

popport = populacao[populacao['Country/Region'] == 'Portugal']['População'].sum()

interessa = casos[casos['País'].isin(ue)]

interessa = pd.pivot_table(interessa, values='CumConfirmed', index=['date'], columns =['País'], aggfunc=np.sum)

if ~(interessa.at[interessa.index[-1],'Austria'] > 0):

    interessa = interessa[:-1]

interessa['União Europeia'] = interessa.sum(axis = 1)

interessa = interessa[['Portugal', 'União Europeia']]

interessa['Portugal'] = 1000000*interessa['Portugal'] / popport

interessa['União Europeia'] = 1000000*interessa['União Europeia'] /popue

us = casos[casos['País'].isin(['US'])][['date','Casos por Milhão']].set_index('date')

us = us.rename(columns={'Casos por Milhão':'US'})

br = casos[casos['País'].isin(['Brazil'])][['date','Casos por Milhão']].set_index('date')

br = br.rename(columns={'Casos por Milhão':'Brasil'})

interessa = pd.concat([interessa, us, br], axis = 1)

interessa = interessa.diff().rolling(window=7).mean()

interessa = interessa.sort_values(by=interessa.index[-1], axis = 1, ascending = False)

ax = interessa[-56:].plot(style = {'Portugal' : 'k-o'}, figsize=(25,10), linewidth = 5, colormap='Paired')

ax.set_xlabel('Data', fontsize=20, labelpad = 20)

ax.set_ylabel("Casos", fontsize=20, labelpad = 20)

ax.set_title("Casos diários por Milhão - Portugal versus União Europeia, Brasil e EUA (média de 7 dias)", pad = 30)

ax.title.set_size(30);
interessa.tail()
interessa = obitos[obitos['País'].isin(ue)]

interessa = pd.pivot_table(interessa, values='CumDeaths', index=['date'], columns =['País'], aggfunc=np.sum)

if ~(interessa.at[interessa.index[-1],'Austria'] > 0):

    interessa = interessa[:-1]

interessa['União Europeia'] = interessa.sum(axis = 1)

interessa = interessa[['Portugal', 'União Europeia']]

interessa['Portugal'] = 1000000*interessa['Portugal'] / popport

interessa['União Europeia'] = 1000000*interessa['União Europeia'] /popue

us = obitos[obitos['País'].isin(['US'])][['date','Óbitos por Milhão']].set_index('date')

us = us.rename(columns={'Óbitos por Milhão':'US'})

br = obitos[obitos['País'].isin(['Brazil'])][['date','Óbitos por Milhão']].set_index('date')

br = br.rename(columns={'Óbitos por Milhão':'Brasil'})

interessa = pd.concat([interessa, us, br], axis = 1)

interessa = interessa.diff().rolling(window=7).mean()

interessa = interessa.sort_values(by=interessa.index[-1], axis = 1, ascending = False)

ax = interessa[-28:].plot(style = {'Portugal' : 'k-o'}, figsize=(25,10), linewidth = 5, colormap='Paired')

ax.set_xlabel('Data', fontsize=20, labelpad = 20)

ax.set_ylabel("Óbitos", fontsize=20, labelpad = 20)

ax.set_title("Óbitos diários por Milhão Portugal versus União Europeia, Brasil e EUA (média de 7 dias)", pad = 30)

ax.title.set_size(30);
interessa.tail()
porpais = novo.pivot_table(index=['date','Country/Region'], aggfunc = np.sum).reset_index()

porpais = porpais.merge(populacao[populacao['População'] > 1000000], on = 'Country/Region',how= 'inner')

porpais['Casos por Milhão'] = (porpais['CumConfirmed'] / porpais['População'])*1000000

porpais['Óbitos por Milhão'] = (porpais['CumDeaths'] / porpais['População'])*1000000

porpais['Recuperados por Milhão'] = (porpais['CumRecovered'] / porpais['População'])*1000000

porpais = porpais[['date','Country/Region','Casos por Milhão', 'Óbitos por Milhão',  'Recuperados por Milhão']]

porpais = porpais.rename(columns={'Country/Region': 'País'})

# Obter os que tem óbitos por milhãpo > 100

paises = porpais['País'].unique()

tudo = pd.DataFrame()

limite = 100

for pais in paises:

    df = porpais[porpais['País'] == pais]

    df = df[df['Óbitos por Milhão'] > limite]

    if len(df.index) > 0:

        df.reset_index(drop=True, inplace = True)

        df.reset_index(inplace = True)

        tudo = pd.concat([tudo, df], axis = 0)

tudo = tudo[['index', 'País','Recuperados por Milhão']]

tudo = tudo.pivot_table(index = 'index', columns = 'País', values = 'Recuperados por Milhão')

ax = tudo[:tudo['Portugal'].last_valid_index()].plot(style = {'Portugal' : 'k-o'}, figsize=(25,10), linewidth = 5, colormap='Paired')

ax.set_xlabel('Data', fontsize=20, labelpad = 20)

ax.set_ylabel("Recuperados", fontsize=20, labelpad = 20)

ax.set_title("Recuperados por Milhão desde o dia de {} óbitos por milhão".format(limite), pad = 30)

ax.title.set_size(30);
semanal = novo.copy()

semanal['week'] = semanal['date'].apply(lambda  x : x.week)

semanal = semanal.sort_values(by='date')

semanal = semanal.groupby(by=['week', 'Country/Region']).last()

semanal.reset_index(inplace=True)

semanal = semanal.pivot_table(index='week', columns='Country/Region', values='CumConfirmed')

maiores = semanal.copy().diff()[1:-1] # Corta primeira e ultima linha

maiores.reset_index(inplace= True)

maiores = maiores.melt(id_vars='week', var_name='Country/Region', value_name='Novos Casos Semanais')

minpop = 100000

maiores = maiores.merge(populacao[populacao['População'] > minpop], on = 'Country/Region',how= 'inner')

maiores['Casos Semanais por Milhão'] = 1000000*(maiores['Novos Casos Semanais'] / maiores['População'])

maiores = maiores.pivot_table(index='week', columns='Country/Region', values='Casos Semanais por Milhão')

maiores.to_excel('semanais.xlsx')

minimodecasos = 521.35

listamaiores = [maiores.columns[i] for i in range(0, len(maiores.columns)) if maiores.at[maiores.index[-1],maiores.columns[i]] > minimodecasos]

maiores = maiores[listamaiores]

maiores.sort_values(by = maiores.index[-1], axis=1, ascending=False, inplace=True, kind='quicksort', na_position='last')

ax = maiores[-10:].plot(figsize=(25,10), linewidth = 5, colormap='Paired');

ax.set_xlabel('Semana', fontsize=20, labelpad = 20)

ax.set_ylabel("Casos", fontsize=20, labelpad = 20)

ax.set_title("Casos semanais por Milhão - última semana pior que a pior de Portugal", pad = 30)

ax.title.set_size(30);
matriz = casos.pivot_table(index = 'date', columns='País', values = 'Casos por Milhão').diff().rolling(window=7).mean()

if ~(matriz.at[matriz.index[-1],'Austria'] > 0):

    maxind = -2

else:

    maxind = -1

matriz.sort_values(by = matriz.index[maxind], axis=1, ascending=False, inplace=True, kind='quicksort', na_position='last')

piores= matriz[matriz.columns[0:10]]

ax = piores[-14:].plot(style = {'Portugal' : 'k-o'}, figsize=(25,10), linewidth = 5, colormap='Paired')

ax.set_xlabel('Data', fontsize=20, labelpad = 20)

ax.set_ylabel("Casos", fontsize=20, labelpad = 20)

ax.set_title("Piores em Novos Casos por Milhão (média de 7 dias)", pad = 30)

ax.title.set_size(30);
omatriz = obitos.pivot_table(index = 'date', columns='País', values = 'Óbitos por Milhão').diff().rolling(window=7).mean()

if ~(omatriz.at[omatriz.index[-1],'Austria'] > 0):

    maxind = -2

else:

    maxind = -1

omatriz.sort_values(by = omatriz.index[maxind], axis=1, ascending=False, inplace=True, kind='quicksort', na_position='last')

piores= omatriz[omatriz.columns[0:10]]

ax = piores[-14:].plot(style = {'Portugal' : 'k-o'}, figsize=(25,10), linewidth = 5, colormap='Paired')

ax.set_xlabel('Data', fontsize=20, labelpad = 20)

ax.set_ylabel("Óbitos", fontsize=20, labelpad = 20)

ax.set_title("Piores em Óbitos diários por Milhão (média de 7 dias)", pad = 30)

ax.title.set_size(30);
comportugal = piores.copy()

comportugal['Portugal'] = omatriz['Portugal']

ax = comportugal[-7:].plot(style = {'Portugal' : 'k-o'}, figsize=(25,10), linewidth = 5, colormap='Paired')

ax.set_xlabel('Data', fontsize=20, labelpad = 20)

ax.set_ylabel("Óbitos", fontsize=20, labelpad = 20)

ax.set_title("Piores em Óbitos diários por Milhão (média de 7 dias), comparados com Portugal", pad = 30)

ax.title.set_size(30);
asseguir = omatriz[omatriz.columns[0:10]]

ax = asseguir[-14:].plot(style = {'Portugal' : 'k-o'}, figsize=(25,10), linewidth = 5, colormap='Paired')

ax.set_xlabel('Date', fontsize=20, labelpad = 20)

ax.set_ylabel("Daily Deaths", fontsize=20, labelpad = 20)

ax.set_title("Daily deaths per milion - worst worldwide cases (7 days average)", pad = 30)

ax.title.set_size(30);
semanais = semanal.diff()[1:-1]

semanais = semanais[[col for col in semanais.columns if semanais.at[semanais.index[-1], col] == semanais[col].max()]]

semanais.sort_values(by = semanais.index[-1], axis=1, ascending=False, inplace=True, kind='quicksort', na_position='last')

ax = semanais[-10:].plot(figsize=(25,16), linewidth = 5, colormap='Paired');

ax.set_xlabel('Semana', fontsize=20, labelpad = 20)

ax.set_ylabel("Casos semanais", fontsize=20, labelpad = 20)

ax.set_title("Países em que a semana passada foi a pior de sempre", pad = 30)

ax.legend(loc='upper left')

ax.title.set_size(30);
"Há {} países em que a semana passada foi a pior de sempre".format(len([col for col in semanais.columns if semanais.at[semanais.index[-1], col] == semanais[col].max()]))
taxa = 3

cresce = 100*tabela.pct_change()

cresce = cresce.sort_values(cresce.index[-1], axis = 1, ascending = False)

crescemed = cresce.rolling(window=7).mean()

crescemed = crescemed.sort_values(crescemed.index[-1], axis = 1, ascending = False)

maiores = crescemed[[ crescemed.columns[i] for i in range(len(crescemed.columns)) if crescemed.at[crescemed.index[-1], crescemed.columns[i]] > taxa]]

maiores = maiores.reset_index()

maiores['Data'] = maiores['Data'].dt.date

maiores = maiores.set_index('Data')

maiores[-7:].plot(title = 'Países com crescimento diário maior do que {}%'.format(taxa), \

            figsize=(25,12), kind='bar', colormap='Paired');

plt.xlabel('Data', labelpad = 20)

plt.ylabel('Percentagem de crescimento', labelpad = 20);
casos = novo.copy()

casos = pd.pivot_table(casos, values='CumConfirmed', index=['date', 'Country/Region'], aggfunc=np.sum)

casos.reset_index(inplace = True)

casos = casos.merge(populacao, on = 'Country/Region',how= 'inner')

casos['Casos por Milhão'] = 1000000*casos['CumConfirmed'] / casos['População']

casos.rename(columns={'Country/Region': 'País'}, inplace = True)

cinteressa = casos[casos['País'].isin(ue)]

cinteressa = pd.pivot_table(cinteressa, values='Casos por Milhão', index=['date'], columns =['País'], aggfunc=np.sum)

if ~(cinteressa.at[cinteressa.index[-1],'Austria'] > 0):

    maxind = -2

else:

    maxind = -1

cinteressa.sort_values(by = cinteressa.index[maxind], axis=1, ascending=False, inplace=True, kind='quicksort', na_position='last')

ax = cinteressa[-96:].plot(style = {'Portugal' : 'k-o'}, figsize=(25,10), linewidth = 5, colormap='Paired')

ax.set_xlabel('Data', fontsize=15, labelpad = 20)

ax.set_ylabel("Casos", fontsize=15, labelpad = 20)

ax.set_title("Total de Casos por Milhão na União Europeia ", pad = 30)

ax.title.set_size(20);
obitos = novo.copy()

obitos = pd.pivot_table(obitos, values='CumDeaths', index=['date', 'Country/Region'], aggfunc=np.sum)

obitos.reset_index(inplace = True)

obitos = obitos.merge(populacao, on = 'Country/Region',how= 'inner')

obitos['Óbitos por Milhão'] = 1000000*obitos['CumDeaths'] / obitos['População']

obitos.rename(columns={'Country/Region': 'País'}, inplace = True)

ointeressa = obitos[obitos['País'].isin(ue)]

ointeressa = pd.pivot_table(ointeressa, values='Óbitos por Milhão', index=['date'], columns =['País'], aggfunc=np.sum)

#ointeressa = ointeressa.diff().rolling(window=7).mean()[-35:]

if ~(ointeressa.at[ointeressa.index[-1],'Austria'] > 0):

    maxind = -2

else:

    maxind = -1

ointeressa.sort_values(by = ointeressa.index[maxind], axis=1, ascending=False, inplace=True, kind='quicksort', na_position='last')

ax = ointeressa[-96:].plot(style = {'Portugal' : 'k-o'}, figsize=(25,10), linewidth = 5, colormap='Paired')

ax.set_xlabel('Data', fontsize=15, labelpad = 20)

ax.set_ylabel("Óbitos", fontsize=15, labelpad = 20)

ax.set_title("Total de Óbitos por Milhão na União Europeia ", pad = 30)

ax.title.set_size(20);
novos14 = cinteressa.diff(14)

novos14.sort_values(by = novos14.index[maxind], axis=1, ascending=False, inplace=True, kind='quicksort', na_position='last')

ax = novos14[-96:].plot(style = {'Portugal' : 'k-o'}, figsize=(20,10), linewidth = 5, colormap='Paired')

ax.set_xlabel('Data', fontsize=15, labelpad = 20)

ax.set_ylabel("Casos", fontsize=15, labelpad = 20)

ax.legend(loc='upper left')

ax.set_title("Casos por Milhão, nos últimos 14 dias, na União Europeia ", pad = 30)

ax.title.set_size(20);
popue = populacao[populacao['Country/Region'].isin(ue)]['População'].sum()

popport = populacao[populacao['Country/Region'] == 'Portugal']['População'].sum()

interessa = casos[casos['País'].isin(ue)]

interessa = pd.pivot_table(interessa, values='CumConfirmed', index=['date'], columns =['País'], aggfunc=np.sum)

if ~(interessa.at[interessa.index[-1],'Austria'] > 0):

    interessa = interessa[:-1]

interessa['União Europeia'] = interessa.sum(axis = 1)

interessa = interessa[['Portugal', 'União Europeia']]

interessa['Portugal'] = 100000*interessa['Portugal'] / popport

interessa['União Europeia'] = 100000*interessa['União Europeia'] /popue

interessa = interessa.diff(14)

interessa = interessa.sort_values(by=interessa.index[-1], axis = 1, ascending = False)

ax = interessa[-56:].plot(style = {'Portugal' : 'k-o'}, figsize=(25,10), linewidth = 5, colormap='Paired')

plt.ylim(0,50)

ax.set_xlabel('Data', fontsize=20, labelpad = 20)

ax.set_ylabel("Casos", fontsize=20, labelpad = 20)

ax.set_title("Casos por 100 mil nos últimos 14 dias - Portugal versus média UE", pad = 30)

ax.title.set_size(30);
interessa = obitos[obitos['País'].isin(ue)]

interessa = pd.pivot_table(interessa, values='CumDeaths', index=['date'], columns =['País'], aggfunc=np.sum)

if ~(interessa.at[interessa.index[-1],'Austria'] > 0):

    interessa = interessa[:-1]

interessa['União Europeia'] = interessa.sum(axis = 1)

interessa = interessa[['Portugal', 'União Europeia']]

interessa['Portugal'] = 100000*interessa['Portugal'] / popport

interessa['União Europeia'] = 100000*interessa['União Europeia'] /popue

interessa = interessa.diff(14)

interessa = interessa.sort_values(by=interessa.index[-1], axis = 1, ascending = False)

ax = interessa[-56:].plot(style = {'Portugal' : 'k-o'}, figsize=(25,10), linewidth = 5, colormap='Paired')

ax.set_xlabel('Data', fontsize=20, labelpad = 20)

ax.set_ylabel("Óbitos", fontsize=20, labelpad = 20)

ax.set_title("Óbitos por 100 mil nos últimos 14 dias - Portugal versus média UE", pad = 30)

plt.ylim(0,2.5)

ax.title.set_size(30);