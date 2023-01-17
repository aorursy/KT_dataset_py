%matplotliblib inline



import pandas as pd

import matplotlib

import numpy as np

import matplotlib.pyplot as plt



# Set ipython's max row display

#pd.set_option('display.max_row', 1000)



# Set iPython's max column width to 50

pd.set_option('display.max_columns', 50)
update_files = False

if(update_files):

    import urllib.request

    urllib.request.urlretrieve("http://www.ispdados.rj.gov.br/Arquivos/BaseDPEvolucaoMensalCisp.csv", "data/BaseDPEvolucaoMensalCisp.csv")

    urllib.request.urlretrieve("http://www.ispdados.rj.gov.br/Arquivos/PopulacaoEvolucaoMensalCisp.xlsx", "data/PopulacaoEvolucaoMensalCisp.xlsx")
data = pd.read_csv('../input/BaseDPEvolucaoMensalCisp.csv', encoding = "ISO-8859-1", sep=";")

data_pop = pd.read_csv('../input/PopulacaoEvolucaoMensalCisp.csv',sep=";")



print(data.shape, data_pop.shape)
data.tail(5)
data_pop.tail(5)
data_pop['CISP'] = data_pop['circ']

data_join = pd.merge(data, data_pop, how="inner", on=['CISP', 'mes', 'vano'])

data_join.shape
#Normalizar para 100.000 habitantes

data_join['per100k_ratio'] = data_join['pop_circ'] / 100000
#Limpeza de NaN

#Aplicando a média por CISP

cols = ['hom_doloso', 'lesao_corp_morte',

       'latrocinio', 'tentat_hom', 'lesao_corp_dolosa', 'estupro',

       'hom_culposo', 'lesao_corp_culposa', 'encontro_cadaver',

       'encontro_ossada', 'roubo_comercio', 'roubo_residencia',

       'roubo_veiculo', 'roubo_carga', 'roubo_transeunte',

       'roubo_em_coletivo', 'roubo_banco', 'roubo_cx_eletronico',

       'roubo_celular', 'roubo_conducao_saque', 'roubo_bicicleta',

       'outros_roubos', 'furto_veiculos', 'furto_bicicleta',

       'outros_furtos', 'sequestro', 'extorsao', 'sequestro_relampago',

       'estelionato', 'apreensao_drogas', 'recuperacao_veiculos',

       'cmp',# OLD 'cump_mandado_prisao',

       'ameaca', 'pessoas_desaparecidas',

       'hom_por_interv_policial', 

       # retirado 'armas_apreendidas',

        #'prisoes', 'grp',

       'apf', #old 'apf_cmp', 'apreensoes', 'gaai', 'aaapai_cmba',

       'registro_ocorrencias', 'pol_militares_mortos_serv',

       'pol_civis_mortos_serv', 'indicador_letalidade',

       'indicador_roubo_rua', 'indicador_roubo_veic']



for c in cols:

    if(data_join[c].isnull().sum() > 0):

        data_join[c] = data_join[['CISP', c]].groupby('CISP').transform(lambda x: x.fillna(np.round(x.mean())))
#Criação de coluna Data

from datetime import date

data_join['data'] = pd.to_datetime(data_join[['vano', 'mes']].apply(lambda x : date(x['vano'], x['mes'],1), axis=1))

data_join['data'].head(5)
data_join.describe()
data_per100k = data_join.copy()

for c in cols:

    data_per100k[c] = data_per100k[c] * data_per100k['per100k_ratio']
data_per100k.head(5)
data_per100k[data_per100k['vano'] == 2016].sort_values('roubo_veiculo', ascending=False)[['Regiao','munic','roubo_veiculo']]
data_per100k[data_per100k['vano'] == 2016].groupby(['CISP','munic']).sum().sort_values('roubo_veiculo', ascending=False)[['roubo_veiculo']].head(5)
import datetime as dt

def mark_period(init, end, color, style="--", width=1, interval=29):

    ax.axvline(init, color=color, linestyle=style, lw=width + 1)

    ax.axvline(end, color=color, linestyle=style, lw=width + 1)

    for i in range((end - init).days):

        if(i % interval == 0):

            ax.axvline(init + dt.timedelta(days=i), color=color, linestyle=style, lw=1)
#Governo Rosinha

data_per100k.groupby(['data']).sum()[['indicador_letalidade','indicador_roubo_rua','indicador_roubo_veic']].plot(figsize=(15,6))



ax = plt.gca()



#GOVERNO

#ax.axvline(pd.to_datetime('2007-01-1'), color='#920000', linestyle='--', lw=2)

mark_period(dt.datetime(2003,1,1), dt.datetime(2007,1,1), '#CC6666')



plt.tick_params(bottom='off', top='off', left='off', right='off')

for spine in ax.spines:

    ax.spines[spine].set_visible(False)

ax.set_title('Governo Rosinha Garotinho')

ax.legend(loc='upper left')

plt.show()
#Governo Cabral

data_per100k.groupby(['data']).sum()[['indicador_letalidade','indicador_roubo_rua','indicador_roubo_veic']].plot(figsize=(15,6))

ax = plt.gca()



#GOVERNO

mark_period(dt.datetime(2007,1,1), dt.datetime(2011,1,1), '#CC6666')

mark_period(dt.datetime(2014,4,4), dt.datetime(2015,1,1), '#FFCCCC')

mark_period(dt.datetime(2011,1,1), dt.datetime(2014,4,3), '#CC6666')



plt.tick_params(bottom='off', top='off', left='off', right='off')

for spine in ax.spines:

    ax.spines[spine].set_visible(False)

ax.set_title('Governo Sérgio Cabral Filho')

plt.show()
#Governo Luiz Fernando Pezão

data_per100k.groupby(['data']).sum()[['indicador_letalidade','indicador_roubo_rua','indicador_roubo_veic']].plot(figsize=(15,6))

ax = plt.gca()



#GOVERNO

ax.axvline(pd.to_datetime('2014-04-3'), color='#CC6666', linestyle='--', lw=2)

ax.axvline(pd.to_datetime('2015-01-1'), color='#CC6666', linestyle='--', lw=2)

ax.axvline(pd.to_datetime('2016-03-28'), color='#CC6666', linestyle='--', lw=1)

ax.axvline(pd.to_datetime('2016-10-31'), color='#CC6666', linestyle='--', lw=1)

ax.axvline(pd.to_datetime('2018-12-31'), color='#CC6666', linestyle='--', lw=2)



mark_period(dt.datetime(2016,3,28), dt.datetime(2016,10,31), '#FFCCCC')

mark_period(dt.datetime(2014,4,3), dt.datetime(2015,1,1), '#CC6666')

mark_period(dt.datetime(2015,1,1), dt.datetime(2016,3,28), '#CC6666')

mark_period(dt.datetime(2016,10,31), dt.datetime(2018,11,29), '#CC6666')





plt.tick_params(bottom='off', top='off', left='off', right='off')

for spine in ax.spines:

    ax.spines[spine].set_visible(False)

ax.set_title('Governo Luiz Fernando Pezão')

plt.show()
data_per100k.groupby(['data']).sum()[['indicador_letalidade','indicador_roubo_rua','indicador_roubo_veic']].plot(figsize=(15,6))



ax = plt.gca()

#Governo Dorneles

mark_period(dt.datetime(2016,3,28), dt.datetime(2016,10,31), '#CC6666')





plt.tick_params(bottom='off', top='off', left='off', right='off')

for spine in ax.spines:

    ax.spines[spine].set_visible(False)

ax.set_title('Governo interino Dorneles')

plt.show()
data_per100k.groupby(['data']).sum()[['indicador_letalidade','indicador_roubo_rua','indicador_roubo_veic']].plot(figsize=(15,6))



ax = plt.gca()

#Governo Witzel

mark_period(dt.datetime(2019,1,1), dt.datetime.now(), '#CC6666')





plt.tick_params(bottom='off', top='off', left='off', right='off')

for spine in ax.spines:

    ax.spines[spine].set_visible(False)

ax.set_title('Governo Witzel')

plt.show()
data_per100k.groupby(['data']).sum()[['indicador_letalidade','indicador_roubo_rua','indicador_roubo_veic']].plot(figsize=(15,6))



ax = plt.gca()



#06

ax.axvline(pd.to_datetime('2006-10-01'), color='#920000', linestyle='--', lw=2)



#08

ax.axvline(pd.to_datetime('2008-10-05'), color='#000092', linestyle='--', lw=2)



#10

ax.axvline(pd.to_datetime('2010-10-03'), color='#920000', linestyle='--', lw=2)



#12

ax.axvline(pd.to_datetime('2012-10-05'), color='#000092', linestyle='--', lw=2)



#14

ax.axvline(pd.to_datetime('2014-10-05'), color='#920000', linestyle='--', lw=2)



#16

ax.axvline(pd.to_datetime('2016-10-02'), color='#000092', linestyle='--', lw=2)



#18

ax.axvline(pd.to_datetime('2018-10-02'), color='#920000', linestyle='--', lw=2)





plt.tick_params(bottom='off', top='off', left='off', right='off')

for spine in ax.spines:

    ax.spines[spine].set_visible(False)

ax.set_title('Eleições')

plt.show()
data_per100k.groupby(['data']).sum()[['indicador_letalidade','indicador_roubo_rua','indicador_roubo_veic']].plot(figsize=(15,6))



ax = plt.gca()



#Cerco Alemão - Vila Cruzeiro

ax.axvline(pd.to_datetime('2010-11-25'), color='#920000', linestyle='--', lw=1)



#Manifestações

ax.axvline(pd.to_datetime('2013-06-20'), color='#009292', linestyle='--', lw=1)



#copa 2014

mark_period(dt.datetime(2014,6,12), dt.datetime(2014,7,13), '#000092', interval=2)





#Ocupação da Maré

mark_period(dt.datetime(2014,4,5), dt.datetime(2015,6,30), '#000000', interval=2)



#olimpiadas

mark_period(dt.datetime(2016,8,15), dt.datetime(2016,8,21), '#009200', interval=2)



#Greve Polícia Civil

mark_period(dt.datetime(2017,1,20), dt.datetime(2017,4,7), 'r', interval=2)









plt.tick_params(bottom='off', top='off', left='off', right='off')

for spine in ax.spines:

    ax.spines[spine].set_visible(False)

ax.set_title('Eleições')

plt.show()
data_per100k.groupby(['data']).sum()[['indicador_letalidade','indicador_roubo_rua','indicador_roubo_veic']].plot(figsize=(15,6))



ax = plt.gca()



#Ocupação da Maré

mark_period(dt.datetime(2014,4,5), dt.datetime(2015,6,30), '#888888')



#Renuncia Cabral e inicio do Governo Pezão

ax.axvline(pd.to_datetime('2014-04-3'), color='#CC6666', linestyle='--', lw=2)



#Greve Polícia Civil

mark_period(dt.datetime(2017,1,20), dt.datetime(2017,4,7), '#66CC66')



#Eleições Municipais de 2012

ax.axvline(pd.to_datetime('2012-10-05'), color='#6666FF', linestyle='-', lw=2)



#Governo Dorneles

mark_period(dt.datetime(2016,3,28), dt.datetime(2016,10,31), '#FF6666')



plt.tick_params(bottom='off', top='off', left='off', right='off')

for spine in ax.spines:

    ax.spines[spine].set_visible(False)

ax.set_title('Destaques')

plt.show()