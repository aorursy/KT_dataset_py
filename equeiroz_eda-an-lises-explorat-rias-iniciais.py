# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style("whitegrid")

sns.set_palette("muted")

from datetime import datetime, date, time
columns_to_read = ["id",

                   "data_inversa",

                   "horario",

                   "uf",                                

                   "br",              

                    "km",                                

                    "municipio",                         

                    "causa_acidente",                    

                    "tipo_acidente",                     

                    "classificacao_acidente",            

                    "fase_dia",                          

                    "sentido_via",                       

                    "condicao_metereologica",            

                    "tipo_pista",                        

                    "tracado_via",                       

                    "uso_solo",                                            

                    "pessoas",                            

                    "mortos",                             

                    "feridos_leves",                      

                    "feridos_graves",                     

                    "ilesos",                             

                    "ignorados",                          

                    "feridos",                            

                    "veiculos"] 
datatran2020 = pd.read_csv('../input/acidentes-rodovias-federais-brasil-jan07-a-jul19/datatran2020.csv', sep = ';', decimal=",", encoding = "ISO-8859-1", usecols=columns_to_read)

datatran2019 = pd.read_csv('../input/acidentes-rodovias-federais-brasil-jan07-a-jul19/datatran2019.csv', sep = ';', decimal=",", encoding = "ISO-8859-1", usecols=columns_to_read)

datatran2018 = pd.read_csv('../input/acidentes-rodovias-federais-brasil-jan07-a-jul19/datatran2018.csv', sep = ';', decimal=",", encoding = "ISO-8859-1", usecols=columns_to_read)

datatran2017 = pd.read_csv('../input/acidentes-rodovias-federais-brasil-jan07-a-jul19/datatran2017.csv', sep = ';', decimal=",", encoding = "ISO-8859-1", usecols=columns_to_read)

datatran2016 = pd.read_csv('../input/acidentes-rodovias-federais-brasil-jan07-a-jul19/datatran2016.csv', sep = ';', decimal=",", encoding = "ISO-8859-1", usecols=columns_to_read)

datatran2015 = pd.read_csv('../input/acidentes-rodovias-federais-brasil-jan07-a-jul19/datatran2015.csv', sep = ';', decimal=",", encoding = "ISO-8859-1", usecols=columns_to_read)

datatran2014 = pd.read_csv('../input/acidentes-rodovias-federais-brasil-jan07-a-jul19/datatran2014.csv', sep = ';', decimal=",", encoding = "ISO-8859-1", usecols=columns_to_read)

datatran2013 = pd.read_csv('../input/acidentes-rodovias-federais-brasil-jan07-a-jul19/datatran2013.csv', sep = ';', decimal=",", encoding = "ISO-8859-1", usecols=columns_to_read)

datatran2012 = pd.read_csv('../input/acidentes-rodovias-federais-brasil-jan07-a-jul19/datatran2012.csv', sep = ';', decimal=",", encoding = "ISO-8859-1", usecols=columns_to_read)

datatran2011 = pd.read_csv('../input/acidentes-rodovias-federais-brasil-jan07-a-jul19/datatran2011.csv', sep = ';', decimal=",", encoding = "ISO-8859-1", usecols=columns_to_read)

datatran2010 = pd.read_csv('../input/acidentes-rodovias-federais-brasil-jan07-a-jul19/datatran2010.csv', sep = ';', decimal=",", encoding = "ISO-8859-1", usecols=columns_to_read)

datatran2009 = pd.read_csv('../input/acidentes-rodovias-federais-brasil-jan07-a-jul19/datatran2009.csv', sep = ';', decimal=",", encoding = "ISO-8859-1", usecols=columns_to_read)

datatran2008 = pd.read_csv('../input/acidentes-rodovias-federais-brasil-jan07-a-jul19/datatran2008.csv', sep = ';', decimal=",", encoding = "ISO-8859-1", usecols=columns_to_read)

datatran2007 = pd.read_csv('../input/acidentes-rodovias-federais-brasil-jan07-a-jul19/datatran2007.csv', sep = ';', decimal=",", encoding = "ISO-8859-1", usecols=columns_to_read)
acidentes = pd.concat([datatran2007,

                   datatran2008,

                   datatran2009,

                   datatran2010,

                   datatran2011,

                   datatran2012,

                   datatran2013,

                   datatran2014,

                   datatran2015,

                   datatran2016,

                   datatran2017,

                   datatran2018,

                   datatran2019,

                   datatran2020], sort=False, ignore_index=True)
to_delete = [datatran2007,

             datatran2008,

                   datatran2009,

                   datatran2010,

                   datatran2011,

                   datatran2012,

                   datatran2013,

                   datatran2014,

                   datatran2015,

                   datatran2016,

                   datatran2017,

                   datatran2018,

                   datatran2019,

                   datatran2020]

del datatran2007, datatran2008, datatran2009, datatran2010, datatran2011, datatran2012, datatran2013, datatran2014, datatran2015, datatran2016, datatran2017, datatran2018, datatran2019, datatran2020

del to_delete

acidentes.info(memory_usage='deep')
acidentes.id = acidentes.id.astype('int32')

acidentes.pessoas = acidentes.pessoas.astype('uint8')

acidentes.mortos = acidentes.mortos.astype('uint8')

acidentes.feridos_leves = acidentes.feridos_leves.astype('uint8')

acidentes.feridos_graves = acidentes.feridos_graves.astype('uint8')

acidentes.ilesos = acidentes.ilesos.astype('uint8')

acidentes.ignorados = acidentes.ignorados.astype('uint8')

acidentes.feridos = acidentes.feridos.astype('uint8')

acidentes.veiculos = acidentes.veiculos.astype('uint8')

acidentes.uf = acidentes.uf.astype('category')

acidentes.br = acidentes.br.astype('category')

acidentes.municipio = acidentes.municipio.astype('category')

acidentes.causa_acidente = acidentes.causa_acidente.astype('category')

acidentes.tipo_acidente = acidentes.tipo_acidente.astype('category')

acidentes.classificacao_acidente = acidentes.classificacao_acidente.astype('category')

acidentes.fase_dia = acidentes.fase_dia.astype('category')

acidentes.sentido_via = acidentes.sentido_via.astype('category')

acidentes.condicao_metereologica = acidentes.condicao_metereologica.astype('category')

acidentes.tipo_pista = acidentes.tipo_pista.astype('category')

acidentes.tracado_via = acidentes.tracado_via.astype('category')

acidentes.uso_solo = acidentes.uso_solo.astype('category')
acidentes['data_hora'] = acidentes['data_inversa'].map(str) + ' ' + acidentes['horario']

acidentes['data_hora'] = pd.to_datetime(acidentes['data_hora'])

acidentes['ano'] = acidentes['data_hora'].dt.year

acidentes['mes'] = acidentes['data_hora'].dt.month

acidentes['hora'] = acidentes['data_hora'].dt.hour

acidentes.drop(['data_inversa', 'horario'], axis=1, inplace = True)
acidentes = acidentes.dropna()
acidentes.info(memory_usage='deep')
acidentes['municipio'] = acidentes['municipio'].str.rstrip()

acidentes['causa_acidente'] = acidentes['causa_acidente'].str.rstrip()

acidentes['tipo_acidente'] = acidentes['tipo_acidente'].str.rstrip()

acidentes['classificacao_acidente'] = acidentes['classificacao_acidente'].str.rstrip()

acidentes['fase_dia'] = acidentes['fase_dia'].str.rstrip()

acidentes['sentido_via'] = acidentes['sentido_via'].str.rstrip()

acidentes['condicao_metereologica'] = acidentes['condicao_metereologica'].str.rstrip()

acidentes['tipo_pista'] = acidentes['tipo_pista'].str.rstrip()

acidentes['tracado_via'] = acidentes['tracado_via'].str.rstrip()

acidentes['uso_solo'] = acidentes['uso_solo'].str.rstrip()

acidentes['uf'] = acidentes['uf'].str.upper()

acidentes['municipio'] = acidentes['municipio'].str.lower()

acidentes['causa_acidente'] = acidentes['causa_acidente'].str.lower()

acidentes['classificacao_acidente'] = acidentes['classificacao_acidente'].str.lower()

acidentes['fase_dia'] = acidentes['fase_dia'].str.lower()

acidentes['sentido_via'] = acidentes['sentido_via'].str.lower()

acidentes['tipo_pista'] = acidentes['tipo_pista'].str.lower()

acidentes['tracado_via'] = acidentes['tracado_via'].str.lower()

acidentes['uso_solo'] = acidentes['uso_solo'].str.lower()
sorted(acidentes.causa_acidente.unique())
mapa_causa_acidente = {'(null)' : '(null)',

    'agressão externa' : 'agressão externa',

 'animais na pista' : 'animais na pista',

 'avarias e/ou desgaste excessivo no pneu' : 'avarias e/ou desgaste excessivo no pneu',

 'carga excessiva e/ou mal acondicionada' : 'carga excessiva e/ou mal acondicionada',

 'condutor dormindo' : 'condutor dormindo',

 'defeito mecânico em veículo' : 'defeito mecânico no veículo',

 'defeito mecânico no veículo' : 'defeito mecânico no veículo',

 'defeito na via' : 'defeito na via',

 'deficiência ou não acionamento do sistema de iluminação/sinalização do veículo' : 'deficiência ou não acionamento do sistema de iluminação/sinalização do veículo',

 'desobediência à sinalização' : 'desobediência à sinalização',

 'desobediência às normas de trânsito pelo condutor' : 'desobediência às normas de trânsito pelo condutor',

 'desobediência às normas de trânsito pelo pedestre' : 'desobediência às normas de trânsito pelo pedestre',

 'dormindo' : 'condutor dormindo',

 'falta de atenção' : 'falta de atenção à condução',

 'falta de atenção do pedestre' : 'falta de atenção do pedestre',

 'falta de atenção à condução' : 'falta de atenção à condução',

 'fenômenos da natureza' : 'fenômenos da natureza',

 'ingestão de substâncias psicoativas' : 'ingestão de substâncias psicoativas',

 'ingestão de álcool' : 'ingestão de álcool',

 'ingestão de álcool e/ou substâncias psicoativas pelo pedestre' : 'ingestão de álcool e/ou substâncias psicoativas pelo pedestre',

 'mal súbito' : 'mal súbito',

 'não guardar distância de segurança' : 'não guardar distância de segurança',

 'objeto estático sobre o leito carroçável' : 'objeto estático sobre o leito carroçável',

 'outras' : 'outras',

 'pista escorregadia' : 'pista escorregadia',

 'restrição de visibilidade' : 'restrição de visibilidade',

 'sinalização da via insuficiente ou inadequada' : 'sinalização da via insuficiente ou inadequada',

 'ultrapassagem indevida' : 'ultrapassagem indevida',

 'velocidade incompatível' : 'velocidade incompatível'}

acidentes.causa_acidente = acidentes.causa_acidente.map(mapa_causa_acidente)
sorted(acidentes.tipo_acidente.unique())
mapa_tipo_acidente = {'Atropelamento de Animal' : 'Atropelamento de animal',

 'Atropelamento de Pedestre' : 'Atropelamento de pedestre',

 'Atropelamento de animal' : 'Atropelamento de animal',

 'Atropelamento de pessoa' : 'Atropelamento de pedestre',

 'Capotamento' : 'Capotamento',

 'Colisão Transversal' : 'Colisão transversal',

 'Colisão com bicicleta' : 'Colisão com bicicleta',

 'Colisão com objeto em movimento' : 'Colisão com objeto em movimento',

 'Colisão com objeto estático' : 'Colisão com objeto estático',

 'Colisão com objeto fixo' : 'Colisão com objeto estático',

 'Colisão com objeto móvel' : 'Colisão com objeto em movimento',

 'Colisão frontal' : 'Colisão frontal',

 'Colisão lateral' : 'Colisão lateral',

 'Colisão transversal' : 'Colisão transversal',

 'Colisão traseira' : 'Colisão traseira',

 'Danos Eventuais' : 'Danos eventuais',

 'Danos eventuais' : 'Danos eventuais',

 'Derramamento de Carga' : 'Derramamento de carga',

 'Derramamento de carga' : 'Derramamento de carga',

 'Engavetamento' : 'Engavetamento',

 'Incêndio' : 'Incêndio',

 'Queda de motocicleta / bicicleta / veículo' : 'Queda de ocupante de veículo',

 'Queda de ocupante de veículo' : 'Queda de ocupante de veículo',

 'Saída de Pista' : 'Saída de pista',

 'Saída de leito carroçável' : 'Saída de pista',

 'Tombamento' : 'Tombamento'}

acidentes.tipo_acidente = acidentes.tipo_acidente.map(mapa_tipo_acidente)
acidentes.condicao_metereologica.unique()
mapa_condicao_metereologica = {'Ceu Claro' : 'Céu Claro',

                               'Chuva' : 'Chuva',

                               'Nublado' : 'Nublado',

                               'Sol' : 'Sol',

                               'Nevoeiro/neblina' : 'Nevoeiro/Neblina',

                               'Ignorada' : 'Ignorada',

                               'Vento' : 'Vento',

                               'Granizo' : 'Granizo',

                               '(null)' : '(null)',

                               'Neve' : 'Neve',

                               'Garoa/Chuvisco' : 'Garoa/Chuvisco',

                               'Céu Claro' : 'Céu Claro',

                               'Ignorado' : 'Ignorada',

                               'Nevoeiro/Neblina' : 'Nevoeiro/Neblina'}
acidentes.condicao_metereologica = acidentes.condicao_metereologica.map(mapa_condicao_metereologica)
selecao = (acidentes.mortos != 0) | (acidentes.feridos_graves !=0)

acidentes_com_mortos_ou_feridos_graves = acidentes[selecao]
df_ano_mes_acidentes_graves = acidentes_com_mortos_ou_feridos_graves.groupby(by=['ano', 'mes'])['mes'].count().reset_index(name='Quantidade')

df_ano_mes_acidentes_graves['ano_mes'] = df_ano_mes_acidentes_graves.ano.map(str) + "-" + df_ano_mes_acidentes_graves.mes.map(str)
ax = sns.lineplot(x = 'ano_mes', y = 'Quantidade', data = df_ano_mes_acidentes_graves)

ax.figure.set_size_inches(22,8)

ax.xaxis.set_major_locator(plt.MaxNLocator(10))

ax.set_title("Quantidade de acidentes com mortos ou feridos graves - Geral Jan/2007 a Dez/2019", fontsize=20)

ax.set_xlabel('Mês-ano (Ex: 2013-1: Jan/2013)')
ax = sns.barplot(x=acidentes_com_mortos_ou_feridos_graves.causa_acidente.value_counts(),

                 y=acidentes_com_mortos_ou_feridos_graves.causa_acidente.value_counts().index)

ax.set_title("Quantidade de acidentes com mortos ou feridos graves por Tipo de Causa - Geral Jan/2007 a Dez/2019", fontsize=20)

ax.set_xlabel('')

ax.figure.set_size_inches(8,12)
df_uf_causa_acidente = acidentes_com_mortos_ou_feridos_graves.groupby(by=['uf', 'causa_acidente'])['causa_acidente'].count().reset_index(name='Quantidade')

df_uf_causa_acidente.uf.unique()
todas_ufs = ['AL', 'AM', 'AP', 'BA', 'CE', 'DF', 'ES', 'GO',

       'MA', 'MG', 'MS', 'MT', 'PA', 'PB', 'PE', 'PI', 'PR', 'RJ', 'RN',

       'RO', 'RR', 'RS', 'SC', 'SE', 'SP', 'TO']
df_uf_causa_acidente_top5 = df_uf_causa_acidente[df_uf_causa_acidente.uf == 'AC'].nlargest(5,'Quantidade')

for UF in todas_ufs:

    df_uf_causa_acidente_top5 = pd.concat([df_uf_causa_acidente_top5, df_uf_causa_acidente[df_uf_causa_acidente.uf == UF].nlargest(5,'Quantidade')])
ax = sns.FacetGrid(df_uf_causa_acidente_top5, col="uf", col_wrap = 3, height = 4)

ax.map(sns.barplot, "Quantidade", "causa_acidente")
df_br_uf_mortos = acidentes.groupby(by=['uf', 'br'])['mortos'].count().reset_index(name='Quantidade')

df2 = acidentes.groupby(by=['uf', 'br'])['mortos'].sum().reset_index(name='Mortos')

df_br_uf_mortos = df_br_uf_mortos.join(df2.Mortos)

df_br_uf_mortos['Mortos por 100 acidentes'] = 100*(df_br_uf_mortos.Mortos / df_br_uf_mortos.Quantidade)

df_br_uf_mortos.nlargest(10,'Mortos')
df_br_uf_mortos[df_br_uf_mortos.Mortos > 100].nlargest(10,'Mortos por 100 acidentes')
df_municipio_mortos = acidentes.groupby(by=['municipio'])['mortos'].count().reset_index(name='Quantidade')

df2 = acidentes.groupby(by=['municipio'])['mortos'].sum().reset_index(name='Mortos')

df_municipio_mortos = df_municipio_mortos.join(df2.Mortos)

df_municipio_mortos['Mortos por 100 acidentes'] = 100*(df_municipio_mortos.Mortos / df_municipio_mortos.Quantidade)

df_municipio_mortos.nlargest(10,'Mortos')
df_municipio_mortos[df_municipio_mortos.Mortos > 100].nlargest(10,'Mortos por 100 acidentes')
df_hora_mortos = acidentes_com_mortos_ou_feridos_graves.groupby(by=['hora'])['mortos'].sum().reset_index(name='mortos')
ax = sns.barplot(x='hora',

                 y='mortos',

                 data = df_hora_mortos,

                 palette="Blues_d")

ax.set_title("Quantidade de mortos por hora do dia - Geral Jan/2007 a Dez/2019", fontsize=20)

ax.set_xlabel('')

ax.figure.set_size_inches(12,8)
acidentes.tail(50)
df_ano_mes_acidentes_graves.tail(24)