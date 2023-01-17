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
t1 = pd.read_csv('../input/dataset-estacao-e-telhado-full1/Dataset_Telhado_Verde_2019_1.csv', delimiter=';', decimal=',')

t2 = pd.read_csv('../input/dataset-estacao-e-telhado-full1/Dataset_Telhado_Verde_2019_2.csv', delimiter=';', decimal=',')

t3 = pd.read_csv('../input/dataset-estacao-e-telhado-full1/Dataset_Telhado_Verde_2019_3.csv', delimiter=';', decimal=',')

telhado = pd.concat([t1, t2, t3])
telhado.head(10)
clima = pd.read_csv('../input/dataset-estacao-e-telhado-full1/Dataset_Estacao_01_01_2018_a_13_09_2019.csv', delimiter=',')
clima.head(10)
telhado = telhado.drop_duplicates(keep='first')
telhado = telhado.sort_values(['data','sensor'])
#Converte coluna data em DateTime

telhado['data'] = pd.to_datetime(telhado['data'])

#Cria novas colunas

telhado['Data'] = telhado['data'].dt.strftime('%Y-%m-%d')

telhado['Hora'] = telhado['data'].dt.strftime('%H:%M:%S')

telhado['hora'] = telhado['data'].dt.strftime('%H')

telhado['minuto'] = telhado['data'].dt.strftime('%M')

telhado['segundo'] = telhado['data'].dt.strftime('%S')

#Converte coluna Data em DateTime

telhado['Data'] = pd.to_datetime(telhado['Data'])

#Converte colunas hora, minuto e segundo em Integer

telhado['hora'] = pd.to_numeric(telhado['hora'], downcast='integer')

telhado['minuto'] = pd.to_numeric(telhado['minuto'], downcast='integer')

telhado['segundo'] = pd.to_numeric(telhado['segundo'], downcast='integer')
telhado.head(10)
#Converte coluna data em DateTime

clima['data'] = pd.to_datetime(clima['data'])

#Cria novas colunas

clima['Data'] = clima['data'].dt.strftime('%Y-%m-%d')

clima['Hora'] = clima['data'].dt.strftime('%H:%M:%S')

clima['hora'] = clima['data'].dt.strftime('%H')

clima['minuto'] = clima['data'].dt.strftime('%M')

clima['segundo'] = clima['data'].dt.strftime('%S')

#Converte coluna Data em DateTime

clima['Data'] = pd.to_datetime(clima['Data'])

#Converte colunas hora, minuto e segundo em Integer

clima['hora'] = pd.to_numeric(clima['hora'], downcast='integer')

clima['minuto'] = pd.to_numeric(clima['minuto'], downcast='integer')

clima['segundo'] = pd.to_numeric(clima['segundo'], downcast='integer')
clima.head(10)
telhado = telhado[(telhado['temperatura'] > 10) & (telhado['temperatura'] < 45)]
telhado
# Verificando a porcetagem de valores faltantes por coluna para definir como contorna-los

atributos_missing = []



for f in telhado.columns:

    missings = telhado[f].isnull().sum()

    if missings > 0:

        atributos_missing.append(f)

        missings_perc = missings/telhado.shape[0]

        

        print('Atributo {} tem {} amostras ({:.2%}) com valores faltantes'.format(f, missings, missings_perc))



print('No total, há {} atributos com valores faltantes'.format(len(atributos_missing)))
telhado_flat = telhado[(telhado['sensor'] == 1) | (telhado['sensor'] == 2)]
telhado_modular = telhado[(telhado['sensor'] == 3) | (telhado['sensor'] == 4)]
telhado_macdrain = telhado[(telhado['sensor'] == 5) | (telhado['sensor'] == 6)]
# Quantidade de temperaturas por data, hora, minuto, segundo, para verificar quantos sensores estão ligados em cada data, hora, minuto, segundo

datas_telhado = pd.DataFrame(telhado.groupby(['data'])['temperatura'].count())
datas_6_sensores = pd.DataFrame(datas_telhado.loc[datas_telhado['temperatura'] == 6])
list_datas_6_sensores = datas_6_sensores.index
# Dataset com somente os dias, horas, minutos e segundos com 6 temperaturas capturadas

telhado = telhado[telhado['data'].isin(list_datas_6_sensores)]
pivot_table_telhado = telhado.pivot_table('temperatura', ['data'], 'sensor')



telhado_pivot = pd.DataFrame(pivot_table_telhado.to_records())

telhado_pivot.rename(columns={'1': 'temp_sensor1', 

                              '2': 'temp_sensor2',

                              '3': 'temp_sensor3',

                              '4': 'temp_sensor4',

                              '5': 'temp_sensor5',

                              '6': 'temp_sensor6'}, inplace=True)
#Converte coluna data em DateTime

telhado_pivot['data'] = pd.to_datetime(telhado_pivot['data'])

#Cria novas colunas

telhado_pivot['Data'] = telhado_pivot['data'].dt.strftime('%Y-%m-%d')

telhado_pivot['Hora'] = telhado_pivot['data'].dt.strftime('%H:%M:%S')

telhado_pivot['hora'] = telhado_pivot['data'].dt.strftime('%H')

telhado_pivot['minuto'] = telhado_pivot['data'].dt.strftime('%M')

telhado_pivot['segundo'] = telhado_pivot['data'].dt.strftime('%S')

#Converte coluna Data em DateTime

telhado_pivot['Data'] = pd.to_datetime(telhado_pivot['Data'])

#Converte colunas hora, minuto e segundo em Integer

telhado_pivot['hora'] = pd.to_numeric(telhado_pivot['hora'], downcast='integer')

telhado_pivot['minuto'] = pd.to_numeric(telhado_pivot['minuto'], downcast='integer')

telhado_pivot['segundo'] = pd.to_numeric(telhado_pivot['segundo'], downcast='integer')
telhado_pivot['Delta_1_2'] = telhado_pivot['temp_sensor1'] - telhado_pivot['temp_sensor2']

telhado_pivot['Delta_3_4'] = telhado_pivot['temp_sensor3'] - telhado_pivot['temp_sensor4']

telhado_pivot['Delta_5_6'] = telhado_pivot['temp_sensor5'] - telhado_pivot['temp_sensor6']
telhado_pivot
pivot_table_telhado_flat = telhado_flat.pivot_table('temperatura', ['data'], 'sensor')



telhado_pivot_flat = pd.DataFrame(pivot_table_telhado_flat.to_records())

telhado_pivot_flat.rename(columns={'1': 'temp_sensor1', 

                                   '2': 'temp_sensor2'}, inplace=True)
# Verificando a porcetagem de valores faltantes por coluna para definir como contorna-los

atributos_missing = []



for f in telhado_pivot_flat.columns:

    missings = telhado_pivot_flat[f].isnull().sum()

    if missings > 0:

        atributos_missing.append(f)

        missings_perc = missings/telhado_pivot_flat.shape[0]

        

        print('Atributo {} tem {} amostras ({:.2%}) com valores faltantes'.format(f, missings, missings_perc))



print('No total, há {} atributos com valores faltantes'.format(len(atributos_missing)))
telhado_pivot_flat.dropna(inplace=True)
#Converte coluna data em DateTime

telhado_pivot_flat['data'] = pd.to_datetime(telhado_pivot_flat['data'])

#Cria novas colunas

telhado_pivot_flat['Data'] = telhado_pivot_flat['data'].dt.strftime('%Y-%m-%d')

telhado_pivot_flat['Hora'] = telhado_pivot_flat['data'].dt.strftime('%H:%M:%S')

telhado_pivot_flat['hora'] = telhado_pivot_flat['data'].dt.strftime('%H')

telhado_pivot_flat['minuto'] = telhado_pivot_flat['data'].dt.strftime('%M')

telhado_pivot_flat['segundo'] = telhado_pivot_flat['data'].dt.strftime('%S')

#Converte coluna Data em DateTime

telhado_pivot_flat['Data'] = pd.to_datetime(telhado_pivot_flat['Data'])

#Converte colunas hora, minuto e segundo em Integer

telhado_pivot_flat['hora'] = pd.to_numeric(telhado_pivot_flat['hora'], downcast='integer')

telhado_pivot_flat['minuto'] = pd.to_numeric(telhado_pivot_flat['minuto'], downcast='integer')

telhado_pivot_flat['segundo'] = pd.to_numeric(telhado_pivot_flat['segundo'], downcast='integer')
telhado_pivot_flat['Delta_1_2'] = telhado_pivot_flat['temp_sensor1'] - telhado_pivot_flat['temp_sensor2']
telhado_pivot_flat
pivot_table_telhado_modular = telhado_modular.pivot_table('temperatura', ['data'], 'sensor')



telhado_pivot_modular = pd.DataFrame(pivot_table_telhado_modular.to_records())

telhado_pivot_modular.rename(columns={'3': 'temp_sensor3', 

                                      '4': 'temp_sensor4'}, inplace=True)
# Verificando a porcetagem de valores faltantes por coluna para definir como contorna-los

atributos_missing = []



for f in telhado_pivot_modular.columns:

    missings = telhado_pivot_modular[f].isnull().sum()

    if missings > 0:

        atributos_missing.append(f)

        missings_perc = missings/telhado_pivot_modular.shape[0]

        

        print('Atributo {} tem {} amostras ({:.2%}) com valores faltantes'.format(f, missings, missings_perc))



print('No total, há {} atributos com valores faltantes'.format(len(atributos_missing)))
telhado_pivot_modular.dropna(inplace=True)
#Converte coluna data em DateTime

telhado_pivot_modular['data'] = pd.to_datetime(telhado_pivot_modular['data'])

#Cria novas colunas

telhado_pivot_modular['Data'] = telhado_pivot_modular['data'].dt.strftime('%Y-%m-%d')

telhado_pivot_modular['Hora'] = telhado_pivot_modular['data'].dt.strftime('%H:%M:%S')

telhado_pivot_modular['hora'] = telhado_pivot_modular['data'].dt.strftime('%H')

telhado_pivot_modular['minuto'] = telhado_pivot_modular['data'].dt.strftime('%M')

telhado_pivot_modular['segundo'] = telhado_pivot_modular['data'].dt.strftime('%S')

#Converte coluna Data em DateTime

telhado_pivot_modular['Data'] = pd.to_datetime(telhado_pivot_modular['Data'])

#Converte colunas hora, minuto e segundo em Integer

telhado_pivot_modular['hora'] = pd.to_numeric(telhado_pivot_modular['hora'], downcast='integer')

telhado_pivot_modular['minuto'] = pd.to_numeric(telhado_pivot_modular['minuto'], downcast='integer')

telhado_pivot_modular['segundo'] = pd.to_numeric(telhado_pivot_modular['segundo'], downcast='integer')
telhado_pivot_modular['Delta_3_4'] = telhado_pivot_modular['temp_sensor3'] - telhado_pivot_modular['temp_sensor4']
telhado_pivot_modular
pivot_table_telhado_macdrain = telhado_macdrain.pivot_table('temperatura', ['data'], 'sensor')



telhado_pivot_macdrain = pd.DataFrame(pivot_table_telhado_macdrain.to_records())

telhado_pivot_macdrain.rename(columns={'5': 'temp_sensor5', 

                                       '6': 'temp_sensor6'}, inplace=True)
# Verificando a porcetagem de valores faltantes por coluna para definir como contorna-los

atributos_missing = []



for f in telhado_pivot_macdrain.columns:

    missings = telhado_pivot_macdrain[f].isnull().sum()

    if missings > 0:

        atributos_missing.append(f)

        missings_perc = missings/telhado_pivot_macdrain.shape[0]

        

        print('Atributo {} tem {} amostras ({:.2%}) com valores faltantes'.format(f, missings, missings_perc))



print('No total, há {} atributos com valores faltantes'.format(len(atributos_missing)))
telhado_pivot_macdrain.dropna(inplace=True)
#Converte coluna data em DateTime

telhado_pivot_macdrain['data'] = pd.to_datetime(telhado_pivot_macdrain['data'])

#Cria novas colunas

telhado_pivot_macdrain['Data'] = telhado_pivot_macdrain['data'].dt.strftime('%Y-%m-%d')

telhado_pivot_macdrain['Hora'] = telhado_pivot_macdrain['data'].dt.strftime('%H:%M:%S')

telhado_pivot_macdrain['hora'] = telhado_pivot_macdrain['data'].dt.strftime('%H')

telhado_pivot_macdrain['minuto'] = telhado_pivot_macdrain['data'].dt.strftime('%M')

telhado_pivot_macdrain['segundo'] = telhado_pivot_macdrain['data'].dt.strftime('%S')

#Converte coluna Data em DateTime

telhado_pivot_macdrain['Data'] = pd.to_datetime(telhado_pivot_macdrain['Data'])

#Converte colunas hora, minuto e segundo em Integer

telhado_pivot_macdrain['hora'] = pd.to_numeric(telhado_pivot_macdrain['hora'], downcast='integer')

telhado_pivot_macdrain['minuto'] = pd.to_numeric(telhado_pivot_macdrain['minuto'], downcast='integer')

telhado_pivot_macdrain['segundo'] = pd.to_numeric(telhado_pivot_macdrain['segundo'], downcast='integer')
telhado_pivot_macdrain['Delta_5_6'] = telhado_pivot_macdrain['temp_sensor5'] - telhado_pivot_macdrain['temp_sensor6']
telhado_pivot_macdrain
def categoriza_SF(s):

    if s == 1 or s == 2:

        return 1

    else:

        return 0

    

def categoriza_SM(s):

    if s == 3 or s == 4:

        return 1

    else:

        return 0

    

def categoriza_SMD(s):

    if s == 5 or s == 6:

        return 1

    else:

        return 0
from datetime import datetime



def categoriza_Dia(hora):

    if hora > 6 and hora < 18:

        return 1

    else:

        return 0

    

def categoriza_Noite(hora):

    if (hora >= 0 and hora <= 6) or (hora >= 18 and hora <= 23):

        return 1

    else:

        return 0

    

def categoriza_Verao(data):

    ano = data.year

    dt_from_1 = datetime.strptime('2018-12-21', '%Y-%m-%d')

    dt_to_1 = datetime.strptime('2018-03-19', '%Y-%m-%d')

    dt_from_2 = datetime.strptime('2019-12-21', '%Y-%m-%d')

    dt_to_2 = datetime.strptime('2019-03-19', '%Y-%m-%d')

    if ano == 2018:

        if data >= dt_from_1:

            return 1

        else:

            if data <= dt_to_1:

                   return 1

            else:

                return 0

    else:

        if ano == 2019:

            if data >= dt_from_2:

                return 1

            else:

                if data <= dt_to_2:

                       return 1

                else:

                    return 0

        else:

            return 0

        

def categoriza_Outono(data):

    ano = data.year

    dt_from_1 = datetime.strptime('2018-03-20', '%Y-%m-%d')

    dt_to_1 = datetime.strptime('2018-06-20', '%Y-%m-%d')

    dt_from_2 = datetime.strptime('2019-03-20', '%Y-%m-%d')

    dt_to_2 = datetime.strptime('2019-06-20', '%Y-%m-%d')

    if ano == 2018:

        if data >= dt_from_1 and data <= dt_to_1:

            return 1

        else:

            return 0

    else:

        if ano == 2019:

            if data >= dt_from_2 and data <= dt_to_2:

                return 1

            else:

                return 0

        else:

            return 0

        

def categoriza_Inverno(data):

    ano = data.year

    dt_from_1 = datetime.strptime('2018-06-21', '%Y-%m-%d')

    dt_to_1 = datetime.strptime('2018-09-21', '%Y-%m-%d')

    dt_from_2 = datetime.strptime('2019-06-21', '%Y-%m-%d')

    dt_to_2 = datetime.strptime('2019-09-21', '%Y-%m-%d')

    if ano == 2018:

        if data >= dt_from_1 and data <= dt_to_1:

            return 1

        else:

            return 0

    else:

        if ano == 2019:

            if data >= dt_from_2 and data <= dt_to_2:

                return 1

            else:

                return 0

        else:

            return 0

        

def categoriza_Primavera(data):

    ano = data.year

    dt_from_1 = datetime.strptime('2018-09-22', '%Y-%m-%d')

    dt_to_1 = datetime.strptime('2018-12-20', '%Y-%m-%d')

    dt_from_2 = datetime.strptime('2019-09-22', '%Y-%m-%d')

    dt_to_2 = datetime.strptime('2019-12-20', '%Y-%m-%d')

    if ano == 2018:

        if data >= dt_from_1 and data <= dt_to_1:

            return 1

        else:

            return 0

    else:

        if ano == 2019:

            if data >= dt_from_2 and data <= dt_to_2:

                return 1

            else:

                return 0

        else:

            return 0
telhado['Sistema_FLAT'] = telhado['sensor'].apply(categoriza_SF)

telhado['Sistema_Modular'] = telhado['sensor'].apply(categoriza_SM)

telhado['Sistema_MacDrain'] = telhado['sensor'].apply(categoriza_SMD)
telhado['Verao'] = telhado['Data'].apply(categoriza_Verao)

telhado['Outono'] = telhado['Data'].apply(categoriza_Outono)

telhado['Inverno'] = telhado['Data'].apply(categoriza_Inverno)

telhado['Primavera'] = telhado['Data'].apply(categoriza_Primavera)
telhado
telhado['Dia'] = telhado['hora'].apply(categoriza_Dia)

telhado['Noite'] = telhado['hora'].apply(categoriza_Noite)
clima['Verao'] = clima['Data'].apply(categoriza_Verao)

clima['Outono'] = clima['Data'].apply(categoriza_Outono)

clima['Inverno'] = clima['Data'].apply(categoriza_Inverno)

clima['Primavera'] = clima['Data'].apply(categoriza_Primavera)
clima['Dia'] = clima['hora'].apply(categoriza_Dia)

clima['Noite'] = clima['hora'].apply(categoriza_Noite)
telhado_pivot['Verao'] = telhado_pivot['Data'].apply(categoriza_Verao)

telhado_pivot['Outono'] = telhado_pivot['Data'].apply(categoriza_Outono)

telhado_pivot['Inverno'] = telhado_pivot['Data'].apply(categoriza_Inverno)

telhado_pivot['Primavera'] = telhado_pivot['Data'].apply(categoriza_Primavera)
telhado_pivot['Dia'] = telhado_pivot['hora'].apply(categoriza_Dia)

telhado_pivot['Noite'] = telhado_pivot['hora'].apply(categoriza_Noite)
telhado_pivot_flat['Verao'] = telhado_pivot_flat['Data'].apply(categoriza_Verao)

telhado_pivot_flat['Outono'] = telhado_pivot_flat['Data'].apply(categoriza_Outono)

telhado_pivot_flat['Inverno'] = telhado_pivot_flat['Data'].apply(categoriza_Inverno)

telhado_pivot_flat['Primavera'] = telhado_pivot_flat['Data'].apply(categoriza_Primavera)
telhado_pivot_flat['Dia'] = telhado_pivot_flat['hora'].apply(categoriza_Dia)

telhado_pivot_flat['Noite'] = telhado_pivot_flat['hora'].apply(categoriza_Noite)
telhado_pivot_modular['Verao'] = telhado_pivot_modular['Data'].apply(categoriza_Verao)

telhado_pivot_modular['Outono'] = telhado_pivot_modular['Data'].apply(categoriza_Outono)

telhado_pivot_modular['Inverno'] = telhado_pivot_modular['Data'].apply(categoriza_Inverno)

telhado_pivot_modular['Primavera'] = telhado_pivot_modular['Data'].apply(categoriza_Primavera)
telhado_pivot_modular['Dia'] = telhado_pivot_modular['hora'].apply(categoriza_Dia)

telhado_pivot_modular['Noite'] = telhado_pivot_modular['hora'].apply(categoriza_Noite)
telhado_pivot_macdrain['Verao'] = telhado_pivot_macdrain['Data'].apply(categoriza_Verao)

telhado_pivot_macdrain['Outono'] = telhado_pivot_macdrain['Data'].apply(categoriza_Outono)

telhado_pivot_macdrain['Inverno'] = telhado_pivot_macdrain['Data'].apply(categoriza_Inverno)

telhado_pivot_macdrain['Primavera'] = telhado_pivot_macdrain['Data'].apply(categoriza_Primavera)
telhado_pivot_macdrain['Dia'] = telhado_pivot_macdrain['hora'].apply(categoriza_Dia)

telhado_pivot_macdrain['Noite'] = telhado_pivot_macdrain['hora'].apply(categoriza_Noite)
telhado_clima = clima
lista_excecao = ['Data', 'hora', 'minuto', 'segundo', 'data', 'Hora']



def inicializar_clima_com_excecao(clima, telhado):    

    for t in telhado:    

        if lista_excecao.count(t) == 0:    

            clima[str(t)] = ''

    clima.head(1)
def addItems(clima, index, rowAdd):    

    for i, t in rowAdd.items():

        if i not in lista_excecao:

            clima[i][index] = t

    return row
def addItemsDeitado(climaDeitado, indexRow, row):    

    for i in range(len(lista_excecao_deitada)):

        index = climaDeitado.size

        climaDeitado.loc[index] = 0.0

        climaDeitado['sensor'][index] = lista_excecao_deitada[i]

        climaDeitado['temperatura_sensor'][index] = float(clima[lista_excecao_deitada[i]][indexRow])

        if i == 0 or i == 1:

            climaDeitado['tipo_sensor'][index] = 0

        elif i == 2 or i == 3:

            climaDeitado['tipo_sensor'][index] = 1

        elif i == 4 or i == 5:

            climaDeitado['tipo_sensor'][index] = 2

        for ind, t in row.items(): 

            if ind not in lista_excecao_deitada:

                climaDeitado[ind][index] = t
inicializar_clima_com_excecao(telhado_clima, telhado_pivot)

telhado_clima.head(1)
aux_telhado_pivot = pd.DataFrame()

for index, row in telhado_clima.iterrows():

    aux_telhado_pivot = telhado_pivot.loc[telhado_pivot['Data'] == row['Data']]

    for indexT, rowR in aux_telhado_pivot.iterrows():

    #for indexT, rowR in telhado_dia_verao.iterrows():

        #if (row['Data'] == rowR['Data']) and (row['hora'] == rowR['hora']) and (row['minuto'] == rowR['minuto'] or row['minuto'] < rowR['minuto']):

        if (row['hora'] == rowR['hora']) and (row['minuto'] == rowR['minuto'] or row['minuto'] < rowR['minuto']):

            addItems(telhado_clima, index, rowR)

            break

        #else:

            #addItems(clima_dia_verao, index, rowR)

            #break

telhado_clima.head(5)

#aux_telhado_dia_verao.head(5)
#telhado_clima_perdidos = clima_dia_verao.loc[clima_dia_verao['Delta_1_2'] = '']

#telhado_clima = clima_dia_verao.loc[clima_dia_verao['Delta_1_2'] != '']
# Um dataset por caracteristica 

df_telhado_dia = telhado[(telhado['Dia'] == 1)]

df_telhado_noite = telhado[(telhado['Noite'] == 1)]

df_telhado_FLAT = telhado[(telhado['Sistema_FLAT'] == 1)]

df_telhado_Modular = telhado[(telhado['Sistema_Modular'] == 1)]

df_telhado_MacDrain = telhado[(telhado['Sistema_MacDrain'] == 1)]

df_telhado_Primavera = telhado[(telhado['Primavera'] == 1)]

df_telhado_Verao = telhado[(telhado['Verao'] == 1)]

df_telhado_Outono = telhado[(telhado['Outono'] == 1)]

df_telhado_Inverno = telhado[(telhado['Inverno'] == 1)]
# Datasets Dia

df_telhado_dia_FLAT = df_telhado_dia[(df_telhado_dia['Sistema_FLAT'] == 1)]

df_telhado_dia_Modular = df_telhado_dia[(df_telhado_dia['Sistema_Modular'] == 1)]

df_telhado_dia_MacDrain = df_telhado_dia[(df_telhado_dia['Sistema_MacDrain'] == 1)]

df_telhado_dia_Primavera = df_telhado_dia[(df_telhado_dia['Primavera'] == 1)]

df_telhado_dia_Verao = df_telhado_dia[(df_telhado_dia['Verao'] == 1)]

df_telhado_dia_Outono = df_telhado_dia[(df_telhado_dia['Outono'] == 1)]

df_telhado_dia_Inverno = df_telhado_dia[(df_telhado_dia['Inverno'] == 1)]
# Datasets Dia\FLAT

df_telhado_dia_FLAT_Primavera = df_telhado_dia_FLAT[(df_telhado_dia_FLAT['Primavera'] == 1)]

df_telhado_dia_FLAT_Verao = df_telhado_dia_FLAT[(df_telhado_dia_FLAT['Verao'] == 1)]

df_telhado_dia_FLAT_Outono = df_telhado_dia_FLAT[(df_telhado_dia_FLAT['Outono'] == 1)]

df_telhado_dia_FLAT_Inverno = df_telhado_dia_FLAT[(df_telhado_dia_FLAT['Inverno'] == 1)]
# Datasets Dia\Modular

df_telhado_dia_Modular_Primavera = df_telhado_dia_Modular[(df_telhado_dia_Modular['Primavera'] == 1)]

df_telhado_dia_Modular_Verao = df_telhado_dia_Modular[(df_telhado_dia_Modular['Verao'] == 1)]

df_telhado_dia_Modular_Outono = df_telhado_dia_Modular[(df_telhado_dia_Modular['Outono'] == 1)]

df_telhado_dia_Modular_Inverno = df_telhado_dia_Modular[(df_telhado_dia_Modular['Inverno'] == 1)]
# Datasets Dia\MacDrain

df_telhado_dia_MacDrain_Primavera = df_telhado_dia_MacDrain[(df_telhado_dia_MacDrain['Primavera'] == 1)]

df_telhado_dia_MacDrain_Verao = df_telhado_dia_MacDrain[(df_telhado_dia_MacDrain['Verao'] == 1)]

df_telhado_dia_MacDrain_Outono = df_telhado_dia_MacDrain[(df_telhado_dia_MacDrain['Outono'] == 1)]

df_telhado_dia_MacDrain_Inverno = df_telhado_dia_MacDrain[(df_telhado_dia_MacDrain['Inverno'] == 1)]
# Datasets Noite

df_telhado_noite_FLAT = df_telhado_noite[(df_telhado_noite['Sistema_FLAT'] == 1)]

df_telhado_noite_Modular = df_telhado_noite[(df_telhado_noite['Sistema_Modular'] == 1)]

df_telhado_noite_MacDrain = df_telhado_noite[(df_telhado_noite['Sistema_MacDrain'] == 1)]

df_telhado_noite_Primavera = df_telhado_noite[(df_telhado_noite['Primavera'] == 1)]

df_telhado_noite_Verao = df_telhado_noite[(df_telhado_noite['Verao'] == 1)]

df_telhado_noite_Outono = df_telhado_noite[(df_telhado_noite['Outono'] == 1)]

df_telhado_noite_Inverno = df_telhado_noite[(df_telhado_noite['Inverno'] == 1)]
# Datasets Noite\FLAT

df_telhado_noite_FLAT_Primavera = df_telhado_noite_FLAT[(df_telhado_noite_FLAT['Primavera'] == 1)]

df_telhado_noite_FLAT_Verao = df_telhado_noite_FLAT[(df_telhado_noite_FLAT['Verao'] == 1)]

df_telhado_noite_FLAT_Outono = df_telhado_noite_FLAT[(df_telhado_noite_FLAT['Outono'] == 1)]

df_telhado_noite_FLAT_Inverno = df_telhado_noite_FLAT[(df_telhado_noite_FLAT['Inverno'] == 1)]
# Datasets Noite\Modular

df_telhado_noite_Modular_Primavera = df_telhado_noite_Modular[(df_telhado_noite_Modular['Primavera'] == 1)]

df_telhado_noite_Modular_Verao = df_telhado_noite_Modular[(df_telhado_noite_Modular['Verao'] == 1)]

df_telhado_noite_Modular_Outono = df_telhado_noite_Modular[(df_telhado_noite_Modular['Outono'] == 1)]

df_telhado_noite_Modular_Inverno = df_telhado_noite_Modular[(df_telhado_noite_Modular['Inverno'] == 1)]
# Datasets Noite\MacDrain

df_telhado_noite_MacDrain_Primavera = df_telhado_noite_MacDrain[(df_telhado_noite_MacDrain['Primavera'] == 1)]

df_telhado_noite_MacDrain_Verao = df_telhado_noite_MacDrain[(df_telhado_noite_MacDrain['Verao'] == 1)]

df_telhado_noite_MacDrain_Outono = df_telhado_noite_MacDrain[(df_telhado_noite_MacDrain['Outono'] == 1)]

df_telhado_noite_MacDrain_Inverno = df_telhado_noite_MacDrain[(df_telhado_noite_MacDrain['Inverno'] == 1)]
# Um dataset por caracteristica 

df_clima_dia = clima[(clima['Dia'] == 1)]

df_clima_noite = clima[(clima['Noite'] == 1)]

df_clima_Primavera = clima[(clima['Primavera'] == 1)]

df_clima_Verao = clima[(clima['Verao'] == 1)]

df_clima_Outono = clima[(clima['Outono'] == 1)]

df_clima_Inverno = clima[(clima['Inverno'] == 1)]
# Datasets Dia

df_clima_dia_Primavera = df_clima_dia[(df_clima_dia['Primavera'] == 1)]

df_clima_dia_Verao = df_clima_dia[(df_clima_dia['Verao'] == 1)]

df_clima_dia_Outono = df_clima_dia[(df_clima_dia['Outono'] == 1)]

df_clima_dia_Inverno = df_clima_dia[(df_clima_dia['Inverno'] == 1)]
# Datasets Noite

df_clima_noite_Primavera = df_clima_noite[(df_clima_noite['Primavera'] == 1)]

df_clima_noite_Verao = df_clima_noite[(df_clima_noite['Verao'] == 1)]

df_clima_noite_Outono = df_clima_noite[(df_clima_noite['Outono'] == 1)]

df_clima_noite_Inverno = df_clima_noite[(df_clima_noite['Inverno'] == 1)]
# Um dataset por caracteristica 

df_telhado_pivot_dia = telhado_pivot[(telhado_pivot['Dia'] == 1)]

df_telhado_pivot_noite = telhado_pivot[(telhado_pivot['Noite'] == 1)]

df_telhado_pivot_Primavera = telhado_pivot[(telhado_pivot['Primavera'] == 1)]

df_telhado_pivot_Verao = telhado_pivot[(telhado_pivot['Verao'] == 1)]

df_telhado_pivot_Outono = telhado_pivot[(telhado_pivot['Outono'] == 1)]

df_telhado_pivot_Inverno = telhado_pivot[(telhado_pivot['Inverno'] == 1)]
# Datasets Dia

df_telhado_pivot_dia_Primavera = df_telhado_pivot_dia[(df_telhado_pivot_dia['Primavera'] == 1)]

df_telhado_pivot_dia_Verao = df_telhado_pivot_dia[(df_telhado_pivot_dia['Verao'] == 1)]

df_telhado_pivot_dia_Outono = df_telhado_pivot_dia[(df_telhado_pivot_dia['Outono'] == 1)]

df_telhado_pivot_dia_Inverno = df_telhado_pivot_dia[(df_telhado_pivot_dia['Inverno'] == 1)]
# Datasets Noite

df_telhado_pivot_noite_Primavera = df_telhado_pivot_noite[(df_telhado_pivot_noite['Primavera'] == 1)]

df_telhado_pivot_noite_Verao = df_telhado_pivot_noite[(df_telhado_pivot_noite['Verao'] == 1)]

df_telhado_pivot_noite_Outono = df_telhado_pivot_noite[(df_telhado_pivot_noite['Outono'] == 1)]

df_telhado_pivot_noite_Inverno = df_telhado_pivot_noite[(df_telhado_pivot_noite['Inverno'] == 1)]
telhado.to_csv('telhado_verde.csv')
clima.to_csv('clima.csv')
telhado_pivot.to_csv('telhado_verde_pivot.csv')
telhado_pivot_flat.to_csv('telhado_verde_pivot_flat.csv')
telhado_pivot_modular.to_csv('telhado_verde_pivot_modular.csv')
telhado_pivot_macdrain.to_csv('telhado_verde_pivot_macdrain.csv')
telhado_clima.to_csv('telhado_clima.csv')
# Um dataset por caracteristica 

df_telhado_dia.to_csv('df_telhado_dia.csv')

df_telhado_noite.to_csv('df_telhado_noite.csv')

df_telhado_FLAT.to_csv('df_telhado_FLAT.csv')

df_telhado_Modular.to_csv('df_telhado_Modular.csv')

df_telhado_MacDrain.to_csv('df_telhado_MacDrain.csv')

df_telhado_Primavera.to_csv('df_telhado_Primavera.csv')

df_telhado_Verao.to_csv('df_telhado_Verao.csv')

df_telhado_Outono.to_csv('df_telhado_Outono.csv')

df_telhado_Inverno.to_csv('df_telhado_Inverno.csv')
# Datasets Dia

df_telhado_dia_FLAT.to_csv('df_telhado_dia_FLAT.csv')

df_telhado_dia_Modular.to_csv('df_telhado_dia_Modular.csv')

df_telhado_dia_MacDrain.to_csv('df_telhado_dia_MacDrain.csv')

df_telhado_dia_Primavera.to_csv('df_telhado_dia_Primavera.csv')

df_telhado_dia_Verao.to_csv('df_telhado_dia_Verao.csv')

df_telhado_dia_Outono.to_csv('df_telhado_dia_Outono.csv')

df_telhado_dia_Inverno.to_csv('df_telhado_dia_Inverno.csv')
# Datasets Dia\FLAT

df_telhado_dia_FLAT_Primavera.to_csv('df_telhado_dia_FLAT_Primavera.csv')

df_telhado_dia_FLAT_Verao.to_csv('df_telhado_dia_FLAT_Verao.csv')

df_telhado_dia_FLAT_Outono.to_csv('df_telhado_dia_FLAT_Outono.csv')

df_telhado_dia_FLAT_Inverno.to_csv('df_telhado_dia_FLAT_Inverno.csv')
# Datasets Dia\Modular

df_telhado_dia_Modular_Primavera.to_csv('df_telhado_dia_Modular_Primavera.csv')

df_telhado_dia_Modular_Verao.to_csv('df_telhado_dia_Modular_Verao.csv')

df_telhado_dia_Modular_Outono.to_csv('df_telhado_dia_Modular_Outono.csv')

df_telhado_dia_Modular_Inverno.to_csv('df_telhado_dia_Modular_Inverno.csv')
# Datasets Dia\MacDrain

df_telhado_dia_MacDrain_Primavera.to_csv('df_telhado_dia_MacDrain_Primavera.csv')

df_telhado_dia_MacDrain_Verao.to_csv('df_telhado_dia_MacDrain_Verao.csv')

df_telhado_dia_MacDrain_Outono.to_csv('df_telhado_dia_MacDrain_Outono.csv')

df_telhado_dia_MacDrain_Inverno.to_csv('df_telhado_dia_MacDrain_Inverno.csv')
# Datasets Noite

df_telhado_noite_FLAT.to_csv('df_telhado_noite_FLAT.csv')

df_telhado_noite_Modular.to_csv('df_telhado_noite_Modular.csv')

df_telhado_noite_MacDrain.to_csv('df_telhado_noite_MacDrain.csv')

df_telhado_noite_Primavera.to_csv('df_telhado_noite_Primavera.csv')

df_telhado_noite_Verao.to_csv('df_telhado_noite_Verao.csv')

df_telhado_noite_Outono.to_csv('df_telhado_noite_Outono.csv')

df_telhado_noite_Inverno.to_csv('df_telhado_noite_Inverno.csv')
# Datasets Noite\FLAT

df_telhado_noite_FLAT_Primavera.to_csv('df_telhado_noite_FLAT_Primavera.csv')

df_telhado_noite_FLAT_Verao.to_csv('df_telhado_noite_FLAT_Verao.csv')

df_telhado_noite_FLAT_Outono.to_csv('df_telhado_noite_FLAT_Outono.csv')

df_telhado_noite_FLAT_Inverno.to_csv('df_telhado_noite_FLAT_Inverno.csv')
# Datasets Noite\Modular

df_telhado_noite_Modular_Primavera.to_csv('df_telhado_noite_Modular_Primavera.csv')

df_telhado_noite_Modular_Verao.to_csv('df_telhado_noite_Modular_Verao.csv')

df_telhado_noite_Modular_Outono.to_csv('df_telhado_noite_Modular_Outono.csv')

df_telhado_noite_Modular_Inverno.to_csv('df_telhado_noite_Modular_Inverno.csv')
# Datasets Noite\MacDrain

df_telhado_noite_MacDrain_Primavera.to_csv('df_telhado_noite_MacDrain_Primavera.csv')

df_telhado_noite_MacDrain_Verao.to_csv('df_telhado_noite_MacDrain_Verao.csv')

df_telhado_noite_MacDrain_Outono.to_csv('df_telhado_noite_MacDrain_Outono.csv')

df_telhado_noite_MacDrain_Inverno.to_csv('df_telhado_noite_MacDrain_Inverno.csv')
# Um dataset por caracteristica 

df_clima_dia.to_csv('df_clima_dia.csv')

df_clima_noite.to_csv('df_clima_noite.csv')

df_clima_Primavera.to_csv('df_clima_Primavera.csv')

df_clima_Verao.to_csv('df_clima_Verao.csv')

df_clima_Outono.to_csv('df_clima_Outono.csv')

df_clima_Inverno.to_csv('df_clima_Inverno.csv')
# Datasets Dia

df_clima_dia_Primavera.to_csv('df_clima_dia_Primavera.csv')

df_clima_dia_Verao.to_csv('df_clima_dia_Verao.csv')

df_clima_dia_Outono.to_csv('df_clima_dia_Outono.csv')

df_clima_dia_Inverno.to_csv('df_clima_dia_Inverno.csv')
# Datasets Noite

df_clima_noite_Primavera.to_csv('df_clima_noite_Primavera.csv')

df_clima_noite_Verao.to_csv('df_clima_noite_Verao.csv')

df_clima_noite_Outono.to_csv('df_clima_noite_Outono.csv')

df_clima_noite_Inverno.to_csv('df_clima_noite_Inverno.csv')
# Um dataset por caracteristica 

df_telhado_pivot_dia.to_csv('df_telhado_pivot_dia.csv')

df_telhado_pivot_noite.to_csv('df_telhado_pivot_noite.csv')

df_telhado_pivot_Primavera.to_csv('df_telhado_pivot_Primavera.csv')

df_telhado_pivot_Verao.to_csv('df_telhado_pivot_Verao.csv')

df_telhado_pivot_Outono.to_csv('df_telhado_pivot_Outono.csv')

df_telhado_pivot_Inverno.to_csv('df_telhado_pivot_Inverno.csv')
# Datasets Dia

df_telhado_pivot_dia_Primavera.to_csv('df_telhado_pivot_dia_Primavera.csv')

df_telhado_pivot_dia_Verao.to_csv('df_telhado_pivot_dia_Verao.csv')

df_telhado_pivot_dia_Outono.to_csv('df_telhado_pivot_dia_Outono.csv')

df_telhado_pivot_dia_Inverno.to_csv('df_telhado_pivot_dia_Inverno.csv')
# Datasets Noite

df_telhado_pivot_noite_Primavera.to_csv('df_telhado_pivot_noite_Primavera.csv')

df_telhado_pivot_noite_Verao.to_csv('df_telhado_pivot_noite_Verao.csv')

df_telhado_pivot_noite_Outono.to_csv('df_telhado_pivot_noite_Outono.csv')

df_telhado_pivot_noite_Inverno.to_csv('df_telhado_pivot_noite_Inverno.csv')