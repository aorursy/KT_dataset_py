# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

import datetime as dt # Funções de data e hora

import seaborn as sns

# Any results you write to the current directory are saved as output.

%matplotlib inline
ultimo='Apurao mensal de indicadores_BD Deslig e Pert_janela DEZ17 a NOV18.xlsx'

analisados=0

aceitos=0

rejeitados=0

ignorados=0

todos = os.listdir("../input/")

for datas in range(2):

    for x in todos:

        if (datas==0 and x==ultimo) or (datas!=0  and x!=ultimo):

            analisados+=1

        if (datas==0 and x==ultimo) or (datas!=0 and x!=ultimo):

            excel=pd.ExcelFile("../input/"+x)

            cont = 0

            #planilha2 = excel.sheet_names

            #print(planilha2)

            #if planilha2 in ['Desligamentos', 'Desligamentos_v2']:

            #    print('Achei')

            for planilha in excel.sheet_names:

                if planilha == 'Desligamentos' or planilha == 'Desligamentos_v2':

                    cont+=1

                    hd = 0

                    while hd < 1000:

                        if planilha == 'Desligamentos':

                            plan1 = pd.read_excel("../input/"+x, sheet_name="Desligamentos", header=hd)

                        else:

                            plan1 = pd.read_excel("../input/"+x, sheet_name="Desligamentos_v2", header=hd)

                        if plan1.columns[0]=='CodPert':

                            if datas==0:

                                df1=plan1

                                for coluna in df1.columns:

                                    if coluna in ['Unnamed: 3', 'Disponibilização pelo Agente', 'DataDevEquip']:

                                        df1.rename(columns={coluna:'DataEntregaAge'}, inplace=True)

                                    if coluna in ['Unnamed: 4', 'Retorno à Operaçao', 'DataRelig']:

                                        df1.rename(columns={coluna:'DataReligEfetivo'}, inplace=True)

                                    if coluna in ['Data de emissão da planilha', 'Referência', 'ClassCausa', 'referência', 'Causas']:

                                        df1=df1.drop(coluna, axis=1)

                            else:

                                if x=='Apurao mensal de indicadores_BD Deslig e Pert_janela ABR17 a MAR18.xlsx' or x=='Apurao mensal de indicadores_BD Deslig e Pert_janela MAI17 a ABR18.xlsx' or x=='Apurao mensal de indicadores_BD Deslig e Pert_janela SET15 a AGO16.xlsx' or x=='Apurao mensal de indicadores_BD Deslig e Pert_janela FEV16 a JAN17.xlsx':

                                    plan1=plan1.head(600)

                                if plan1.DataDeslForc.min()<dt.datetime(2014,8,1):

                                    plan1=plan1[plan1.DataDeslForc<dt.datetime(2014,8,1)]

                                else:

                                    plan1=plan1[plan1.DataDeslForc<dt.datetime((plan1.DataDeslForc.min()+dt.timedelta(days=32)).year,(plan1.DataDeslForc.min()+dt.timedelta(days=32)).month,1)]

                                for coluna in plan1.columns:

                                    if coluna in ['Unnamed: 3', 'Disponibilização pelo Agente', 'DataDevEquip']:

                                        plan1.rename(columns={coluna:'DataEntregaAge'}, inplace=True)

                                    if coluna in ['Unnamed: 4', 'Retorno à Operaçao', 'DataRelig']:

                                        plan1.rename(columns={coluna:'DataReligEfetivo'}, inplace=True)

                                    if coluna in ['Data de emissão da planilha', 'Referência', 'ClassCausa', 'referência', 'Causas']:

                                        plan1=plan1.drop(coluna, axis=1)

                                df1=df1.append(plan1, sort=False)

                            print('Arquivo ' + str(analisados) + ': ' + x +': Ok na linha ' + str(hd) + ' - iteração: ' + str(datas))

                            hd=999

                            aceitos+=1

                        hd+=1

            if cont==0:

                print('Arquivo ' + str(analisados)  + ': ' + x + ' não tem planilha Desligamentos')

                rejeitados+=1

        else:

            if (datas==0 and x==ultimo) or (datas!=0  and x!=ultimo):

                print('Arquivo ' + str(analisados) + ': ' + x +': Ignorado')

                ignorados+=1

df1=df1.sort_values('DataDeslForc')

print('Total de arquivos analisados: ' + str(analisados))

print('Total de arquivos ignorados: ' + str(ignorados))

print('Total de arquivos aceitos: ' + str(aceitos))

print('Total de arquivos não aceitos: ' + str(rejeitados))
type(excel)
type(df1)
df1.count()
df1.isnull().sum()
df1['DescNaturezaCausa'].value_counts().plot.bar(figsize=(14,10))
df1['DescNaturezaEletrica'].value_counts().plot.barh(figsize=(14,10), title=('Natureza Elétrica dos Desligamentos'))
df1['DescLocal'].value_counts().plot.barh(figsize=(14,30), title=('Local dos Desligamentos'))
df1['DescCausa'].value_counts().plot.barh(figsize=(14,30), title=('Causas dos Desligamentos'))
explode = (0.4, 0.4, 0.4, 0, 0 )

df1['DescOrigem'].value_counts().sort_values().plot.pie(explode =explode, autopct='%1.1f%%', shadow=True, startangle=140, figsize=(14,10))
origem = df1

origem = origem[['CodOrigem', 'DescOrigem']]

origem = origem.drop_duplicates('CodOrigem')

type(origem)
origem = origem.to_dict

type(origem)
origem
local = df1

local = local[['CodLocal', 'DescLocal']]

local = local.drop_duplicates('CodLocal')

local.to_dict
causa = df1

causa = causa[['CodCausa', 'DescCausa']]

causa = causa.drop_duplicates('CodCausa')

causa.to_dict
naturezacausa = df1

naturezacausa = naturezacausa[['CodNaturezaCausa', 'DescNaturezaCausa']]

naturezacausa = naturezacausa.drop_duplicates('CodNaturezaCausa')

naturezacausa.to_dict
naturezaeletrica = df1

naturezaeletrica = naturezaeletrica[['CodNaturezaEletrica', 'DescNaturezaEletrica']]

naturezaeletrica = naturezaeletrica.drop_duplicates('CodNaturezaEletrica')

naturezaeletrica.to_dict
for coluna in df1.columns:

    if coluna in ['DescOrigem', 'DescLocal', 'DescCausa', 'DescNaturezaCausa', 'DescNaturezaEletrica']:

        df1=df1.drop(coluna, axis=1)
df1=df1[df1['CodOrigem'].isin (['I' , 'S'])]
df1['UF']=df1['EqpId'].str[0:2]
df1['UF'].value_counts().plot.barh(figsize=(14,30), title=('Desligamentos / UF'))
df1['anoDeslig']=df1['DataDeslForc'].dt.year
df1['mesDeslig']=df1['DataDeslForc'].dt.month
df1['anoDeslig'].value_counts().plot.bar(figsize=(14,10), title=('Desligamentos / ano'))
df1['mesDeslig'].value_counts().plot.bar(figsize=(14,10), title=('Desligamentos / mês'))
sns.factorplot(x='mesDeslig', data=df1, col='anoDeslig', kind='count')