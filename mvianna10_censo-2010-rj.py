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



import seaborn as sns # seaborn package

import matplotlib.pyplot as plt # matplotlib library



from scipy import stats
# Carrega dados de pessoas do Rio de Janeiro (RJ) por sexo, cor e faixa de idade

os.chdir('/kaggle/input/2010-brazilian-census-rio-de-janeiro-state/')

P03_UF = pd.read_csv('Pessoa03_RJ.csv', sep=';', decimal = ',', encoding = 'latin_1')

P03_UF.head()
# 

def resumetable(df):

    print(f"Dataset Shape: {df.shape}")

    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])

    summary = summary.reset_index()

    summary['Name'] = summary['index']

    summary = summary[['Name','dtypes']]

    summary['Missing'] = df.isnull().sum().values    

    summary['Uniques'] = df.nunique().values

    summary['First Value'] = df.loc[0].values

    summary['Second Value'] = df.loc[1].values

    summary['Third Value'] = df.loc[2].values



    for name in summary['Name'].value_counts().index:

        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 



    return summary



#Convertendo variáveis numéricas

def converte_num(df, ini, fim):

    for v in range(ini,fim+1):

        var = 'V'+str(v).zfill(3)

        df[var] = pd.to_numeric(df[var], downcast ='integer', errors='coerce')

        

# cores de pessoas 

cor = ['branca','preta','amarela','parda','indigena']



# gera DF com dimensões a partir de df, considerando a faixa de idade (lista), cor e 

# variáveis iniciais para totais masculino e feminino

def gera_dimensoes (df, faixa, cor, ini_m, ini_f, exclusao = []):



    df_dim = pd.DataFrame(columns=['Cod_setor', 'Sexo', 'Idade', 'Cor', 'Pessoas'])



    v = 0

    for f in faixa:

        for c in cor:

            if f not in exclusao:

                varm = 'V'+str(v + ini_m).zfill(3)

                varf = 'V'+str(v + ini_f).zfill(3)

                # masculino

                df_tmp = df[['Cod_setor' , varm]].query(varm + ' > 0')

                df_tmp.rename(columns = {varm: "Pessoas"}, inplace = True)

                df_tmp['Sexo'] = 'M'

                df_tmp['Idade'] = f

                df_tmp['Cor'] = c

                df_tmp = df_tmp[['Cod_setor', 'Sexo', 'Idade', 'Cor', 'Pessoas']]

                df_dim = pd.concat([df_dim, df_tmp], sort=True)

        

                # feminino

                df_tmp = df[['Cod_setor' , varf]].query(varf + ' > 0')

                df_tmp.rename(columns = {varf: "Pessoas"}, inplace = True)

                df_tmp['Sexo'] = 'F'

                df_tmp['Idade'] = f

                df_tmp['Cor'] = c

                df_tmp = df_tmp[['Cod_setor', 'Sexo', 'Idade', 'Cor', 'Pessoas']]

                df_dim = pd.concat([df_dim, df_tmp], sort=True  )      

            v += 1 

            

    return df_dim



# Histogramas de domicilios, moradores e rendimentos

def graf_barras(df):

    fig, axis  = plt.subplots(1,3,figsize=(20,4))

    df.groupby('Sexo')['Pessoas'].sum().plot.bar(x='Sexo', y = 'Pessoas', ax = axis[0])

    df.groupby('Cor')['Pessoas'].sum().plot.bar(x='Cor', y = 'Pessoas', ax = axis[1])

    df.groupby('Idade')['Pessoas'].sum().plot.bar(x='Idade', y = 'Pessoas', ax = axis[2])
resumetable(P03_UF)
#Convertendo variáveis numéricas

converte_num(P03_UF,1,251)

resumetable(P03_UF)
# Cria lista de faixas descritas do dicionário de dados

# a maioria é de 5 em 5, mas existem algumas exceções

faixa = []

for f in range(5,75,5):

    faixa.append(f)

faixa.extend([7,15.1,18])

faixa.remove(65)

faixa.sort()

exclusao = [15.1,18]

P03b_UF = gera_dimensoes(P03_UF, faixa, cor, 87, 167, exclusao)



P03b_UF.shape
graf_barras(P03b_UF)
# Carrega dados do Rio de Janeiro (RJ)

os.chdir('/kaggle/input/2010-brazilian-census-rio-de-janeiro-state/')

P05_UF = pd.read_csv('Pessoa05_RJ.csv', sep=';', decimal = ',', encoding = 'latin_1')

P05_UF.head()
resumetable(P05_UF)
#Convertendo variáveis numéricas

converte_num(P05_UF,1,10)

resumetable(P05_UF)
# Cria lista de faixas descritas do dicionário de dados

faixa = [0]

P05b_UF = gera_dimensoes(P05_UF, faixa, cor, 1, 6)



P05b_UF.shape
graf_barras(P05b_UF)
P00a_UF = pd.concat([P03b_UF, P05b_UF], sort=True)
graf_barras(P00a_UF)
# Carrega dados de alfabetizdos do Rio de Janeiro (RJ)

os.chdir('/kaggle/input/2010-brazilian-census-rio-de-janeiro-state/')

P04_UF = pd.read_csv('Pessoa04_RJ.csv', sep=';', decimal = ',', encoding = 'latin_1')

P04_UF.head()
resumetable(P04_UF)
converte_num(P04_UF,1,155)

resumetable(P04_UF)
# Cria lista de faixas descritas do dicionário de dados

# a maioria é de 5 em 5, mas existem algumas exceções

faixa = []

for f in range(10,75,5):

    faixa.append(f)

faixa.extend([7,15.1,18])

faixa.remove(65)

faixa.sort()

exclusao = [15.1,18]

P04b_UF = gera_dimensoes(P04_UF, faixa, cor, 1, 76, exclusao)



P04b_UF.shape
#altabetizados

graf_barras(P04b_UF)
#todos

graf_barras(P00a_UF)