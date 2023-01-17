# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import sys

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input/"))



print(os.listdir("../input/igm-indice-governancao-municipal"))

print(os.listdir("../input/votos1turnodfcomsecao"))

print(os.listdir("../input/brazil-elections-2018"))

df=pd.DataFrame()

#df=df.append(pd.read_pickle('../input/brazil-elections-2018/bweb_1t_PA_101020182003.csv.gz.pickle',compression='gzip'),

#             ignore_index=True)

        

df.head().T
fields=['NM_MUNICIPIO','DS_CARGO_PERGUNTA','QT_VOTOS','NM_VOTAVEL','SG_ UF']

for i in os.listdir("../input/brazil-elections-2018"):

    try:

        df=df.append(pd.read_pickle('../input/brazil-elections-2018/'+i, compression='gzip')[fields],ignore_index=True)

        df=df[df.DS_CARGO_PERGUNTA=='Presidente']

        print(i, ", SIZE: {:,} bytes".format(sys.getsizeof(df)))

    except Exception as e:

        print(i, "Error loading file", repr(e))



# Any results you write to the current directory are saved as output.
df.NM_MUNICIPIO=df.NM_MUNICIPIO.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
df.shape
#Carrega a tabela de igm

df_igm = pd.read_csv('../input/igm-indice-governancao-municipal/igm_modificado.csv', encoding='latin1', decimal=',')



#Retira apenas os do GO para deixar a tabela mais enxuta

#df_igm_go=df_igm[df_igm.estado=='GO']



#Verifica se tem cidade com o mesmo nome

#df_igm_go.municipio.duplicated().sum()



df_igm.head().T
#Le o arquivo da votação do 1º turno no GO

#df = pd.read_csv('../input/votos1turnodfcomsecao/votos1turnoGO.csv', sep=';', encoding='latin1', decimal=',')

#Comentado para não dar impacto na apuração do pais todo

#Limpa a Colunia NM_MUNICIPIO que tem acento para fazer o join com a tabela igm_modificado

#df.NM_MUNICIPIO=df.NM_MUNICIPIO.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

#Verificando se a quantidade de municipios entre as tabelas é igual

#if  df.NM_MUNICIPIO.nunique()==df_igm_go.municipio.nunique():

#    print('ok')

#else:

#    print('nok')



#df[df['NM_MUNICIPIO'] == 'GOIANIA'].QT_VOTOS.sum()
tabela_final=pd.merge(df, df_igm, left_on=['NM_MUNICIPIO','SG_ UF'], right_on=['municipio','estado'], how='inner')

tabela_final.head().T
variavelASerEstudada='idhm'
df2=tabela_final[tabela_final.DS_CARGO_PERGUNTA=='Presidente'][['QT_VOTOS','NM_VOTAVEL','NM_MUNICIPIO','estado',variavelASerEstudada,'porte']].groupby(['NM_VOTAVEL','NM_MUNICIPIO','estado',variavelASerEstudada,'porte']).sum().sort_values(['NM_MUNICIPIO','estado','QT_VOTOS'], ascending=False)
df2=df2.reset_index()
df2
df2['MAX_VOTOS_POR_MUNICIPIO']=df2.groupby(['NM_MUNICIPIO','estado'])['QT_VOTOS'].transform('max')
df2['rank'] = df2.groupby(['NM_MUNICIPIO','estado'])['QT_VOTOS'].rank(ascending=False)
df2.groupby('NM_VOTAVEL')['rank'].value_counts()
#df2[(df2['NM_VOTAVEL']=='CIRO GOMES') & (df2['rank']==2)]
df3=df2[[variavelASerEstudada,'NM_VOTAVEL','rank','porte', 'QT_VOTOS']]
df4=df3.groupby(['porte','NM_VOTAVEL',])[['rank','QT_VOTOS']].sum().sort_values(['porte','rank'],ascending='False')

df5=df3.groupby(['porte'])[['QT_VOTOS']].sum().sort_values(['porte'],ascending='False').reset_index()

df6=pd.merge(df4.reset_index(), df5, left_on='porte', right_on='porte', how='inner')

df6
df6['media']=df6.QT_VOTOS_x/df6.QT_VOTOS_y*100

df6.drop(columns=['QT_VOTOS_x','QT_VOTOS_y','rank']).set_index(['porte','NM_VOTAVEL']).unstack().plot.bar(figsize=(12,12))

df7=df6.drop(columns=['QT_VOTOS_x','QT_VOTOS_y','rank']).set_index(['porte','NM_VOTAVEL']).unstack()

df7.columns=['ALVARO DIAS', 'Branco', 'CABO DACIOLO', 'CIRO GOMES', 'EYMAEL', 'FERNANDO HADDAD', 'GERALDO ALCKMIN', 'GUILHERME BOULOS', 'HENRIQUE MEIRELLES', 'JAIR BOLSONARO', 'JOÃO AMOÊDO', 'JOÃO GOULART FILHO', 'MARINA SILVA', 'Nulo', 'VERA', 'Anulada e apurada em separado']

df7[['JAIR BOLSONARO','FERNANDO HADDAD','CIRO GOMES','Branco','Nulo']].plot.bar(figsize=(8,8),rot=0)
#df4[['QT_VOTOS']].unstack().plot.bar(figsize=(12,12), stacked=True)

df4[['QT_VOTOS']].unstack().plot.bar(figsize=(12,12))





df3.groupby(['porte','NM_VOTAVEL'])[['rank',variavelASerEstudada,'QT_VOTOS']].mean().sort_values(['porte','rank'],ascending='False')  #df2.groupby(['NM_MUNICIPIO'])['QT_VOTOS'].transform('max')
tabela_final.columns
tabela_final[variavelASerEstudada] = tabela_final[variavelASerEstudada].astype(float)
tabelaVariavelASerEstudada = tabela_final[tabela_final['DS_CARGO_PERGUNTA']=='Presidente'].groupby(['municipio','NM_VOTAVEL']).agg({'QT_VOTOS':'sum', variavelASerEstudada:'mean'})
tabelaVariavelASerEstudada[variavelASerEstudada] = pd.qcut(tabelaVariavelASerEstudada[variavelASerEstudada], 20)
tabelaVariavelASerEstudada2 = (tabelaVariavelASerEstudada.groupby([variavelASerEstudada,'NM_VOTAVEL'])['QT_VOTOS'].sum()/tabelaVariavelASerEstudada.groupby([variavelASerEstudada])['QT_VOTOS'].sum())
tabelaVariavelASerEstudada[variavelASerEstudada].value_counts()

tabelaVariavelASerEstudada2.unstack()
tabelaVariavelASerEstudada2.unstack()[['JAIR BOLSONARO','FERNANDO HADDAD','CIRO GOMES','Branco','Nulo']].plot.line(figsize=(12,12))