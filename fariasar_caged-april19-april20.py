# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_19 = pd.read_csv('../input/caged-1904-2004/CAGEDEST_042019_utf.txt',sep=';',low_memory=False)

df_20 = pd.read_csv('../input/caged-1904-2004/CAGEDMOV202004.txt',sep=';',low_memory=False)
df_19.shape,df_20.shape
# PADRONIZANDO AS COLUNAS

df_19.columns = df_19.columns.str.replace(u'á','a')

df_19.columns = df_19.columns.str.replace(u'â','a')

df_19.columns = df_19.columns.str.replace(u'ã','a')

df_19.columns = df_19.columns.str.replace(u'à','a')

df_19.columns = df_19.columns.str.replace(u'é','e')

df_19.columns = df_19.columns.str.replace(u'è','e')

df_19.columns = df_19.columns.str.replace(u'ê','e')

df_19.columns = df_19.columns.str.replace(u'í','i')

df_19.columns = df_19.columns.str.replace(u'ì','i')

df_19.columns = df_19.columns.str.replace(u'î','i')

df_19.columns = df_19.columns.str.replace(u'ó','o')

df_19.columns = df_19.columns.str.replace(u'ò','o')

df_19.columns = df_19.columns.str.replace(u'õ','o')

df_19.columns = df_19.columns.str.replace(u'ô','o')

df_19.columns = df_19.columns.str.replace(u'ú','u')

df_19.columns = df_19.columns.str.replace(u'ù','u')

df_19.columns = df_19.columns.str.replace(u'û','u')

df_19.columns = df_19.columns.str.replace(u'ç','c')

df_19.columns = df_19.columns.str.replace(u' ','')



df_20.columns = df_20.columns.str.replace(u'á','a')

df_20.columns = df_20.columns.str.replace(u'â','a')

df_20.columns = df_20.columns.str.replace(u'ã','a')

df_20.columns = df_20.columns.str.replace(u'à','a')

df_20.columns = df_20.columns.str.replace(u'é','e')

df_20.columns = df_20.columns.str.replace(u'è','e')

df_20.columns = df_20.columns.str.replace(u'ê','e')

df_20.columns = df_20.columns.str.replace(u'í','i')

df_20.columns = df_20.columns.str.replace(u'ì','i')

df_20.columns = df_20.columns.str.replace(u'î','i')

df_20.columns = df_20.columns.str.replace(u'ó','o')

df_20.columns = df_20.columns.str.replace(u'ò','o')

df_20.columns = df_20.columns.str.replace(u'õ','o')

df_20.columns = df_20.columns.str.replace(u'ô','o')

df_20.columns = df_20.columns.str.replace(u'ú','u')

df_20.columns = df_20.columns.str.replace(u'ù','u')

df_20.columns = df_20.columns.str.replace(u'û','u')

df_20.columns = df_20.columns.str.replace(u'ç','c')

df_20.columns = df_20.columns.str.replace(u' ','')



df_19.columns = [x.lower() for x in df_19.columns]

df_20.columns = [x.lower() for x in df_20.columns]
for i in df_19.columns:

    if i.startswith('reg'):

        print('Apagando a variável ',i)

        df_19 = df_19.drop([i],axis=1)



for i in df_19.columns:

    if i.startswith('bairro'):

        print('Apagando a variável ',i)

        df_19 = df_19.drop([i],axis=1)
for i in ['admitidos/desligados','anodeclarado','cnae1.0classe','distritossp','ibgesubsetor','mesorregiao','microrregiao','indportadordefic','sub-regiaosenaipr','tempoemprego','tipoestab']:

  print('Apagando a variável ',i)

  df_19 = df_19.drop([i],axis=1)
for i in ['categoria','fonte','tipoempregador','tipoestabelecimento']:

  print('Apagando a variável ',i)

  df_20 = df_20.drop([i],axis=1)
df_19.columns = ['competencia','municipio','cbo2002ocupacao','classe','subclasse','tamestabjan','graudeinstrucao','horascontratuais','idade','indicadoraprendiz','racacor','salario','saldomovimentacao','sexo',

                 'tipodedeficiencia','tipomovimentacao','uf','indtrabparcial','indtrabintermitente']
df_19['tamestabjan'] = df_19['tamestabjan'].apply(lambda x: x+1)
df_19['racacor'].value_counts()
df_19['racacor'] = df_19['racacor'].apply(lambda x: 3 if x == 8  else (

                                                    5 if x == 1  else (

                                                    1 if x == 2  else (

                                                    2 if x == 4  else (

                                                    4 if x == 6  else (

                                                    6 if x == -1 else 9))))))
df_19['racacor'].value_counts()
df_19['sexo'].value_counts()
df_19['sexo'] = df_19['sexo'].apply(lambda x: 1 if x == 1  else (

                                              3 if x == 2  else 9))
df_19['sexo'].value_counts()
df_19['tipomovimentacao'].value_counts()
df_19['tipomovimentacao'] = df_19['tipomovimentacao'].apply(lambda x: 35 if x == 10 else (

                                                                      10 if x == 1  else (

                                                                      20 if x == 2  else (

                                                                      31 if x == 4  else (

                                                                      32 if x == 5  else (

                                                                      40 if x == 6  else (

                                                                      45 if x == 11 else (

                                                                      50 if x == 7  else (

                                                                      60 if x == 8  else (

                                                                      70 if x == 3  else (

                                                                      80 if x == 9  else (

                                                                      25 if x == 25 else (

                                                                      43 if x == 43 else (

                                                                      90 if x == 90 else 98))))))))))))))
df_19['tipomovimentacao'].value_counts()
df_19.info()
df_20.info()
df_19.salario.head()
df_19['salario'] = df_19['salario'].str.replace(u',','.')

df_19['salario'] = df_19['salario'].astype('float64')
for i in df_20.idade:

  if i.is_integer():

    None

  else:

    print(i)
df_20 = df_20.dropna(subset=['idade'])

df_20['idade'] = df_20['idade'].astype('int64')
df_all = df_20.append(df_19)

df_all.shape,df_19.shape,df_20.shape
df_all.head()
v_cols = df_all.columns

n_rows = df_all.shape[0]

n_cols = df_all.shape[1]

print('O Dataset possui' , n_rows, 'linhas e' , n_cols , 'colunas.')
if df_all.isnull().values.any() == True:

    print ("{:<20} {:<10} {:<5}".format('Variável','Nulos','Percentual'))

    for i in df_all.columns:

        if df_all[i].isna().sum() > 0:

            print("{:<20} {:<10} {:<5}".format(i,df_all[i].isna().sum(),(df_all[i].isnull().sum()/n_rows).round(4)*100))

else:

  print('O Dataset não possui valores nulos.')
df_all['regiao'] = df_all['uf'].apply(lambda x: 1 if ((x > 10) & (x < 20)) else (

                                                2 if ((x > 20) & (x < 30)) else (

                                                3 if ((x > 30) & (x < 40)) else (

                                                4 if ((x > 40) & (x < 50)) else (

											    5 if ((x > 49) & (x < 55)) else 6)))))
df_all['classe'] = df_all['subclasse'].apply(lambda x: int(x/100))
df_all['secao'] = df_all['classe'].apply(lambda x: 'A' if ((x > 1000) & (x < 4000)) else (

                                              'B' if ((x > 5000) &  (x < 10000)) else (

                                              'C' if ((x > 10000) & (x < 34000)) else (

                                              'D' if ((x > 35000) & (x < 36000)) else (

                                              'E' if ((x > 36000) & (x < 40000)) else (

                                              'F' if ((x > 41000) & (x < 44000)) else (

                                              'G' if ((x > 45000) & (x < 48000)) else (

                                              'H' if ((x > 49000) & (x < 54000)) else (

                                              'I' if ((x > 55000) & (x < 57000)) else (

                                              'J' if ((x > 58000) & (x < 64000)) else (

                                              'K' if ((x > 64000) & (x < 67000)) else (

                                              'L' if ((x > 68000) & (x < 69000)) else (

                                              'M' if ((x > 69000) & (x < 76000)) else (

                                              'N' if ((x > 77000) & (x < 83000)) else (

                                              'O' if ((x > 84000) & (x < 85000)) else (

                                              'P' if ((x > 85000) & (x < 86000)) else (

                                              'Q' if ((x > 86000) & (x < 89000)) else (

                                              'R' if ((x > 90000) & (x < 94000)) else (

                                              'S' if ((x > 94000) & (x < 97000)) else (

                                              'T' if ((x > 97000) & (x < 98000)) else (

                                              'U' if (x > 99000) else 'Z')))))))))))))))))))))
for col in df_all.columns:

    if (len(df_all[col].unique()) < 3):

        print(df_all[col].value_counts())

        print(col)
df_all.info()
df_all.regiao.value_counts()
df_all[(df_all['regiao']==6)].head()
df_all = df_all[(df_all['regiao'] !=6 )]
print("{:<20} {:<10} {:<10} {:<15}".format('VARIÁVEL','MAX','COUNT','PERCENTUAL'))

for i in df_all.columns:

    x = df_all[i].max()

    if x in (9,99,999,9999,99999,999999,9999999):

        print("{:<20} {:<10} {:<10} {:<15}".format(i,df_all[i].max(),df_all.sexo[(df_all[i] == df_all[i].max())].count(),round(df_all.sexo[(df_all[i] == df_all[i].max())].count()/n_rows,4)*100))
df_all = df_all[(df_all['cbo2002ocupacao']   != 999999 )]

df_all = df_all[(df_all['horascontratuais']  != 99 )]

df_all = df_all[(df_all['tipodedeficiencia'] != 9 )]

df_all = df_all[(df_all['indtrabparcial']    != 9 )]

df_all = df_all[(df_all['tamestabjan']       != 99 )]

df_all.loc[((df_all['racacor'] == 9) ), 'racacor'] = 6
# CRIANDO TIPO FAIXAS SALARIAIS

'''

0 ->  0 - 1/2   salário mínimo

1 ->  1/2 - 1   salário mínimo

2 ->  1 - 2     salários mínimos

3 ->  2 - 3     salários mínimos

4 ->  3 - 5     salários mínimos

5 ->  5 - 10    salários mínimos

6 ->  > 10

'''

df_all['faixasalarial'] = df_all['salario'].apply(lambda x: '1'  if x <= 1039  else (

                                                            '2'  if ((x > 1039) & (x <= 2078))  else (

                                                            '3'  if ((x > 2078) & (x <= 3117))  else (

                                                            '4'  if x > 3117  else None))))
# CRIANDO TIPO PORTE DA EMPRESA

'''

Micro               - Até 19 empregados

Pequeno Porte (EPP) - 20 a 99 empregados

Médio porte         - 100 a 499 empregados

Grandes empresas    - 500 ou mais empregados



fonte: https://m.sebrae.com.br/Sebrae/Portal%20Sebrae/UFs/SP/Pesquisas/MPE_conceito_empregados.pdf

'''

df_all['porte'] = df_all['tamestabjan'].apply(lambda x: 'Micro'   if x <= 4 else (

                                                        'Pequena' if ((x > 4)  & (x <= 6))  else (

                                                        'Media'   if ((x > 6) & (x <= 8))  else (

                                                        'Grande'  if x > 8  else None))))
# CRIANDO TIPO SETORES ECONOMICOS

'''

https://www2.ufjf.br/poseconomia//files/2012/08/Dissertacao-Marc%c3%adlio-Zanelli-Pereira.pdf

'''

df_all['setores'] = df_all['secao'].apply(lambda x: 'Primario'   if x in ['A','B'] else (

                                                    'Secundario' if x in ['C','D','E','F'] else 'Terciario'))
# CRIANDO TIPO CARGA HORÁRIA

'''

0 ->  >= 40     TEMPO INTEGRAL

1 ->  20 - < 40 MEIO PERÍODO

2 ->  < 20      IRREGULAR

'''

df_all['cargahoraria'] = df_all['horascontratuais'].apply(lambda x: 'Tempo Integral' if x >= 40 else (#'Tempo Parcial')

                                                                    'Meio Período'   if ((x < 40)  & (x >= 20))  else (

                                                                    'Irregular'      if x < 20  else None)))
mapping_dictionary = {"uf":{

11:"RO",

12:"AC",

13:"AM",

14:"RR",

15:"PA",

16:"AP",

17:"TO",

21:"MA",

22:"PI",

23:"CE",

24:"RN",

25:"PB",

26:"PE",

27:"AL",

28:"SE",

29:"BA",

31:"MG",

32:"ES",

33:"RJ",

35:"SP",

41:"PR",

42:"SC",

43:"RS",

50:"MS",

51:"MT",

52:"DF",

53:"DF"

}}

df_all['uf_2'] = df_all['uf']

df_all = df_all.replace(mapping_dictionary)
mapping_dictionary = {"sexo":{

1:"Homem",

2:"Mulher",

3:"Mulher",

9:"N Informado"

}}

df_all['sexo_2'] = df_all['sexo']

df_all = df_all.replace(mapping_dictionary)
mapping_dictionary = {"regiao":{

1:"Norte",

2:"Nordeste",

3:"Sudeste",

4:"Sul",

5:"Centro-Oeste",

}}

df_all = df_all.replace(mapping_dictionary)
def plotPerColumnDistribution(df1, nGraphShown, nGraphPerRow):

    nunique = df1.nunique()

    nRow, nCol = df1.shape

    columnNames = list(df1)

    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow

    plt.figure(num = None, figsize = (15 * nGraphPerRow, 8 * nGraphRow), dpi = 60, facecolor = 'w', edgecolor = 'k')

    for i in range(min(nCol, nGraphShown)):

        plt.subplot(nGraphRow, nGraphPerRow, i + 1)

        columnDf = df1.iloc[:, i]

        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):

            valueCounts = columnDf.value_counts()

            valueCounts.plot.bar()

        else:

            columnDf.hist()

        plt.ylabel('counts')

        plt.xticks(rotation = 45, fontsize=20)

        plt.title(f'{columnNames[i]}', fontsize=30)

    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)

    sns.set()  

    plt.show()





plotPerColumnDistribution(df_all, 25, 4)
df_cbo19 = df_all[(df_all['competencia'] == 201904)].groupby(['cbo2002ocupacao']).saldomovimentacao.sum().nlargest(5).reset_index()

df_cbo19.columns = ['cbo2002ocupacao','saldo_2019']

df_tmp = df_all[((df_all['competencia'] == 202004) & (df_all['cbo2002ocupacao'].isin(df_cbo19['cbo2002ocupacao'])))].groupby(['cbo2002ocupacao']).saldomovimentacao.sum().reset_index()

test = pd.merge(df_cbo19,df_tmp,on='cbo2002ocupacao')

df_cbo19[['cbo2002ocupacao','saldo_2019','saldo_2020']] = test

df_cbo19
labels = df_cbo19.cbo2002ocupacao

x_2019 = df_cbo19.saldo_2019

y_2020 = df_cbo19.saldo_2020



x = np.arange(len(labels))  # the label locations

width = 0.25  # the width of the bars



fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, x_2019, width, label='2019')

rects2 = ax.bar(x + width/2, y_2020, width, label='2020')



ax.set_ylabel('Saldo')

ax.set_title('Comparativo Saldo TOP 5 de 2019 x 2020')

ax.set_xticks(x)

ax.set_xticklabels(labels,rotation = 45)

ax.legend()



def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(10, -5),

                    textcoords="offset points",

                    ha='left', va='bottom',fontsize=9)



autolabel(rects1)

autolabel(rects2)

fig.tight_layout()



plt.show()
df_cbo20 = df_all[(df_all['competencia'] == 202004)].groupby(['cbo2002ocupacao']).saldomovimentacao.sum().nlargest(5).reset_index()

df_cbo20.columns = ['cbo2002ocupacao','saldo_2020']

df_tmp = df_all[((df_all['competencia'] == 201904) & (df_all['cbo2002ocupacao'].isin(df_cbo20['cbo2002ocupacao'])))].groupby(['cbo2002ocupacao']).saldomovimentacao.sum().reset_index()

test = pd.merge(df_cbo20,df_tmp,on='cbo2002ocupacao')

df_cbo20[['cbo2002ocupacao','saldo_2020','saldo_2019']] = test

df_cbo20
labels = df_cbo20.cbo2002ocupacao

x_2019 = df_cbo20.saldo_2019

y_2020 = df_cbo20.saldo_2020



x = np.arange(len(labels))  # the label locations

width = 0.25  # the width of the bars



fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, x_2019, width, label='2019')

rects2 = ax.bar(x + width/2, y_2020, width, label='2020')



ax.set_ylabel('Saldo')

ax.set_title('Comparativo Saldo TOP 5 de 2020 x 2019')

ax.set_xticks(x)

ax.set_xticklabels(labels,rotation = 45)

ax.legend()



def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(10, -5),

                    textcoords="offset points",

                    ha='left', va='bottom',fontsize=9)



autolabel(rects1)

autolabel(rects2)

fig.tight_layout()



plt.show()
cbo_all = df_all[(df_all['competencia'] == 202004)].groupby(['cbo2002ocupacao']).saldomovimentacao.sum().reset_index()

cbo_all.columns = ['cbo2002ocupacao','saldo_2020']

df_tmp = df_all[((df_all['competencia'] == 201904) & (df_all['cbo2002ocupacao'].isin(cbo_all['cbo2002ocupacao'])))].groupby(['cbo2002ocupacao']).saldomovimentacao.sum().reset_index()

test = pd.merge(cbo_all,df_tmp,on='cbo2002ocupacao')

cbo_all[['cbo2002ocupacao','saldo_2020','saldo_2019']] = test

cbo_all = cbo_all.dropna()

cbo_all['saldo'] = cbo_all.saldo_2020 - cbo_all.saldo_2019
plt.bar(range(3),(cbo_all.saldo[(cbo_all['saldo'] < 0)].count(),cbo_all.saldo[(cbo_all['saldo'] == 0)].count(),cbo_all.saldo[(cbo_all['saldo'] > 0)].count()))

plt.xticks(range(3), ('Queda','Estabilidade','Crescimento'), rotation='45')

plt.title('Saldo Geral de 2020-2019')

plt.show()
df_uf19 = df_all[(df_all['competencia'] == 201904)].groupby(['uf']).saldomovimentacao.sum().nlargest(5).reset_index()

df_uf19.columns = ['uf','saldo_2019']

df_tmp = df_all[((df_all['competencia'] == 202004) & (df_all['uf'].isin(df_uf19['uf'])))].groupby(['uf']).saldomovimentacao.sum().reset_index()

test = pd.merge(df_uf19,df_tmp,on='uf')

df_uf19[['uf','saldo_2019','saldo_2020']] = test

df_uf19
labels = df_uf19.uf

x_2019 = df_uf19.saldo_2019

y_2020 = df_uf19.saldo_2020



x = np.arange(len(labels))  # the label locations

width = 0.25  # the width of the bars



fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, x_2019, width, label='2019')

rects2 = ax.bar(x + width/2, y_2020, width, label='2020')



ax.set_ylabel('Saldo')

ax.set_title('Comparativo Saldo UF TOP 5 de 2019 x 2020')

ax.set_xticks(x)

ax.set_xticklabels(labels,rotation = 45)

ax.legend()



def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(10, -5),

                    textcoords="offset points",

                    ha='left', va='bottom',fontsize=9)



autolabel(rects1)

autolabel(rects2)

fig.tight_layout()



plt.show()
uf_all = df_all[(df_all['competencia'] == 201904)].groupby(['uf']).saldomovimentacao.sum().reset_index()

uf_all.columns = ['uf','saldo_2019']

df_tmp = df_all[((df_all['competencia'] == 202004) & (df_all['uf'].isin(uf_all['uf'])))].groupby(['uf']).saldomovimentacao.sum().reset_index()

test = pd.merge(uf_all,df_tmp,on='uf')

uf_all[['uf','saldo_2019','saldo_2020']] = test

uf_all = uf_all.dropna()

uf_all['saldo'] = uf_all.saldo_2020 - uf_all.saldo_2019
x = range(len(uf_all.uf))

plt.subplot(2,1,1)

x_2019 = plt.bar(x, uf_all.saldo_2019, label='2019')

plt.subplot(2,1,2)

x_2020 = plt.bar(x, uf_all.saldo_2020,color='gray', label='2020')

plt.suptitle('SALDO 2019 X SALDO 2020 - UF')

plt.xticks(x, uf_all.uf, rotation='vertical')



plt.show()
df_secao19 = df_all[(df_all['competencia'] == 201904)].groupby(['secao']).saldomovimentacao.sum().nlargest(5).reset_index()

df_secao19.columns = ['secao','saldo_2019']

df_tmp = df_all[((df_all['competencia'] == 202004) & (df_all['secao'].isin(df_secao19['secao'])))].groupby(['secao']).saldomovimentacao.sum().reset_index()

test = pd.merge(df_secao19,df_tmp,on='secao')

df_secao19[['secao','saldo_2019','saldo_2020']] = test

df_secao19
labels = df_secao19.secao

x_2019 = df_secao19.saldo_2019

y_2020 = df_secao19.saldo_2020



x = np.arange(len(labels))  # the label locations

width = 0.25  # the width of the bars



fig, ax = plt.subplots()

rects1 = ax.bar(x - width/2, x_2019, width, label='2019')

rects2 = ax.bar(x + width/2, y_2020, width, label='2020')



ax.set_ylabel('Saldo')

ax.set_title('Comparativo Saldo SEÇÃO TOP 5 de 2019 x 2020')

ax.set_xticks(x)

ax.set_xticklabels(labels,rotation = 45)

ax.legend()



def autolabel(rects):

    for rect in rects:

        height = rect.get_height()

        ax.annotate('{}'.format(height),

                    xy=(rect.get_x() + rect.get_width() / 2, height),

                    xytext=(10, -5),

                    textcoords="offset points",

                    ha='left', va='bottom',fontsize=9)



autolabel(rects1)

autolabel(rects2)

fig.tight_layout()



plt.show()
secao_all = df_all[(df_all['competencia'] == 201904)].groupby(['secao']).saldomovimentacao.sum().reset_index()

secao_all.columns = ['secao','saldo_2019']

df_tmp = df_all[((df_all['competencia'] == 202004) & (df_all['secao'].isin(secao_all['secao'])))].groupby(['secao']).saldomovimentacao.sum().reset_index()

test = pd.merge(secao_all,df_tmp,on='secao')

secao_all[['secao','saldo_2019','saldo_2020']] = test

secao_all = secao_all.dropna()

secao_all['saldo'] = secao_all.saldo_2020 - secao_all.saldo_2019
plt.bar(range(3),(secao_all.saldo[(secao_all['saldo'] < 0)].count(),secao_all.saldo[(secao_all['saldo'] == 0)].count(),secao_all.saldo[(secao_all['saldo'] > 0)].count()))

plt.xticks(range(3), ('Queda','Estabilidade','Crescimento'), rotation='45')

plt.title('Saldo Geral de 2020-2019')

plt.show()
x = range(len(secao_all.secao))

plt.subplot(2,1,1)

x_2019 = plt.bar(x, secao_all.saldo_2019, label='2019')

plt.subplot(2,1,2)

x_2020 = plt.bar(x, secao_all.saldo_2020,color='gray', label='2020')

plt.suptitle('SALDO 2019 X SALDO 2020 - SEÇÃO')

plt.xticks(x, secao_all.secao, rotation='vertical')



plt.show()
feats = [c for c in df_all.columns if c not in ['saldomovimentacao','secao','competencia','regiao','cargahoraria','setores','porte','tipomovimentacao','indtrabintermitente','indtrabparcial','tipodedeficiencia','uf','sexo']]
from sklearn.model_selection import train_test_split

from sklearn.metrics import plot_confusion_matrix

from sklearn.ensemble import RandomForestClassifier

import seaborn as sns



y = df_all['saldomovimentacao'][(df_all['competencia']==202004)]

x_train,x_test,y_train,y_test = train_test_split(df_all[feats][(df_all['competencia']==202004)],y,test_size=0.25,random_state=42)
rf = RandomForestClassifier(max_depth=None, random_state=42, n_jobs=4, n_estimators=400,

                            min_impurity_decrease=1e-3, min_samples_leaf=2,

                            verbose=0, class_weight='balanced')
rf.fit(x_train[feats], y_train)
sns.set_style("dark")

plot_confusion_matrix(rf,x_test,y_test,display_labels=['1','-1'],cmap=plt.cm.Blues)
importances = list(rf.feature_importances_)

feature_importances = [(feature, round(importance, 9)) for feature, importance in zip(feats, importances)]

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]