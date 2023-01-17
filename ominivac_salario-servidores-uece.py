%matplotlib inline

import pandas as pd

import numpy as np

import re

import matplotlib.pyplot as plt

import scipy.stats as stats

from pandas.tools.plotting import table





plt.style.use('ggplot')

import scipy.stats as stats

from pandas.tools.plotting import table



headers = ["nome", "orgao", "cargo", "sal_bruto", "sal_liq"]

governor_salary = 17094.77 #R$ 11.926,49 liquido
df = pd.read_csv("../input/uece.txt", delimiter='\t', index_col=False,decimal=',' ,names=headers, header=0)
df.head(5)
df.dtypes
df.head(5)
df['sal_bruto'] = df['sal_bruto'].str.strip('R$')
df['sal_liq'] = df['sal_liq'].str.strip('R$')
df.head(5)
df.dtypes
df['sal_bruto'] = df['sal_bruto'].str.replace('.', '')
df['sal_liq'] = df['sal_liq'].str.replace('.', '')
df.head(5)
df.dtypes
df.to_csv('uece-pandas.txt',  encoding='utf-8', header= False , index= False, sep= '|')
df = pd.read_csv("uece-pandas.txt", delimiter='|', index_col=False,decimal=',' ,names=headers )
df.dtypes
df.head(5)
df.loc[ df.sal_liq == 0 ]
df.loc[ df.sal_bruto == 0 ]
df.loc[df.sal_bruto == 0.0, 'sal_bruto'] =  np.nan
df.loc[ np.isnan(df.sal_bruto) ]
df.loc[df.sal_liq == 0.0, 'sal_liq'] =  np.nan
df.loc[ np.isnan(df.sal_liq ) ]
df.drop(1362, inplace=True)
df.loc[ np.isnan(df.sal_liq ) ]
maskara_bruto = (df['sal_bruto'] > governor_salary)

bruto_maiores = df.loc[maskara_bruto]
bruto_maiores
maskara_liquido = (df['sal_liq'] > governor_salary)

liquido_maiores = df.loc[maskara_liquido]
liquido_maiores
df['diff_bruto'] =  df['sal_bruto'] - governor_salary
df['diff_liq'] =  df['sal_liq'] - governor_salary
filtro_liq = (df['diff_liq'] > 0.0)

df_liq_maiores = df.loc[filtro_liq]

df_liq_maiores
filtro_bruto = (df['diff_bruto'] > 0.0)

df_bruto_maiores = df.loc[filtro_bruto]

df_bruto_maiores
valor_mais_liq = df_liq_maiores['diff_liq'].sum()

print('Total a mais pelo sal. líquido: R$ {:,.2f}'.format(valor_mais_liq) + ' por ano')
valor_mais_bruto = df_bruto_maiores['diff_bruto'].sum()

print('Total a mais pelo sal. bruto: R$ {:,.2f}'.format(valor_mais_bruto) + ' por ano')
result_maiores_per_sal_liq = df.loc[df.groupby('cargo').sal_liq.agg('idxmax')]

result_maiores_per_sal_liq.to_html('maiores_sal_liq_por_cargo_UECE_.html')

result_maiores_per_sal_liq
grouped2 =  df.groupby(['cargo','nome'])['sal_liq']

agregado2 = grouped2.agg(np.min)

agregado2
result_menores_per_sal_liq = df.loc[df.groupby('cargo').sal_liq.agg('idxmin')]

result_menores_per_sal_liq.to_html('menores_sal_liquido_por_cargo_UECE_SET.html')

result_menores_per_sal_liq
grouped =  df.groupby(['cargo','nome'])['sal_liq']

agregado = grouped.agg(np.max)

agregado
result_maiores_per_sal_bruto = df.loc[df.groupby('cargo').sal_bruto.agg('idxmax')]

result_maiores_per_sal_bruto.to_html('maiores_sal_bruto_por_cargo_UECE.html')

result_maiores_per_sal_bruto
result_menores_per_sal_bruto = df.loc[df.groupby('cargo').sal_bruto.agg('idxmin')]

result_menores_per_sal_bruto.to_html('menores_sal_bruto_por_cargo_UECE_SET.html')

result_menores_per_sal_bruto
serie_cargo = df["cargo"].value_counts()

serie_cargo
print (len(serie_cargo), ' cargos distintos no conjunto de dados' )
fig = plt.figure(figsize=(20,20), dpi=200)

ax = plt.subplot(111)



serie_cargo.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=270, fontsize=8)
lista_cargos = serie_cargo.index.values.tolist
lista_cargos_unique = list(lista_cargos() )
lista_cargos_unique
len(lista_cargos_unique)
groupby_regiment = df['sal_liq'].groupby(df['cargo'])

groupby_regiment
list(df['sal_liq'].groupby(df['cargo']))
df['sal_liq'].groupby(df['cargo']).describe()
maskara_vigia = (df['cargo'] =='VIGIA')

vigias = df.loc[maskara_vigia]
funcoes = lista_cargos_unique

resul = []

resul_med = []

resul_sum = []

i = 0

for funcao in funcoes:

    resul.append([i, funcao, len(df[df.cargo == funcao])])

    resul_med.append([funcao, len(df[df.cargo == funcao]), np.median(df[df.cargo == funcao].sal_liq)])

    resul_sum.append([i, funcao, np.sum(df[df.cargo == funcao].sal_liq)])

    i += 1

resul = np.array(resul)

resul_med = np.array(resul_med)

resul_sum = np.array(resul_sum)
resul_med

dat_resul = pd.DataFrame(resul_med, columns=("Função", "Quantidade de Funcionários", "Salário(líquido) Mediano"))

dat_resul.to_html("mediana_salarial_porCargo-uece_ago.html")
resul_med
type(resul_med)
mydic = dict(enumerate(resul_sum))

mydic
lista_resul_med =  resul_med.tolist

print(lista_resul_med)
my_list = map(lambda x: x[0], resul_med)

print(my_list)

print(list(my_list))
df_adm = df[df.cargo == 'ADMINISTRADOR']

df_adm.head(10)
maskara_liquido_menores = (df_adm['sal_liq'] < 13566.0)

menores_adm = df_adm.loc[maskara_liquido_menores]
menores_adm
len(menores_adm)
maskara_liquido_maiores = (df_adm['sal_liq'] > 13566.0)

maiores_adm = df_adm.loc[maskara_liquido_maiores]
maiores_adm
len(maiores_adm)
plt.scatter(df_adm.sal_liq, df_adm.sal_bruto)

plt.axhline(y= 6075.18, color = 'r', linestyle = '-')

plt.title('Dispersão sal. líquido - Cargo Administrador')

plt.rcParams["figure.figsize"] = [15,7]

plt.figure(dpi=40)

plt.show()
df_professor = df[df.cargo == 'PROFESSOR']

df_professor
maskara_prof_liq_mais = (df_professor['sal_liq'] > 14559.29)

maiores_professor = df_professor.loc[maskara_prof_liq_mais]
maiores_professor
len(maiores_professor)
maskara_prof_liq_menos = (df_professor['sal_liq'] < 14559.29)

menores_professor = df_professor.loc[maskara_prof_liq_menos]
menores_professor
len(menores_professor)
plt.scatter(df_professor.sal_liq, df_professor.sal_liq)

plt.axhline(y= 14559.29, color = 'r', linestyle = '-')

plt.title('Dispersão sal. líquido - Cargo Professor -UECE - AGO/17')

plt.rcParams["figure.figsize"] = [15,7]

plt.figure(dpi=40)

plt.show()
plt.scatter(df_professor.sal_bruto, df_professor.sal_bruto)

plt.axhline(y= 14559.29, color = 'r', linestyle = '-')

plt.title('Dispersão sal. Bruto - Cargo Professor -UECE - AGO/17')

plt.rcParams["figure.figsize"] = [15,7]

plt.figure(dpi=40)

plt.show()