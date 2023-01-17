## Estudo dos salários de servidores da Universida
%matplotlib inline

import pandas as pd

import numpy as np

import re

import matplotlib.pyplot as plt

import scipy.stats as stats

from pandas.tools.plotting import table



#pd.set_option('display.max_columns', None)

#pd.set_option('display.max_rows', None)



plt.style.use('ggplot')
colunas = ["cpf", "nome", "link_det_serv","sal_bruto", "sal_liq", "link_det_remu", "cargo","vinculo", "carga_horaria"]

teto = 17094.77 #R$ 11.926,49 liquido
df = pd.read_csv("../input/servidores_v4_UFPA_todos.csv",  sep='|', thousands = '.' , decimal = ',',  names = colunas)
len(df)
df.shape
df.dtypes
df.loc[ df.sal_liq == 0 ]
len(df.loc[ df.sal_liq == 0 ])
df.loc[ df.sal_bruto == 0 ]
len(df.loc[ df.sal_bruto == 0 ])
df.loc[df.sal_liq == 0.0, 'sal_liq'] =  np.nan
df.loc[df.sal_bruto == 0.0, 'sal_bruto'] =  np.nan
maskara_bruto = (df['sal_bruto'] > teto)

bruto_maiores = df.loc[maskara_bruto]
bruto_maiores
maskara_liquido = (df['sal_liq'] > teto)

liquido_maiores = df.loc[maskara_liquido]
liquido_maiores
df['diff_bruto'] =  df['sal_bruto'] - teto
df['diff_liq'] =  df['sal_liq'] - teto
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

result_maiores_per_sal_liq.to_html('maiores_sal_liq_por_cargo_UFPA_SET.html')

result_maiores_per_sal_liq
result_maiores_per_sal_bruto = df.loc[df.groupby('cargo').sal_bruto.agg('idxmax')]

result_maiores_per_sal_bruto.to_html('maiores_sal_bruto_por_cargo_UFPA_SET.html')

result_maiores_per_sal_bruto
grouped =  df.groupby(['cargo','nome'])['sal_liq']

agregado = grouped.agg(np.max)

agregado
result_menores_per_sal_liq = df.loc[df.groupby('cargo').sal_liq.agg('idxmin')]

result_menores_per_sal_liq.to_html('menores_sal_bruto_por_cargo_UFC_SET.html')

result_menores_per_sal_liq
result_menores_per_sal_bruto = df.loc[df.groupby('cargo').sal_bruto.agg('idxmin')]

result_menores_per_sal_bruto.to_html('menores_sal_bruto_por_cargo_UFC_SET.html')

result_menores_per_sal_bruto
grouped2 =  df.groupby(['cargo','nome'])['sal_liq']

agregado2 = grouped2.agg(np.min)

agregado2
plt.hist(list(agregado), range=(0, 30000))

plt.show()
plt.hist(list(agregado2), range=(0, 30000))

plt.show()
serie_cargo = df["cargo"].value_counts()

serie_cargo
serie_cargo.sum()
serie_vinculo = df["vinculo"].value_counts()

serie_vinculo
serie_carga_horaria = df["carga_horaria"].value_counts()

serie_carga_horaria
fig = plt.figure(figsize=(40,40), dpi=80)

#ax = plt.subplot(11)



plt.legend(labels=serie_vinculo.index, loc="best")

explode = (0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.3, 0.4, 0.4, 0.5,0.6, 1.1, 1.1)



serie_vinculo.plot(kind='pie',explode=explode, autopct='%0.2f%%', startangle=90, fontsize=14)
fig = plt.figure(figsize=(20,20), dpi=200)

ax = plt.subplot(111)



serie_cargo.plot(kind='pie', ax=ax, autopct='%1.1f%%', startangle=270, fontsize=8)
cargos =  df['cargo']

print(type(cargos))

cargos_tratados =  pd.Series.dropna(cargos)
print(type(cargos_tratados))
lista_cargos = cargos_tratados.tolist()

print(lista_cargos)
import itertools
lista_cargos.sort()
lista_cargos_unique =  list(k for k,_ in itertools.groupby(lista_cargos))

lista_cargos_unique
groupby_regiment = df['sal_liq'].groupby(df['cargo'])

groupby_regiment
list(df['sal_liq'].groupby(df['cargo']))
df['sal_liq'].groupby(df['cargo']).describe()
df2 = df
df2.dropna(how='any', inplace= True)
df2.loc[ np.isnan(df2.sal_liq ) ]
df2.loc[ np.isnan(df2.sal_bruto ) ]
funcoes = lista_cargos_unique

resul = []

resul_med = []

resul_sum = []

i = 0

for funcao in funcoes:

    resul.append([i, funcao, len(df2[df2.cargo == funcao])])

    resul_med.append([funcao, len(df2[df2.cargo == funcao]), np.median(df2[df2.cargo == funcao].sal_liq)])

    resul_sum.append([i, funcao, np.sum(df2[df2.cargo == funcao].sal_liq)])

    i += 1

resul = np.array(resul)

resul_med = np.array(resul_med)

resul_sum = np.array(resul_sum)
resul_med

dat_resul = pd.DataFrame(resul_med, columns=("Função", "Quantidade de Funcionários", "Salário(líquido) Mediano"))

dat_resul.to_html("mediana_salarial_porCargo-ufc_set.html")
resul_med
type(resul_med)
mydic = dict(enumerate(resul_sum))

mydic
lista_resul_med =  resul_med.tolist

print(lista_resul_med)
my_list = map(lambda x: x[0], resul_med)

print(my_list)

print(list(my_list))
df_adm = df2[df2.cargo == 'ADMINISTRADOR']

df_adm.head(10)
maskara_liquido_menores = (df_adm['sal_liq'] < 6490.04)

menores_adm = df_adm.loc[maskara_liquido_menores]
menores_adm
len(menores_adm)
maskara_liquido = (df_adm['sal_liq'] >= 6490.04)

maiores_adm = df_adm.loc[maskara_liquido]
maiores_adm
len(maiores_adm)
df_contador = df2[df2.cargo == 'CONTADOR']

df_contador
len(df_contador)
maskara_contador_liq_menos = (df_contador['sal_liq'] < 6075.18)

menores_contador = df_contador.loc[maskara_contador_liq_menos]

menores_contador
len(menores_contador)
maskara_contador_liq_mais = (df_contador['sal_liq'] > 6075.18)

maiores_contador = df_contador.loc[maskara_contador_liq_mais]
maiores_contador
len(maiores_contador)
import matplotlib.pyplot as plt
plt.scatter(df_contador.sal_liq, df_contador.sal_liq)



plt.axhline(y= 6075.18, color = 'r', linestyle = '-')

plt.title('Dispersão sal. líquido - Cargo Contador')

plt.rcParams["figure.figsize"] = [15,7]

plt.figure(dpi=40)

plt.show()
df_prof_magisterio = df2[df2.cargo == 'PROFESSOR DO MAGISTERIO SUPERIOR']

df_prof_magisterio
maskara_prof_liq_mais = (df_prof_magisterio['sal_liq'] >= 8175.99)

maiores_professor = df_prof_magisterio.loc[maskara_prof_liq_mais]
maskara_prof_liq_menos = (df_prof_magisterio['sal_liq'] < 8175.99)

menores_professor = df_prof_magisterio.loc[maskara_prof_liq_menos]
maiores_professor
menores_professor
len(maiores_professor)
len(menores_professor)
plt.scatter(df_prof_magisterio.sal_liq, df_prof_magisterio.sal_liq)

plt.axhline(y= 6075.18, color = 'b', linestyle = '-')

plt.title('Dispersão sal. líquido - Cargo Professor do Magistério Superior')

plt.ylabel('Salário líquido')

plt.xlabel('Salário líquido')

plt.rcParams["figure.figsize"] = [15,7]

plt.legend('reta - R$ 8.175,99')

plt.figure(dpi=40)

#plt.figure(figsize= (50,50), dpi=200)



plt.show()