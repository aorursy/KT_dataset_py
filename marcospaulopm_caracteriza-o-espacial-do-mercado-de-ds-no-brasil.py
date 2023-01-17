import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.ticker as ticker

import re

%matplotlib inline



pd.set_option('display.max_columns', 200)
respostas = pd.read_csv('../input/pesquisa-data-hackers-2019/datahackers-survey-2019-anonymous-responses.csv')

respostas.head()
mask_br = respostas["('P3', 'living_in_brasil')"] == 1

respostas_br = respostas[mask_br]

respostas_fora = respostas[~mask_br]

respostas_br.shape, respostas_fora.shape
campos_caract = ["('P5', 'living_state')", "('P19', 'is_data_science_professional')", "('P12', 'workers_number')", "('D6', 'anonymized_role')", "('P16', 'salary_range')", "('P1', 'age')", "('P2', 'gender')"]

respostas_br_caract = respostas_br[campos_caract].dropna(subset=["('P5', 'living_state')"])

respostas_br_caract["('P5', 'living_state')"].value_counts()
respostas_br_ds_caract = respostas_br_caract[respostas_br_caract["('P19', 'is_data_science_professional')"] == True]

respostas_br_ds_caract["('P5', 'living_state')"].value_counts()
respostas_br_caract["('P12', 'workers_number')"].value_counts()
col_ordem = ['Minas Gerais (MG)', 'São Paulo (SP)', 'Paraná (PR)', 'Rio Grande do Sul (RS)', 'Rio de Janeiro (RJ)', 'Espírito Santo (ES)', 'Santa Catarina (SC)']

ordem = ['de 1 a 5', 'de 6 a 10', 'de 11 a 50', 'de 51 a 100', 'de 101 a 500', 'de 501 a 1000', 'de 1001 a 3000', 'Acima de 3000']

ax = sns.catplot(x="('P12', 'workers_number')", data=respostas_br_caract, col="('P5', 'living_state')", col_wrap=4, kind="count", sharex=False, order=ordem, col_order=col_ordem)

ax.set_xticklabels(rotation=65, horizontalalignment='right')

ax.set(ylabel='Total de funcionários')

plt.tight_layout()

plt.subplots_adjust(top=0.9)

ax.fig.suptitle('Quantidade de funcionários nas empresas - Geral', fontsize='18', fontweight='bold')
respostas_br_ds_caract["('P12', 'workers_number')"].value_counts()
ordem = ['de 1 a 5', 'de 6 a 10', 'de 11 a 50', 'de 51 a 100', 'de 101 a 500', 'de 501 a 1000', 'de 1001 a 3000', 'Acima de 3000']

ax = sns.catplot(x="('P12', 'workers_number')", data=respostas_br_ds_caract, col="('P5', 'living_state')", col_wrap=4, kind="count", sharex=False, order=ordem, col_order=col_ordem)

ax.set_xticklabels(rotation=65, horizontalalignment='right')

ax.set(ylabel='Total de funcionários')

plt.tight_layout()

plt.subplots_adjust(top=0.9)

ax.fig.suptitle('Quantidade de funcionários nas empresas - Profissionais de DS', fontsize='18', fontweight='bold')
respostas_br_caract["('D6', 'anonymized_role')"].value_counts()
labels=['Outras', 'Analista de Dados', 'Analista de BI', 'Cientista de Dados', 'Analista de IM', 'Engenheiro', 'Dev. ou Eng. de Software',

       'Business Analyst', 'Engenheiro de Dados', 'Estatístico', 'Analista de Marketing', 'DBA/Adm. de BD', 'Eng. de ML', 'Economista']

ax = sns.catplot(x="('D6', 'anonymized_role')", data=respostas_br_caract, col="('P5', 'living_state')", col_wrap=4, kind="count", sharex=False, col_order=col_ordem)

ax.set_xticklabels(labels=labels, rotation=65, horizontalalignment='right')

ax.set(ylabel='Total de funcionários')

plt.tight_layout()

plt.subplots_adjust(top=0.9)

ax.fig.suptitle('Cargos - Geral', fontsize='18', fontweight='bold')
respostas_br_ds_caract["('D6', 'anonymized_role')"].value_counts()
labels=['Outras', 'Analista de Dados', 'Analista de BI', 'Cientista de Dados', 'Analista de IM', 'Engenheiro', 'Dev. ou Eng. de Software',

       'Business Analyst', 'Engenheiro de Dados', 'Estatístico', 'Analista de Marketing', 'DBA/Adm. de BD', 'Eng. de ML', 'Economista']

ax = sns.catplot(x="('D6', 'anonymized_role')", data=respostas_br_ds_caract, col="('P5', 'living_state')", col_wrap=4, kind="count", sharex=False, col_order=col_ordem)

ax.set_xticklabels(labels=labels, rotation=65, horizontalalignment='right')

ax.set(ylabel='Total de funcionários')

plt.tight_layout()

plt.subplots_adjust(top=0.9)

ax.fig.suptitle('Cargos - Profissionais de DS', fontsize='18', fontweight='bold')
respostas_br_caract["('P16', 'salary_range')"].value_counts()
def get_media_salarios_aparada(salary_range_str):

    search = re.search(r"de R\$ (\d+\.?\d+?)\/mês a R\$ (\d+\.?\d+?)\/mês", str(salary_range_str))

    salary_range = []

    if (search is not None):

        salary_range = list(search.groups())

    else:

        return 0

    

    if "." in salary_range[0]:

        salary_range[0] = salary_range[0].replace(".", "")

    

    if "." in salary_range[1]:

        salary_range[1] = salary_range[1].replace(".", "")

    

    return round((int(salary_range[0]) + int(salary_range[1])) / 2)
ordem=['Menos de R$ 1.000/mês', 'de R$ 1.001/mês a R$ 2.000/mês', 'de R$ 2.001/mês a R$ 3000/mês',  'de R$ 3.001/mês a R$ 4.000/mês',

       'de R$ 4.001/mês a R$ 6.000/mês', 'de R$ 6.001/mês a R$ 8.000/mês', 'de R$ 8.001/mês a R$ 12.000/mês', 'de R$ 12.001/mês a R$ 16.000/mês',

       'de R$ 16.001/mês a R$ 20.000/mês', 'de R$ 20.001/mês a R$ 25.000/mês', 'Acima de R$ 25.001/mês']

labels=['< R$ 1.000', 'R$ 1.001 a 2.000', 'R$ 2.001 a 3.000', 'R$ 3.001 a 4.000', 'R$ 4.001 a 6.000', 'R$ 6.001 a 8.000', 'R$ 8.001 a 12.000',

         'R$ 12.001 a 16.000', 'R$ 16.001 a 20.000', 'R$ 20.001 a 25.000', '> R$ 25.001']

ax = sns.catplot(x="('P16', 'salary_range')", data=respostas_br_caract, col="('P5', 'living_state')", col_wrap=4, kind="count", sharex=False, order=ordem, col_order=col_ordem)

ax.set_xticklabels(labels=labels, rotation=65, horizontalalignment='right')

ax.set(ylabel='Total de funcionários')

plt.tight_layout()

plt.subplots_adjust(top=0.9)

ax.fig.suptitle('Salários (por mês) - Geral', fontsize='18', fontweight='bold')
df_aux = respostas_br_caract.copy()

df_aux["('P16', 'salary_range')"] = df_aux["('P16', 'salary_range')"].apply(get_media_salarios_aparada)

df_aux = pd.DataFrame(df_aux.groupby("('P5', 'living_state')")["('P16', 'salary_range')"].mean().sort_values(ascending=False))

df_aux = df_aux.rename(columns={"('P16', 'salary_range')": "Média Geral"})

df_aux
respostas_br_ds_caract["('P16', 'salary_range')"].value_counts()
ordem=['Menos de R$ 1.000/mês', 'de R$ 1.001/mês a R$ 2.000/mês', 'de R$ 2.001/mês a R$ 3000/mês',  'de R$ 3.001/mês a R$ 4.000/mês',

       'de R$ 4.001/mês a R$ 6.000/mês', 'de R$ 6.001/mês a R$ 8.000/mês', 'de R$ 8.001/mês a R$ 12.000/mês', 'de R$ 12.001/mês a R$ 16.000/mês',

       'de R$ 16.001/mês a R$ 20.000/mês', 'de R$ 20.001/mês a R$ 25.000/mês', 'Acima de R$ 25.001/mês']

labels=['< R$ 1.000', 'R$ 1.001 a 2.000', 'R$ 2.001 a 3.000', 'R$ 3.001 a 4.000', 'R$ 4.001 a 6.000', 'R$ 6.001 a 8.000', 'R$ 8.001 a 12.000',

         'R$ 12.001 a 16.000', 'R$ 16.001 a 20.000', 'R$ 20.001 a 25.000', '> R$ 25.001']

ax = sns.catplot(x="('P16', 'salary_range')", data=respostas_br_ds_caract, col="('P5', 'living_state')", col_wrap=4, kind="count", sharex=False, order=ordem, col_order=col_ordem)

ax.set_xticklabels(labels=labels, rotation=65, horizontalalignment='right')

ax.set(ylabel='Total de funcionários')

plt.tight_layout()

plt.subplots_adjust(top=0.9)

ax.fig.suptitle('Salários (por mês) - Profissionais de DS', fontsize='18', fontweight='bold')
df_aux_ds = respostas_br_ds_caract.copy()

df_aux_ds["('P16', 'salary_range')"] = df_aux_ds["('P16', 'salary_range')"].apply(get_media_salarios_aparada)

df_aux_ds = pd.DataFrame(df_aux_ds.groupby("('P5', 'living_state')")["('P16', 'salary_range')"].mean().sort_values(ascending=False))

df_aux_ds = df_aux_ds.rename(columns={"('P16', 'salary_range')": "Média DS"})

df_aux = df_aux.merge(df_aux_ds, left_on="('P5', 'living_state')", right_on="('P5', 'living_state')")

df_aux["Estado"] = df_aux.index
df_aux_melted = df_aux.melt('Estado', var_name='cols',  value_name='vals')

df_aux_melted
ax = sns.catplot(x="Estado", y="vals", data=df_aux_melted, hue='cols', kind="bar", order=col_ordem)

ax.set_xticklabels(rotation=65, horizontalalignment='right')

ax.set(ylabel='Salário médio (em R$)')

plt.subplots_adjust(top=0.9)

ax.fig.suptitle('Salários médios (por mês)', fontsize='18', fontweight='bold')
ax = sns.distplot(respostas_br_caract["('P1', 'age')"])

ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
print("Média de Idade Global: {}".format(respostas_br_caract["('P1', 'age')"].mean()))

print("Média de Idade Masculino: {}".format(respostas_br_caract[respostas_br_caract["('P2', 'gender')"] == "Masculino"]["('P1', 'age')"].mean()))

print("Média de Idade Feminino: {}".format(respostas_br_caract[respostas_br_caract["('P2', 'gender')"] == "Feminino"]["('P1', 'age')"].mean()))
sns.catplot(x="('P2', 'gender')", data=respostas_br_caract, kind="count")
mask_homem = respostas_br_caract["('P2', 'gender')"] == 'Masculino'

mask_mulher = respostas_br_caract["('P2', 'gender')"] == 'Feminino'

(respostas_br_caract[mask_mulher].groupby(["('P5', 'living_state')"])["('P2', 'gender')"].value_counts() / respostas_br_caract.groupby(["('P5', 'living_state')"])["('P2', 'gender')"].count()).sort_values(ascending=False)
mask = respostas_br_caract["('P2', 'gender')"] == 'Masculino'

ax = sns.catplot(x="('P1', 'age')", data=respostas_br_caract[mask], col="('P5', 'living_state')", col_wrap=4, kind="count", sharex=False)

ax.set_xticklabels(rotation=90, horizontalalignment='right')

ax.set(ylabel='Total de funcionários')

plt.tight_layout()

plt.subplots_adjust(top=0.9)

ax.fig.suptitle('Idade dos funcionários - Masculino', fontsize='18', fontweight='bold')
respostas_br_caract[respostas_br_caract["('P2', 'gender')"] == 'Masculino'].groupby(["('P5', 'living_state')"])["('P1', 'age')"].mean().sort_values(ascending=False)
mask = respostas_br_caract["('P2', 'gender')"] == 'Feminino'

ax = sns.catplot(x="('P1', 'age')", data=respostas_br_caract[mask], col="('P5', 'living_state')", col_wrap=4, kind="count", sharex=False, col_order=col_ordem)

ax.set_xticklabels(rotation=90, horizontalalignment='right')

ax.set(ylabel='Total de funcionários')

plt.tight_layout()

plt.subplots_adjust(top=0.9)

ax.fig.suptitle('Idade dos funcionários - Feminino', fontsize='18', fontweight='bold')
respostas_br_caract[respostas_br_caract["('P2', 'gender')"] == 'Feminino'].groupby(["('P5', 'living_state')"])["('P1', 'age')"].mean().sort_values(ascending=False)
ax = sns.distplot(respostas_br_ds_caract["('P1', 'age')"])

ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
print("Média de Idade Global: {}".format(respostas_br_ds_caract["('P1', 'age')"].mean()))

print("Média de Idade Masculino: {}".format(respostas_br_ds_caract[respostas_br_ds_caract["('P2', 'gender')"] == "Masculino"]["('P1', 'age')"].mean()))

print("Média de Idade Feminino: {}".format(respostas_br_ds_caract[respostas_br_ds_caract["('P2', 'gender')"] == "Feminino"]["('P1', 'age')"].mean()))
sns.catplot(x="('P2', 'gender')", data=respostas_br_ds_caract, kind="count", order=['Masculino', 'Feminino'])
razao_global = len(respostas_br_caract[respostas_br_caract["('P2', 'gender')"] == 'Masculino']) / len(respostas_br_caract[respostas_br_caract["('P2', 'gender')"] == 'Feminino'])

razao_ds = len(respostas_br_ds_caract[respostas_br_caract["('P2', 'gender')"] == 'Masculino']) / len(respostas_br_ds_caract[respostas_br_caract["('P2', 'gender')"] == 'Feminino'])

print("Proporção global entre homens e mulheres: {}".format(razao_global))

print("Proporção em DS entre homens e mulheres: {}".format(razao_ds))                                                                                               
mask_homem = respostas_br_ds_caract["('P2', 'gender')"] == 'Masculino'

mask_mulher = respostas_br_ds_caract["('P2', 'gender')"] == 'Feminino'

(respostas_br_ds_caract[mask_mulher].groupby(["('P5', 'living_state')"])["('P2', 'gender')"].value_counts() / respostas_br_ds_caract.groupby(["('P5', 'living_state')"])["('P2', 'gender')"].count()).sort_values(ascending=False)
mask = respostas_br_ds_caract["('P2', 'gender')"] == 'Masculino'

ax = sns.catplot(x="('P1', 'age')", data=respostas_br_ds_caract[mask], col="('P5', 'living_state')", col_wrap=4, kind="count", sharex=False, col_order=col_ordem)

ax.set_xticklabels(rotation=90, horizontalalignment='right')

ax.set(ylabel='Total de funcionários')

plt.tight_layout()

plt.subplots_adjust(top=0.9)

ax.fig.suptitle('Idade dos funcionários - Masculino', fontsize='18', fontweight='bold')
mask = respostas_br_ds_caract["('P2', 'gender')"] == 'Feminino'

ax = sns.catplot(x="('P1', 'age')", data=respostas_br_ds_caract[mask], col="('P5', 'living_state')", col_wrap=4, kind="count", sharex=False, col_order=col_ordem)

ax.set_xticklabels(rotation=90, horizontalalignment='right')

ax.set(ylabel='Total de funcionários')

plt.tight_layout()

plt.subplots_adjust(top=0.9)

ax.fig.suptitle('Idade dos funcionários - Feminino', fontsize='18', fontweight='bold')
respostas_br_ds_caract[respostas_br_caract["('P2', 'gender')"] == 'Feminino'].groupby(["('P5', 'living_state')"])["('P1', 'age')"].mean().sort_values(ascending=False)
for sexo in ['Masculino', 'Feminino']:

    df_aux = respostas_br_caract.copy()

    df_aux = pd.DataFrame(respostas_br_caract[respostas_br_caract["('P2', 'gender')"] == sexo].groupby(["('P5', 'living_state')"])["('P1', 'age')"].mean().sort_values(ascending=False))

    df_aux = df_aux.rename(columns={"('P1', 'age')": "Média Geral"})



    df_aux_ds = respostas_br_ds_caract.copy()

    df_aux_ds = pd.DataFrame(respostas_br_ds_caract[respostas_br_ds_caract["('P2', 'gender')"] == sexo].groupby(["('P5', 'living_state')"])["('P1', 'age')"].mean().sort_values(ascending=False))

    df_aux_ds = df_aux_ds.rename(columns={"('P1', 'age')": "Média DS"})

    df_aux = df_aux.merge(df_aux_ds, left_on="('P5', 'living_state')", right_on="('P5', 'living_state')")

    df_aux["Estado"] = df_aux.index



    df_aux_melted = df_aux.melt('Estado', var_name='cols',  value_name='vals')



    ax = sns.catplot(x="Estado", y="vals", data=df_aux_melted, hue='cols', kind="bar", order=col_ordem)

    ax.set_xticklabels(rotation=65, horizontalalignment='right')

    ax.set(ylabel='Média das idades')

    plt.subplots_adjust(top=0.9)

    ax.fig.suptitle('Média das idades - {}'.format(sexo), fontsize='18', fontweight='bold')
from category_encoders import OneHotEncoder
def get_media_salarios(salary_range_str):

    search = re.search(r"de R\$ (\d+\.?\d+?)\/mês a R\$ (\d+\.?\d+?)\/mês", str(salary_range_str))

    search_acima = re.search(r"Acima de R\$ (\d+\.?\d+?)\/mês", str(salary_range_str))

    search_abaixo = re.search(r"Menos de R\$ (\d+\.?\d+?)\/mês", str(salary_range_str))

    salary_range = []

    if (search is not None):

        salary_range = list(search.groups())

    elif (search_acima is not None):

        return 30000

    elif (search_abaixo is not None):

        return 500

    else:

        return 0

    

    if "." in salary_range[0]:

        salary_range[0] = salary_range[0].replace(".", "")

    

    if "." in salary_range[1]:

        salary_range[1] = salary_range[1].replace(".", "")

    

    return round((int(salary_range[0]) + int(salary_range[1])) / 2)



def get_media_funcionarios(func_range_str):

    search = re.search(r"de (\d+) a (\d+)", str(func_range_str))

    search_acima = re.search(r"Acima de (\d+)", str(func_range_str))

    func_range = []

    if (search is not None):

        func_range = list(search.groups())

    elif (search_acima is not None):

        return 3250

    else:

        return 0

    

    return round((int(func_range[0]) + int(func_range[1])) / 2)



def transform_gender(gender):

    return int(gender == 'Feminino')
df_aux = respostas_br_ds_caract.copy()

df_aux["('P16', 'salary_range')"] = df_aux["('P16', 'salary_range')"].apply(get_media_salarios)

df_aux["('P2', 'gender')"] = df_aux["('P2', 'gender')"].apply(transform_gender)

df_aux["('P12', 'workers_number')"] = df_aux["('P12', 'workers_number')"].apply(get_media_funcionarios)

ohe_cargo = OneHotEncoder(cols=["('D6', 'anonymized_role')"], use_cat_names=True, drop_invariant=True)

df_aux = ohe_cargo.fit_transform(df_aux)

df_aux = df_aux.drop("('D6', 'anonymized_role')_nan", axis=1)

df_aux.head()
corr = df_aux.corr()

sns.heatmap(corr)