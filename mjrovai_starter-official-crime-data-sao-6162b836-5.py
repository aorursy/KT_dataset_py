import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('/kaggle/input/ds_SSP_MonthlyOccurences_SP-BR_utf8_2001-2020.csv',low_memory=False)
df.shape
df.head()
df.Natureza.value_counts()
df.info()
df.Jun.value_counts()
df = df.replace('...', 0.0)
df['Jun'] = df['Jun'].astype('float64')
df['Jul'] = df['Jul'].astype('float64')
df['Ago'] = df['Ago'].astype('float64')
df['Set'] = df['Set'].astype('float64')
df['Out'] = df['Out'].astype('float64')
df['Nov'] = df['Nov'].astype('float64')
df['Dez'] = df['Dez'].astype('float64')
df.info()
df_hom_dol_sampa = df[(df.Natureza == 'HOMICÍDIO DOLOSO (2)') & (df.Cidade == 'São Paulo')].sort_values('Ano')
df_hom_dol_sampa.head(2)
plt.figure(figsize=(10, 5))
sns.barplot(x='Ano', y='Total', data=df_hom_dol_sampa, color='firebrick')
plt.xticks(rotation=90)
plt.title('Homicídios Dolosos - Cidade de São Paulo', fontsize=14)
plt.xlabel('Ano', fontsize=10)
plt.ylabel('Número de homicídios', fontsize=10);
def plot_ssp_crimes(df, type_occurrence, city, graph='Total'):
    dt = df[(df.Natureza == type_occurrence)
                          & (df.Cidade == city)].sort_values('Ano')

    plt.figure(figsize=(10, 5))
    sns.barplot(x='Ano', y=graph, data=dt, color='firebrick')
    plt.xticks(rotation=90)
    
    if type_occurrence[-1] == ')': 
        title = type_occurrence[:-4]
    else:
        title = type_occurrence
    plt.title('{} em {}'.format(title, city), fontsize=14)
    plt.xlabel('Ano', fontsize=10)
    plt.ylabel('Número de ocorrências', fontsize=10);
plot_ssp_crimes(df, 'ROUBO DE CARGA', 'Campinas', graph='Total')
plot_ssp_crimes(df, 'TOTAL DE ESTUPRO (4)', 'Osasco', graph='Total')
df_sampa = df[(df.Cidade == 'São Paulo')].sort_values('Ano')
dt = df_sampa.groupby(['Natureza','Ano']).sum()
dt.isnull().sum()
dt.head(10)
dt = dt.T
dt.head(2)
list(df.Natureza.unique())
hom_sp = dt['Nº DE VÍTIMAS EM HOMICÍDIO DOLOSO (3)'] + dt[
    'HOMICÍDIO CULPOSO POR ACIDENTE DE TRÂNSITO'] + dt[
        'HOMICÍDIO CULPOSO OUTROS'] + dt['Nº DE VÍTIMAS EM LATROCÍNIO']
hom_sp
hom_sp = hom_sp.T.reset_index()
hom_sp['Num. Month'] = [12 for _ in range(len(hom_sp))]
hom_sp.at[18, 'Num. Month'] = 5
hom_sp['HOMICIDIOS (Média Mensal)'] = (hom_sp['Total']/hom_sp['Num. Month']).astype('int64')
hom_sp
def plot_sub_dataset(dt, type_occurrence, city):
    plt.figure(figsize=(10, 5))
    sns.barplot(x='Ano', y=type_occurrence, data=dt, color='firebrick')
    plt.xticks(rotation=90);

    plt.title('{} - {}'.format(type_occurrence, city), fontsize=14)
    plt.xlabel('Ano', fontsize=10)
    plt.ylabel('Casos', fontsize=10);
plot_sub_dataset(hom_sp, 'HOMICIDIOS (Média Mensal)', 'São Paulo')
def get_sub_dataset(df, type_occurrence, city):
    dt = df[(df.Cidade == city)].sort_values('Ano')
    dt = dt.groupby(['Natureza','Ano']).sum().T
    dt = dt[type_occurrence].T.reset_index()
    dt['Num. Month'] = [12 for _ in range(len(dt))]
    dt.at[18, 'Num. Month'] = 5
    dt['Total Year'] = dt.iloc[:, 1:13].sum(axis=1)
    if type_occurrence[-1] == ')': 
        label = type_occurrence[:-4]+' (Média Mensal)'
    else:
        label = type_occurrence +' (Média Mensal)'
    dt[label] = (dt['Total Year']/dt['Num. Month']).astype('int64')
    plot_sub_dataset(dt, label, city)
    return dt
estupros_sp = get_sub_dataset(df, 'TOTAL DE ESTUPRO (4)', 'São Paulo')
roubo_veiculo_sp = get_sub_dataset(df, 'ROUBO DE VEÍCULO', 'São Paulo')
roubo_veiculo_sp = get_sub_dataset(df, 'ROUBO DE VEÍCULO', 'Osasco')
plt.figure(figsize=(10, 5))
plt.plot(hom_sp['Ano'], hom_sp['HOMICIDIOS (Média Mensal)']);
plt.plot(hom_sp['Ano'], estupros_sp['TOTAL DE ESTUPRO (Média Mensal)'])
plt.xticks(rotation=90);
plt.gca().set_xticks(hom_sp['Ano'].unique());

plt.title('Medias Mensais de Homicídios versus Estupros na cidade de São Paulo', fontsize=14)
plt.xlabel('Ano', fontsize=10)
plt.ylabel('Casos', fontsize=10);