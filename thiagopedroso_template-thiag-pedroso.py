import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset = pd.read_csv('../input/dataviz-facens-20182-aula-1-exerccio-2/anv.csv', delimiter=',')

dataset.info()
dataset.describe()
dataset.head(20)
dataset.tail(20)
datasetviz = dataset[['aeronave_pmd_categoria','aeronave_pmd','aeronave_ano_fabricacao','aeronave_fabricante',
                      'aeronave_tipo_veiculo',
                      'aeronave_operador_categoria',
                      'aeronave_motor_tipo',
                      'aeronave_motor_quantidade',
                      'aeronave_fase_operacao',
                      'total_fatalidades']]
datasetviz.tail(100)
arr_classific = [['aeronave_pmd_categoria','Qualitativa Nominal'],
                 ['aeronave_pmd_categoria','Qualitativa Nominal'],
                ['aeronave_ano_fabricacao','Qualitativa Ordinal'],
              ['aeronave_tipo_veiculo','Qualitativa Nominal'],
              ['aeronave_operador_categoria','Qualitativa Nominal'],
              ['aeronave_motor_tipo','Qualitativa Nominal'],
              ['aeronave_motor_quantidade','Qualitativa Ordinal'],
              ['aeronave_fase_operacao','Qualitativa Nominal'],
              ['total_fatalidades','Quantitativa Discreta'],
                ['aeronave_pmd','Quantitativa Continua']
                ]
dtclassific = pd.DataFrame(data=arr_classific, columns=['Variable','Classification'])
dtclassific
pmd=datasetviz[['aeronave_pmd_categoria','aeronave_pmd']].copy()
pmd[pmd['aeronave_pmd_categoria']=='***']
datasetviz['aeronave_pmd_categoria'].replace(
    to_replace=['***'],
    value='SEM PESO',
    inplace=True
)

df_freq_fabric = pd.DataFrame(data=datasetviz['aeronave_pmd_categoria'].value_counts())
df_freq_fabric
#datasetviz[datasetviz['aeronave_ano_fabricacao'].isnull()]
datasetviz.update(datasetviz['aeronave_ano_fabricacao'].fillna(2008.0))
datasetviz['aeronave_ano_fabricacao'].replace(0,1936,inplace=True)

datasetviz['aeronave_ano_fabricacao'] = datasetviz['aeronave_ano_fabricacao'].astype(int)
df_ano_fabric = pd.DataFrame(data=datasetviz['aeronave_ano_fabricacao'].value_counts())
df_ano_fabric.reset_index().sort_values(by='index',ascending=True)
datasetviz['aeronave_fabricante'].replace(
    to_replace=['***'],
    value='NAO INFORMADO',
    inplace=True
)

df_freq_fabric = pd.DataFrame(data=datasetviz['aeronave_fabricante'].value_counts())
df_freq_fabric

df_freq_type = pd.DataFrame(data=datasetviz['aeronave_tipo_veiculo'].value_counts())
df_freq_type
df_freq_oper_cat = pd.DataFrame(data=datasetviz['aeronave_operador_categoria'].value_counts())
df_freq_oper_cat
datasetviz['aeronave_motor_tipo'].replace(
    to_replace=['***'],
    value='NAO INFORMADO',
    inplace=True
)
df_freq_motor_type = pd.DataFrame(data=datasetviz['aeronave_motor_tipo'].value_counts())
df_freq_motor_type
datasetviz['aeronave_motor_quantidade'].replace(
    to_replace=['***'],
    value='NAO INFORMADO',
    inplace=True
)
df_freq_motor_quant = pd.DataFrame(data=datasetviz['aeronave_motor_quantidade'].value_counts())
df_freq_motor_quant
datasetviz['aeronave_fase_operacao'].replace(
    to_replace=['***'],
    value='NAO INFORMADO',
    inplace=True
)
df_freq_fase_oper = pd.DataFrame(data=datasetviz['aeronave_fase_operacao'].value_counts())
df_freq_fase_oper

df_freq_type = datasetviz['aeronave_tipo_veiculo'].value_counts().rename_axis('Tipo').reset_index(name='Freq')
df_freq_type = df_freq_type.sort_values('Freq', ascending=False)

df_freq_type.plot(x ='Tipo', y='Freq', kind = 'barh')
plt.show()
#agrupado por fatalidade
#dt_operador_fatal = datasetviz.loc[datasetviz['total_fatalidades'] > 0][['aeronave_operador_categoria','total_fatalidades']].copy()
dt_operador_fatal = datasetviz[['aeronave_operador_categoria','total_fatalidades']].copy()
dt_operador_fatal = dt_operador_fatal.groupby(by=['aeronave_operador_categoria'],as_index=False)['total_fatalidades'].sum()
dt_operador_fatal = dt_operador_fatal.sort_values('aeronave_operador_categoria', ascending=True)
#agrupado por ocorrencias
dt_operador_freq = datasetviz['aeronave_operador_categoria'].value_counts().reset_index().rename(columns={'index':'Operador','aeronave_operador_categoria':'Ocorrencias'})
dt_operador_freq = dt_operador_freq.sort_values('Operador', ascending=True)
#adicionando novo campo de ocorrencias no dataframe
new_ocorr = []
for index,row in dt_operador_fatal.iterrows():
    new_ocorr.append(dt_operador_freq.loc[dt_operador_freq['Operador']==row['aeronave_operador_categoria']]['Ocorrencias'].values[0])

dt_operador_fatal['Ocorrencias'] = new_ocorr
dt_operador_fatal=dt_operador_fatal.sort_values('total_fatalidades', ascending=True)
dt_operador_fatal


fig, ax = plt.subplots(figsize=(18, 8))
x = np.arange(len(dt_operador_fatal))
# Define bar width. We'll use this to offset the second bar.
bar_width = 0.4
# Note we add the `width` parameter now which sets the width of each bar.
b1 = ax.bar(x, dt_operador_fatal['Ocorrencias'],
            width=bar_width, label='Ocorrências')
# Same thing, but offset the x by the width of the bar.
b2 = ax.bar(x + bar_width,dt_operador_fatal['total_fatalidades'],
            width=bar_width, label='Fatalidades')

ax.set_xticks(x + bar_width / 2)
ax.set_xticklabels(dt_operador_fatal['aeronave_operador_categoria'])
ax.legend()
ax.set_xlabel('', labelpad=15)
ax.set_ylabel('Fatalidade/Ocorrência', labelpad=15)
ax.set_title('Fatalidade vs Ocorrência por categoria de operador', pad=15)
ax.yaxis.grid(True, color='#C0C0C0')
ax.xaxis.grid(False)
dt_pmd_fatalidade = datasetviz[['aeronave_pmd','total_fatalidades']].copy()
dt_pmd_fatalidade = dt_pmd_fatalidade[dt_pmd_fatalidade['total_fatalidades']>0]
#dt_pmd_fatalidade

x = dt_pmd_fatalidade['aeronave_pmd']
y = dt_pmd_fatalidade['total_fatalidades']
fig, ax = plt.subplots(figsize=(18, 8))
plt.scatter(x, y, label = 'Pontos',  marker = 'o')
ax.set_xlabel('Peso Máximo Decolagem', labelpad=15)
ax.set_ylabel('Fatalidade', labelpad=15)
ax.set_title('Relação do peso de decolagem com o número de fatalidades da ocorrência', pad=15)
  
plt.show()
#agrupado por fatalidade

dt_pmd_categ_fatal = datasetviz[['aeronave_pmd_categoria','total_fatalidades']].copy()
dt_pmd_categ_fatal = dt_pmd_categ_fatal.groupby(by=['aeronave_pmd_categoria'],as_index=False, sort=True)['total_fatalidades'].sum()
#dt_pmd_categ_fatal = dt_pmd_categ_fatal.sort_values('aeronave_pmd_categoria', ascending=True)
#agrupado por ocorrencias
dt_pmd_categ_freq = datasetviz['aeronave_pmd_categoria'].value_counts().reset_index().rename(columns={'index':'Categoria_PMD','aeronave_pmd_categoria':'Ocorrencias'})
dt_pmd_categ_freq = dt_pmd_categ_freq.sort_values('Categoria_PMD', ascending=True)
#adicionando novo campo de ocorrencias no dataframe
new_ocorr = []
for index,row in dt_pmd_categ_fatal.iterrows():
    new_ocorr.append(dt_pmd_categ_freq.loc[dt_pmd_categ_freq['Categoria_PMD']==row['aeronave_pmd_categoria']]['Ocorrencias'].values[0])

dt_pmd_categ_fatal['Ocorrencias'] = new_ocorr
dt_pmd_categ_fatal=dt_pmd_categ_fatal.sort_values('total_fatalidades', ascending=True)
dt_pmd_categ_fatal


fig, ax = plt.subplots(figsize=(18, 8))
x = np.arange(len(dt_pmd_categ_fatal))
# Define bar width. We'll use this to offset the second bar.
bar_width = 0.4
# Note we add the `width` parameter now which sets the width of each bar.
b1 = ax.bar(x, dt_pmd_categ_fatal['Ocorrencias'],
            width=bar_width, label='Ocorrências')
# Same thing, but offset the x by the width of the bar.
b2 = ax.bar(x + bar_width,dt_pmd_categ_fatal['total_fatalidades'],
            width=bar_width, label='Fatalidades')

ax.set_xticks(x + bar_width / 2)
ax.set_xticklabels(dt_pmd_categ_fatal['aeronave_pmd_categoria'])
ax.legend()
ax.set_xlabel('', labelpad=15)
ax.set_ylabel('Fatalidade/Ocorrência', labelpad=15)
ax.set_title('Ocorrência e Fatalidade com Base no Peso após Decolagem', pad=15)
ax.yaxis.grid(True, color='#C0C0C0')
ax.xaxis.grid(False)
#agrupado por fatalidade
dt_ano_fab_fatal = datasetviz[['aeronave_ano_fabricacao','total_fatalidades']].copy()
dt_ano_fab_fatal = dt_ano_fab_fatal.groupby(by=['aeronave_ano_fabricacao'],as_index=False, sort=True)['total_fatalidades'].sum()

#agrupado por ocorrencias
dt_ano_fab_freq = datasetviz['aeronave_ano_fabricacao'].value_counts().reset_index().rename(columns={'index':'Ano_Fabricacao','aeronave_ano_fabricacao':'Ocorrencias'})
#dt_ano_fab_freq = dt_operador_freq.sort_values('Ano_Fabricacao', ascending=True)
#adicionando novo campo de ocorrencias no dataframe
new_ocorr = []
for index,row in dt_ano_fab_fatal.iterrows():
    new_ocorr.append(dt_ano_fab_freq.loc[dt_ano_fab_freq['Ano_Fabricacao']==row['aeronave_ano_fabricacao']]['Ocorrencias'].values[0])

dt_ano_fab_fatal['Ocorrencias'] = new_ocorr
dt_ano_fab_fatal=dt_ano_fab_fatal.sort_values('aeronave_ano_fabricacao', ascending=True)
dt_ano_fab_fatal



# multiple line plot
# x = aeronave_ano_fabricacao
# y1 = Ocorrencias
# y2 = total_fatalidades
fig, ax = plt.subplots(figsize=(18, 8))
ax.plot('aeronave_ano_fabricacao', 'Ocorrencias', data=dt_ano_fab_fatal, marker='', color='blue', linewidth=2,)
ax.plot('aeronave_ano_fabricacao', 'total_fatalidades', data=dt_ano_fab_fatal, marker='', color='red', linewidth=2)
#ax.axis([1936, 2017,0, dt_ano_fab_fatal['Ocorrencias'].sum()]) # [xmin, xmax, ymin, ymax]
ax.set_title('Evolução de Fatalidades e Ocorrências conforme Ano de Fabricação', pad=15)
ax.legend()

