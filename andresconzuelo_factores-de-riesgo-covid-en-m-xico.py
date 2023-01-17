import os
import pandas as pd
import numpy as np

files = os.listdir('../input/covid19-mx')
names = [name.split('.')[0].lower() for name in files]
for name, file in zip(names, files):
  globals()[name] = pd.read_csv('../input/covid19-mx/' + file)
general = globals()['covid-19_general_mx']
general = general.replace({'SECTOR':sector.to_dict()['DESCRIPCIÓN'],
                'SEXO':{1: 'MUJER', 2:'HOMBRE', 99: 'OTRO'},
                'TIPO_PACIENTE':tipo_paciente.to_dict()['DESCRIPCIÓN'],
                'RESULTADO': {1: True, 2:False, 3:np.nan},
                'NACIONALIDAD':nacionalidad.to_dict()['DESCRIPCIÓN']})

general = general[general['RESULTADO'] == True]
estados = ['ENTIDAD_UM', 'ENTIDAD_RES']
general[estados] = general[estados].sub(1)
ent_dict = entidades.to_dict()['ENTIDAD_FEDERATIVA']
#estado_totales.reset_index().info()
#estado_totales.reset_index().replace({'ENTIDAD_UM': ent_dict}).head()
general[estados] = general[estados].replace(ent_dict)
edad = general.EDAD
nacionalidad = general.NACIONALIDAD
for column in general.iloc[:, 9:24].columns.values.tolist():
    general.loc[:, column] = general.loc[:, column].map({1:1, 2:0,
                           97:np.nan, 98:np.nan,
                          99:np.nan})
general.EDAD = edad
general.NACIONALIDAD = nacionalidad.replace({'NO ESPECIFICADO': 'EXTRANJERA', 
                                             'EXTRANJERA': 'MEXICANA'})
general.NACIONALIDAD.value_counts()
general.loc[:, 'UCI'] = general.loc[:, 'UCI'].map({1: True, 2: False,
                               97:np.nan, 98:np.nan, 99:np.nan})
general['UCI']
general['ESTADO'] = general['FECHA_DEF'].replace({'9999-99-99': np.nan}).isna()
general.ESTADO = general.ESTADO.replace({True:'VIVO', False:'MUERTO'})
general['FECHA_DEF']= general['FECHA_DEF'].replace({'9999-99-99':np.nan})
general.FECHA_INGRESO = pd.to_datetime(general.FECHA_INGRESO)
general.FECHA_SINTOMAS = pd.to_datetime(general.FECHA_SINTOMAS)
general.FECHA_DEF = pd.to_datetime(general.FECHA_DEF)
general.EDAD.head()
#dummy_general = pd.get_dummies(general, columns =['SEXO', 'TIPO_PACIENTE'])
#dummy_general.head()
def grupo_edad(x):
    if x <=10:
        grupo = 'INFANTE'
    elif x<=20:
        grupo = 'ADOLESCENTE'
    elif x<=30:
        grupo = 'ADULTO 20s'
    elif x<=40:
        grupo = 'ADULTO 30s'
    elif x<=50:
        grupo = 'ADULTO 40s'
    elif x<=60:
        grupo = 'ADULTO 50s'
    elif x <=70:
        grupo = 'ADULTO 60s'
    else:
        grupo = 'ANCIANO'
    return grupo

general['GRUPO'] = general.EDAD.apply(grupo_edad)
import matplotlib.pyplot as plt
general['SEXO'].hist()
plt.show()
import seaborn as sns

col_hm_dict = {'HOMBRE': 'b', 'MUJER': 'r'}
sns.set(style = 'whitegrid', palette ='pastel', color_codes = True)

ax = sns.violinplot(x = 'ESTADO', y = 'EDAD',
               hue = 'SEXO', split = True,
               inner = 'quart', data = general,
               palette = col_hm_dict,
               saturation = 0.8)
#ax.set(xlabel = 'ESTATUS')

plt.show()



grupo_edad = general.groupby(['GRUPO', 'ESTADO'])['ESTADO'].count()
total_grupo = general.groupby('GRUPO')['ESTADO'].count()
total_grupo = pd.DataFrame(total_grupo).rename(columns = {'ESTADO':'TOTAL'}).reset_index()
#sobrevivencia = grupo_edad.merge(total_grupo, on = 'GRUPO')
grupo_edad = pd.DataFrame(grupo_edad).rename(columns = {'ESTADO':'CONTEO'}).reset_index()
superv_grupo = grupo_edad.merge(total_grupo, on ='GRUPO')
superv_grupo['PROPORCIONAL'] = superv_grupo.CONTEO.values/superv_grupo.TOTAL.values
superv_grupo.set_index(
    ['GRUPO', 'ESTADO']
)['PROPORCIONAL'].unstack().reset_index().sort_values('MUERTO').plot(
x = 'GRUPO', kind = 'barh', stacked = True)
plt.show()
mortalidad = superv_grupo[superv_grupo['ESTADO'] == 'MUERTO']
supervivencia = superv_grupo[superv_grupo['ESTADO'] == 'VIVO']
grupo_sexo = general.groupby(['GRUPO', 'SEXO', 'ESTADO'])['ESTADO'].count().unstack()
grupo_sexo['MORTALIDAD'] = grupo_sexo.MUERTO/(grupo_sexo.MUERTO+grupo_sexo.VIVO)
mortalidad_grupo_sexo = grupo_sexo['MORTALIDAD'].unstack().reset_index().sort_values('HOMBRE')
mortalidad_grupo_sexo.head()
mortalidad_grupo_sexo.plot(x = 'GRUPO', kind = 'barh',  width = 0.8)
plt.title('Mortalidad del CoVID por sexo \n y grupo de edad')
plt.show()
mortalidad_grupo_sexo.assign(
TOTAL = lambda x: x.HOMBRE + x.MUJER, 
HOMBRE_N = lambda x: x.HOMBRE/x.TOTAL,
MUJER_N = lambda x: x.MUJER/x.TOTAL)[['GRUPO','HOMBRE_N', 'MUJER_N']].sort_values('HOMBRE_N').plot(
    x = 'GRUPO', kind = 'barh', stacked = True, width = 1)
plt.title('Mortalidad relativa por grupo de edad')
plt.xlabel('Porcentaje de casos fatales')
plt.show()
factores = ['ESTADO', 'GRUPO', 'SEXO', 'DIABETES', 'EPOC', 'ASMA',
              'INMUSUPR', 'CARDIOVASCULAR',
             'OBESIDAD', 'RENAL_CRONICA', 
              'TABAQUISMO', 'OTRA_CON']

factores_df = general[factores]
factores_df.index.name = 'ID'
factores_df = factores_df.replace({np.nan:False})
factores_df = factores_df.assign(
    SANO = lambda x: ~x.iloc[:, 3:12].any(axis = 1))

factores_df['ANTECEDENTES'] = ~factores_df.SANO
factores_df = factores_df.replace({True:1, False:0})
factores_df.head()
def filter_drop(data, column, value):
  df = data[data[column]==value].drop(columns = column)
  return df
fact_df = factores_df.groupby(['GRUPO', 
                               'ESTADO', 
                               'SEXO']).agg(sum).reset_index()
fact_df.head()
fact_melt = fact_df.melt(id_vars=['GRUPO', 'ESTADO', 'SEXO'],
             var_name='CONDICION', value_name='CASOS')
muertos_melt = filter_drop(fact_melt, 'ESTADO', 'MUERTO')
vivos_melt = filter_drop(fact_melt, 'ESTADO', 'VIVO')
agrupaciones = ['GRUPO', 'SEXO', 'ESTADO']
sano_yn = ['SANO', 'ANTECEDENTES']
cond_list = list(['DIABETES', 'EPOC', 'ASMA', 'INMUSUPR', 
             'CARDIOVASCULAR', 'OBESIDAD', 'RENAL_CRONICA',
             'TABAQUISMO', 'OTRA_CON'])
norm_df = fact_df.groupby(['GRUPO', 'SEXO']).sum()
fact_norm =fact_df.set_index(['GRUPO', 'SEXO'])/norm_df
fact_norm = fact_norm.reset_index()
fact_norm['ESTADO'] = ['MUERTO', 'VIVO']*16
fact_norm_melt = fact_norm.melt(id_vars = ['GRUPO', 'SEXO', 'ESTADO'],
                           var_name = 'CONDICION', value_name = 'TASA')


sup_antecedentes = fact_norm_melt[fact_norm_melt['CONDICION'].isin(sano_yn)]
sup_antecedentes = filter_drop(sup_antecedentes, 'ESTADO', 'VIVO')
sup_condiciones = fact_norm_melt[fact_norm_melt['CONDICION'].isin(cond_list)]
sup_condiciones = filter_drop(sup_condiciones, 'ESTADO', 'VIVO')
sup_condiciones
def mapa_cuadrado(*args, **kwargs):
    data = kwargs.pop('data')
    sortby = kwargs.pop('sortby')
    d = data.pivot(index = args[1], columns = args[0], values = args[2]
                   ).sort_values(sortby, ascending = False)
    sns.heatmap(d*100//1 -1, **kwargs)
sexo = 'HOMBRE'
mapa_cuadrado('CONDICION', 'GRUPO', 'TASA',
              data = filter_drop(sup_antecedentes, 'SEXO', 'HOMBRE'), 
              annot = True, sortby ='ANTECEDENTES')
plt.title('Porcentaje de supervivencia en hombres \n segun grupo y antecedentes')
plt.show()
mapa_cuadrado('CONDICION', 'GRUPO', 'TASA', 
              data = filter_drop(sup_condiciones, 'SEXO', 'MUJER'),
              annot = True, sortby = 'DIABETES')
plt.title('Porcentaje de supervivencia en mujeres \n segun grupo y condicion')
plt.show()
def corr_heatmap(df, method = 'pearson', **kwargs):
  corr_df = df.corr(method = method)
  mask = np.zeros_like(corr_df)
  mask[np.triu_indices_from(mask)] = True
  sns.heatmap(corr_df, mask = mask, **kwargs)
predictores = general[cond_list]
corr_heatmap(predictores, vmax = 0.19)
pred_hosp = general[['TIPO_PACIENTE']+ cond_list]
p_hosp = filter_drop(pred_hosp, 'TIPO_PACIENTE', 'HOSPITALIZADO')
corr_heatmap(p_hosp, vmax = 0.12)
pred_muerto = general[['ESTADO']+cond_list]
p_muerto = filter_drop(pred_muerto, 'ESTADO', 'MUERTO')
corr_heatmap(p_muerto, vmax = 0.175)
sit_pacientes = ['SECTOR', 'SEXO',  'EDAD', 'ENTIDAD_UM', 
                 'ENTIDAD_RES', 'FECHA_INGRESO', 
                 'FECHA_SINTOMAS','FECHA_DEF', 
                 'INTUBADO', 'NACIONALIDAD', 'UCI', 'GRUPO', 'ESTADO']
sit_df = general[sit_pacientes]
sit_df = sit_df.assign(DIA = lambda x:
                x.FECHA_INGRESO - x.FECHA_INGRESO.min())
sit_df.DIA = sit_df.DIA.dt.days
sit_df.head()
sns.violinplot(x = 'SECTOR', y = 'EDAD', 
               hue = 'NACIONALIDAD', split = True, data = sit_df)
plt.xticks(rotation = 70)
plt.show()
def filter_index(df, column, values):
    return df[df[column].isin(values)]
estado_diarios = sit_df.groupby(['ENTIDAD_UM', 'FECHA_INGRESO'])\
[['FECHA_INGRESO', 'UCI', 'INTUBADO']].count()
estado_diarios.columns =['INGRESADOS', 'UCI', 'INTUBADOS']
estado_diarios = estado_diarios.reset_index()
top_casos =estado_diarios.groupby(['ENTIDAD_UM']).sum().nlargest(10,
                                                                 'INGRESADOS').index.values.tolist()
sns.lineplot(x = 'FECHA_INGRESO', y = 'INGRESADOS', hue = 'ENTIDAD_UM',
             data = estado_diarios[estado_diarios.ENTIDAD_UM.isin(top_casos)])
plt.legend(bbox_to_anchor = (1.05, 1), loc = 2, borderaxespad=0.)
plt.title('Evolución diaria de la pandemia \n en los estados más afectados')
plt.show()
estado_semana = estado_diarios.assign(SEMANA = lambda x: x['FECHA_INGRESO'].dt.week)\
.groupby(['ENTIDAD_UM', 'SEMANA']).sum().reset_index()
top_semanas = filter_index(estado_semana, 'ENTIDAD_UM', top_casos)
sns.lineplot(x = 'SEMANA', y = 'INGRESADOS', hue = 'ENTIDAD_UM', data = top_semanas)
plt.legend(bbox_to_anchor =(1.05, 1), loc = 2, borderaxespad = 0)
plt.title('Evolución semanal de la pandemia \n en los estados más afectados')
plt.show()