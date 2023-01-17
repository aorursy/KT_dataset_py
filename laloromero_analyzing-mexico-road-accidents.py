import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn import preprocessing, linear_model, cluster

import seaborn as sns

import matplotlib.pyplot as plt

import altair as alt

import holidays

import statsmodels.api as sm
df2017 = pd.read_csv('../input/mexico-road-accidents-during-2019/incidentes-viales-c5-2017.csv')

df2018 = pd.read_csv('../input/mexico-road-accidents-during-2019/incidentes-viales-c5-2018.csv')

df2019 = pd.read_csv('../input/mexico-road-accidents-during-2019/incidentes-viales-c5-2019.csv')



df = pd.concat([df2017,df2018,df2019])
df.info()
df['is_real_case'] = df.codigo_cierre.str.contains('(I)|(A)')
df = df.query("is_real_case == True")
df.hora_creacion = pd.to_datetime(df.hora_creacion)

df.fecha_creacion = pd.to_datetime(df.fecha_creacion)



df.hora_creacion = pd.to_datetime(df.hora_creacion)

df.fecha_creacion = pd.to_datetime(df.fecha_creacion)



df.año_cierre = pd.to_numeric(df.año_cierre)
print(df.delegacion_inicio.describe())

print(df.delegacion_cierre.describe())
df.at[pd.isnull(df.delegacion_inicio),'delegacion_inicio'] = df.delegacion_inicio.describe().values[2]

df.at[pd.isnull(df.delegacion_cierre),'delegacion_cierre'] = df.delegacion_cierre.describe().values[2]
le_cc = preprocessing.LabelEncoder()

le_ic = preprocessing.LabelEncoder()

le_di = preprocessing.LabelEncoder()

le_cfa = preprocessing.LabelEncoder()

le_te = preprocessing.LabelEncoder()

le_mc = preprocessing.LabelEncoder()



le_cc.fit(df.codigo_cierre.values)

le_di.fit(df.delegacion_inicio.values)

le_ic.fit(df.incidente_c4.values)

le_cfa.fit(df.clas_con_f_alarma.values)

le_te.fit(df.tipo_entrada.values)

le_mc.fit(df.mes_cierre.values)



df['codigo_cierre_encode'] = le_cc.transform(df.codigo_cierre.values)

df['delegacion_inicio_encode'] = le_di.transform(df.delegacion_inicio.values)

df['incidente_c4_encode'] = le_ic.transform(df.incidente_c4.values)

df['clas_con_f_alarma_encode'] = le_cfa.transform(df.clas_con_f_alarma.values)

df['tipo_entrada'] = le_te.transform(df.tipo_entrada.values)

df['delegacion_cierre_encode'] = le_di.transform(df.delegacion_inicio.values)

df['mes_cierre_encode'] = le_mc.transform(df.mes_cierre.values) + 1
_,ax = plt.subplots(1,1,figsize=(10,5))

# sns.lineplot(data=df.groupby('mes_cierre_encode').folio.count().reset_index(), x='mes_cierre_encode', y='folio', ax=ax[0])

sns.lineplot(data=df.groupby(['mes','año_cierre']).folio.count().reset_index().groupby('mes').folio.mean().reset_index(), x='mes', y='folio', ax=ax)

# sns.lineplot(data=df.groupby('mes_cierre_encode').folio.count().reset_index(), x='mes_cierre_encode', y='folio', ax=ax[1])
# _,ax = plt.subplots(1,2,figsize=(10,5))

df['creacion_day_name'] = df.fecha_creacion.apply(lambda x : x.dayofweek)

group_dayname_creacion = df.groupby(['creacion_day_name','año_cierre']).folio.count().reset_index().groupby('creacion_day_name').folio.mean().reset_index()



sns.lineplot(data=group_dayname_creacion.sort_values(by='creacion_day_name'), x='creacion_day_name', y='folio')
df['numhora_creacion'] = df.hora_creacion.apply(lambda x:  x.time().hour)

sns.barplot(data=df.groupby(['numhora_creacion','año_cierre']).folio.count().reset_index().groupby('numhora_creacion').folio.mean().reset_index(), 

            x='numhora_creacion', y='folio')
mx_holidays = holidays.Mexico()

df['is_holiday'] = df.fecha_creacion.apply(lambda x: mx_holidays.get(x))

df.head()
# Encoding holidays

le_ho = preprocessing.LabelEncoder()

df.loc[~df.is_holiday.isnull(),'is_holiday_encoded'] = le_ho.fit_transform(df[~df.is_holiday.isnull()].is_holiday)
sns.barplot(data=df.groupby(['is_holiday','año_cierre']).folio.count().reset_index().groupby('is_holiday').folio.mean().reset_index(),

            x='folio',y='is_holiday')
# I set freq=365 because it tells the method that it's a daily data

obj = sm.tsa.seasonal_decompose(df.groupby('fecha_creacion').folio.count(), freq=365)
_,ax = plt.subplots(2,2,figsize=(20,10), sharex=True, squeeze=False)

sns.lineplot(data=obj.observed.reset_index(), x='fecha_creacion', y='folio', ax=ax[0,0])

sns.lineplot(data=obj.resid.reset_index(), x='fecha_creacion', y='folio', ax=ax[0,1])

sns.lineplot(data=obj.seasonal.reset_index(), x='fecha_creacion', y='folio', ax=ax[1,0])

sns.lineplot(data=obj.trend.reset_index(), x='fecha_creacion', y='folio', ax=ax[1,1])



ax[0,0].set_title('Observed')

ax[0,1].set_title('Residual')

ax[1,0].set_title('Seasonal')

ax[1,1].set_title('Trend')
alt.Chart(

    df.groupby(["año_cierre","delegacion_inicio"]).folio.count().reset_index()

).mark_line().encode(

    x='año_cierre',

    y='folio',

    color='delegacion_inicio',

    shape='delegacion_inicio',

    tooltip=['delegacion_inicio','folio']

).properties(

    width=1000,

    height=400

)