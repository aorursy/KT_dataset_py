import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 23)
pd.set_option('display.max_rows', 200)

plt.style.use('default') # haciendo los graficos un poco mas bonitos en matplotlib
#plt.rcParams['figure.figsize'] = (20, 10)

sns.set(style="whitegrid") # seteando tipo de grid en seaborn

df = pd.read_csv('../input/events/events.csv')
pd.options.mode.chained_assignment = None
df["timestamp"].min()
df["timestamp"].max()
df_visitas =  df[['timestamp','person','event']]
df_visitas['timestamp'] = pd.to_datetime(df_visitas['timestamp'])
df_visitas = df_visitas.sort_values(by=["timestamp"])
df_visitas['just_date'] = df_visitas['timestamp'].dt.date
def visited (row):
   if row['event'] == 'visited site':
      return 1
   return 0
df_visitas["visitas"] = df.apply(lambda row: visited (row),axis=1)
visitas_por_persona_por_dia = df_visitas.groupby(["person","just_date"]).agg({'visitas':'sum'})
visitas_por_persona_por_dia["visitas"]=visitas_por_persona_por_dia["visitas"].replace(0,1)
visitas_por_persona_por_dia.head(50)
visitas_por_dia = visitas_por_persona_por_dia.groupby(['just_date']).agg({'visitas':'sum'})
visitas_por_dia.head(50)
plt.figure()
g1 = visitas_por_dia.plot(kind='line', color={'darkblue'},linewidth=2,figsize=(12,8),title='Rating de visitas 2018',
                          legend=False,fontsize=12)
g1.set_xlabel("Fecha",fontsize=14)
g1.set_ylabel("Visitas",fontsize=14)
x_lab=['Ene','Feb','Mar','Abr','May','Jun']
g1.set_xticklabels(x_lab)
g1.title.set_size(20)


def converted (row):
   if row['event'] == 'conversion':
      return 1
   return 0
df_visitas["conversiones"] = df.apply(lambda row: converted (row),axis=1)

visitas_por_dia["conversiones"] = df_visitas.groupby(["just_date"]).agg({'conversiones':'sum'})
visitas_por_dia.head(50)
plt.figure()
g2 = visitas_por_dia.plot(kind='line',color={'darkblue','tomato'}, linewidth=2,figsize=(12,8),title='Proporcion de conversiones',
                          legend=True,fontsize=12)
g2.set_xlabel("Fecha",fontsize=14)
g2.set_ylabel("Visitas",fontsize=14)
x_lab=['Ene','Feb','Mar','Abr','May','Jun']
g2.set_xticklabels(x_lab)
g2.title.set_size(20)
plt.figure()
g3 = visitas_por_dia["conversiones"].plot(kind='line',color='tomato',linewidth=2,figsize=(12,8),title='Proporcion de conversiones',
                          legend=True,fontsize=12)
g3.set_xlabel("Fecha",fontsize=14)
g3.set_ylabel("Visitas",fontsize=14)
x_lab=['Ene','Feb','Mar','Abr','May','Jun']
g3.set_xticklabels(x_lab)
g3.title.set_size(20)
fig = g3.get_figure()

df_cant_visitas =visitas_por_persona_por_dia["visitas"].value_counts().to_frame()
df_cant_visitas.index
df_cant_visitas=df_cant_visitas.transpose()
df_cant_visitas
v =  [5, 6, 7, 8, 9, 10, 11, 12, 14, 13, 42, 35]
def sumar_cols(data,vector):
    columna = 0
    for i in vector:
        columna = columna + data[i]
        del data[i]
    return columna
df_cant_visitas['+5']=sumar_cols(df_cant_visitas,v)
df_cant_visitas
df_eventos_visitas = pd.merge(visitas_por_persona_por_dia,df_visitas, on=['person','just_date'], how='inner').rename(index=str, columns={"visitas_x": "visitas"})

ct = pd.crosstab(df_eventos_visitas.event, df_eventos_visitas.visitas)
ct
ct['+5']=sumar_cols(ct,v)

ct
def porcentual(data,inicio,fin,total):
    for i in range(inicio,fin):
        data[i] = (data[i]/total[i])*100
    return
porcentual(ct,1,5,df_cant_visitas.loc["visitas"])
ct['+5']=ct['+5']/df_cant_visitas.loc["visitas"]['+5']*100
ct=ct.round(1)
ct
ct=ct.reset_index()
event=ct["event"]
ct['total']=ct[1]+ct[2]+ct[3]+ct[4]+ct['+5']
ct
def normalizar_filas(data,inicio_f,fin_f,inicio_c,fin_c):
    for i in range (inicio_f,fin_f):
        for j in range(inicio_c,fin_c):
            data.transpose()[i][j]=data.transpose()[i][j]/data.transpose()[i]['total']
    return
ct=ct.rename(index=str, columns={"+5": 5})
ct.index=[0,1,2,3,4,5,6,7,8,9,10]
ct=ct.drop(["event"],axis=1)
normalizar_filas(ct,0,11,1,6)
ct.index=event
ct=ct.drop(["total"],axis=1)
ct=ct.rename(index=str, columns={5: "+5"})
ct
g4 = ct.plot(kind='bar', stacked=True,title='Eventos segun cantidad de visitas de una persona por dia', edgecolor='black', rot=25, linewidth=0.5, width=0.7, figsize=(27,18), fontsize=20)
g4.set_xlabel("Eventos",fontsize=20)
g4.title.set_size(24)
g4.legend(loc='best', prop={'size': 20})

df_visitas['just_month'] = df_visitas['timestamp'].dt.month
visitas_por_persona_por_dia=visitas_por_persona_por_dia.reset_index()
visitas_por_persona_por_dia['just_date'] = pd.to_datetime(visitas_por_persona_por_dia['just_date'])
visitas_por_persona_por_dia["just_month"] = visitas_por_persona_por_dia['just_date'].dt.month

visitas_por_persona_por_dia.head(25)
visitas_por_persona_por_mes = visitas_por_persona_por_dia.groupby(["person","just_month"]).agg({'visitas':'sum'})
visitas_por_persona_por_mes.head(25)
df_cant_visitas_mes =visitas_por_persona_por_mes["visitas"].value_counts().to_frame()
df_cant_visitas_mes=df_cant_visitas_mes.sort_index(axis=0)
df_cant_visitas_mes.index
df_cant_visitas_mes=df_cant_visitas_mes.transpose()
df_cant_visitas_mes.transpose()
v1= [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
v2= [16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,30]
v3 = [31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51, 52, 53, 54, 56] 
v4 = [62, 63, 64, 67, 68, 69, 70, 71, 73, 76, 79, 81, 85, 90, 101, 119, 144]

df_cant_visitas_mes['3-15']=sumar_cols(df_cant_visitas_mes,v1)
df_cant_visitas_mes['16-30']=sumar_cols(df_cant_visitas_mes,v2)
df_cant_visitas_mes['31-60']=sumar_cols(df_cant_visitas_mes,v3)
df_cant_visitas_mes['+61']=sumar_cols(df_cant_visitas_mes,v4)

df_cant_visitas_mes
df_eventos_visitas_por_mes = pd.merge(visitas_por_persona_por_mes,df_visitas, on=['person','just_month'], how='inner').rename(index=str, columns={"visitas_x": "visitas"})

ct_mes = pd.crosstab(df_eventos_visitas_por_mes.event, df_eventos_visitas_por_mes.visitas)
ct_mes
ct_mes['3-15']=sumar_cols(ct_mes,v1)
ct_mes['16-30']=sumar_cols(ct_mes,v2)
ct_mes['31-60']=sumar_cols(ct_mes,v3)
ct_mes['+61']=sumar_cols(ct_mes,v4)
ct_mes
ct_mes=ct_mes.rename(index=str, columns={"3-15": 3, "16-30":4,"31-60":5,"+61":6})
df_cant_visitas_mes=df_cant_visitas_mes.rename(index=str, columns={"3-15": 3, "16-30":4,"31-60":5,"+61":6})
porcentual(ct_mes,1,7,df_cant_visitas_mes.loc["visitas"])
ct_mes=ct_mes.round(1)
ct_mes
ct_mes=ct_mes.reset_index()
event=ct_mes["event"]
ct_mes['total']=ct_mes[1]+ct_mes[2]+ct_mes[3]+ct_mes[4]+ct_mes[5]+ct_mes[6]
ct_mes
ct_mes.index=[0,1,2,3,4,5,6,7,8,9,10]
ct_mes=ct_mes.drop(["event"],axis=1)
normalizar_filas(ct_mes,0,11,1,7)
ct_mes.index=event
ct_mes=ct_mes.drop(["total"],axis=1)

ct_mes=ct_mes.rename(index=str, columns={3: "3-15", 4:"16-30",5:"31-60",6:"+61"})
ct_mes
g5 = ct_mes.plot(kind='bar', stacked=True,title='Eventos segun cantidad de visitas de una persona por mes', edgecolor='black', rot=25, linewidth=0.5, width=0.7, figsize=(27,18), fontsize=20)
g5.set_xlabel("Eventos",fontsize=20)
g5.title.set_size(24)
g5.legend(loc='best', prop={'size': 20})

df_tiempos=  df[['timestamp','person']]
df_tiempos['timestamp'] = pd.to_datetime(df_tiempos['timestamp'])
df_tiempos= df_tiempos.sort_values(by=["timestamp"])
df_tiempos['just_date'] = df_tiempos['timestamp'].dt.date
df_tiempos.reset_index(drop = True, inplace = True)
df_tiempos['diff'] = df_tiempos.groupby(['person'])['timestamp'].diff()
df_tiempos['diff'] = df_tiempos['diff'].astype(str)
df_tiempos['new_user'] = df_tiempos['diff'] =='NaT'
df_tiempos['diff2'] = (df_tiempos['timestamp'] - (df_tiempos['timestamp'].shift())) / np.timedelta64(1, 'h')
df_tiempos['diff'] = df_tiempos.groupby(['person'])['timestamp'].diff()
df_tiempos=df_tiempos.fillna(0)
df_tiempos['new_session_same_user'] = df_tiempos['diff2'] > 0.48
df_tiempos["diff"]=df_tiempos["diff"]/np.timedelta64(1, 'h')
df_tiempos["new_session_new_user"] = df_tiempos["diff"] > 0.48
df_tiempos["new_session"]=df_tiempos["new_user"]|df_tiempos["new_session_same_user"]|df_tiempos["new_session_new_user"]
df_tiempos['sessionid'] = df_tiempos['new_session'].cumsum()
df_tiempos=df_tiempos[df_tiempos.new_session==False]
tiempos_sesiones=df_tiempos.groupby(["sessionid"]).agg({'diff':'sum'})
tiempos_sesiones.describe()
df_tiempos["sessionid"].max()
for i in range(86643,89071):
    tiempos_sesiones = tiempos_sesiones.append({'diff': 0}, ignore_index=True)
tiempos_sesiones.describe()
plt.figure()
g= tiempos_sesiones.reset_index().plot(kind='scatter', y='index', x='diff',alpha=0.1,figsize=(12,8),color='purple',title='Tiempo de permanencia de un usuario',legend=False,fontsize=12)


g.set_xlabel("Horas",fontsize=14)
g.set_ylabel("ID_sesion",fontsize=14)
g.title.set_size(20)

tiempos_sesiones["menor_a_30_seg"] = tiempos_sesiones["diff"] < 0.00833
tiempos_sesiones.head(10)
tiempos_sesiones["menor_a_30_seg"].value_counts()
plt.figure()
g2=tiempos_sesiones["menor_a_30_seg"].value_counts().plot(kind='bar',figsize=(10,8),rot=0,title='Tasa de rebote',legend=False,fontsize=12)

g2.set_ylabel("Cantidad de sesiones",fontsize=14)
x_lab=['> 30 seg','< 30 seg']
g2.set_xticklabels(x_lab)
g2.title.set_size(20)

df_osv = df.loc[df['operating_system_version'].notnull(), : ]
person_osv = df_osv[['person', ]]
df_osv = df_osv[['timestamp', 'operating_system_version','person']]
df_osv.head()
os = df_osv['operating_system_version'].value_counts()
s_os = os.nlargest(10)
os.describe()
os.head()
df_osv.describe()
os_per = df_osv.groupby('person')['operating_system_version'].nunique().value_counts()
os_per
person_osv.describe()
df_windows = df_osv.loc[df_osv['operating_system_version'].str.contains('Windows')]
df_ios = df_osv.loc[df_osv['operating_system_version'].str.contains('iOS')]
df_android = df_osv.loc[df_osv['operating_system_version'].str.contains('Android')]
df_mac = df_osv.loc[df_osv['operating_system_version'].str.contains('Mac')]
df_linux = df_osv.loc[df_osv['operating_system_version'].str.contains('Linux')]

df_windows['OS'] = 'Windows'
df_ios['OS'] = 'iOS'
df_android['OS'] = 'Android'
df_mac['OS'] = 'Mac'
df_linux

df_windows = df_windows[['timestamp','OS','person']]
df_ios = df_ios[['timestamp','OS','person']]
df_android = df_android[['timestamp','OS','person']]
df_mac = df_mac[['timestamp','OS','person']]




frames = [df_windows, df_ios, df_android, df_mac]
df_os = pd.concat(frames)
df_os['date'] = pd.to_datetime(df_os['timestamp'])
df_os['month'] = df_os['date'].dt.month
ct = pd.crosstab(df_os.month, df_os.OS)
ct.head(10)
ct

plt.subplots(figsize=(8,8))
grafico_dia_mes=sns.heatmap(ct,linewidths=.5,fmt="d",annot=True,cmap="BuPu")
grafico_dia_mes.set_title("Ingresos de Sistemas Operativos Por Mes",fontsize=20)
grafico_dia_mes.set_xlabel("Sistema Operativos",fontsize=12)
grafico_dia_mes.set_ylabel("Mes",fontsize=12)
df_os['OS'].value_counts()
g = sns.barplot(x=df_os['OS'].value_counts().index, y=df_os['OS'].value_counts().values, orient = 'v')
plt.xticks(rotation=90)
g.set_title("Ingresos al sitio mediante los sistemas operativos mas comunes", fontsize = 15)
g.set_xlabel("Sistema Operativo", fontsize = 12)
g.set_ylabel("Frecuencia", fontsize = 12)
df_android_full= df[df['person'].isin(df_android['person'])]
df_android_viewed = df_android_full[df_android_full['event'] == 'viewed product']
top_10_models_viewed_android = df_android_viewed['model'].value_counts().head(10)
top_10_models_viewed_android

g = sns.barplot(x=top_10_models_viewed_android.index, y=top_10_models_viewed_android.values, orient = 'v')
plt.xticks(rotation=90)
g.set_title("Los 10 teléfonos mas vistos por usuarios de Android", fontsize = 15)
g.set_xlabel("Modelo", fontsize = 12)
g.set_ylabel("Frecuencia", fontsize = 12)

df_android_conversion = df_android_full[df_android_full['event'] == 'conversion']
top_10_models_bought_android = df_android_conversion['model'].value_counts().head(10)
top_10_models_bought_android

g = sns.barplot(x=top_10_models_bought_android.index, y=top_10_models_bought_android.values, orient = 'v')
plt.xticks(rotation=90)
g.set_title("Los 10 teléfonos mas comprados por usuarios de Android", fontsize = 15)
g.set_xlabel("Modelo", fontsize = 12)
g.set_ylabel("Frecuencia", fontsize = 12)
df_windows_full= df[df['person'].isin(df_windows['person'])]
df_windows_viewed = df_windows_full[df_windows_full['event'] == 'viewed product']
top_10_models_viewed_windows = df_windows_viewed['model'].value_counts().head(10)
g = sns.barplot(x=top_10_models_viewed_windows.index, y=top_10_models_viewed_windows.values, orient = 'v')
plt.xticks(rotation=90)
g.set_title("Los 10 teléfonos mas vistos por usuarios de Windows", fontsize = 15)
g.set_xlabel("Modelo", fontsize = 12)
g.set_ylabel("Frecuencia", fontsize = 12)
df_windows_conversion = df_windows_full[df_windows_full['event'] == 'conversion']
top_10_models_bought_windows = df_windows_conversion['model'].value_counts().head(10)
top_10_models_bought_windows

g = sns.barplot(x=top_10_models_bought_windows.index, y=top_10_models_bought_windows.values, orient = 'v')
plt.xticks(rotation=90)
g.set_title("Los 10 teléfonos mas comprados por usuarios de Windows", fontsize = 15)
g.set_xlabel("Modelo", fontsize = 12)
g.set_ylabel("Frecuencia", fontsize = 12)
df_iOS_full= df[df['person'].isin(df_ios['person'])]
df_iOS_viewed = df_iOS_full[df_iOS_full['event'] == 'viewed product']
top_10_models_viewed_iOS = df_iOS_viewed['model'].value_counts().head(10)
g = sns.barplot(x=top_10_models_viewed_iOS.index, y=top_10_models_viewed_iOS.values, orient = 'v')
plt.xticks(rotation=90)
g.set_title("Los 10 teléfonos mas vistos por usuarios de iOS", fontsize = 15)
g.set_xlabel("Modelo", fontsize = 12)
g.set_ylabel("Frecuencia", fontsize = 12)
df_iOS_conversion = df_iOS_full[df_iOS_full['event'] == 'conversion']
top_10_models_bought_iOS = df_iOS_conversion['model'].value_counts().head(10)
top_10_models_bought_iOS

g = sns.barplot(x=top_10_models_bought_iOS.index, y=top_10_models_bought_iOS.values, orient = 'v')
plt.xticks(rotation=90)
g.set_title("Los 10 teléfonos mas comprados por usuarios de iOS", fontsize = 15)
g.set_xlabel("Modelo", fontsize = 12)
g.set_ylabel("Frecuencia", fontsize = 12)
## Operating System durante los meses

df_osv = df.loc[df['operating_system_version'].notnull(), : ]
person_osv = df_osv[['person', ]]
df_osv = df_osv[['timestamp', 'operating_system_version','person']]
df_osv.head()
os = df_osv['operating_system_version'].value_counts()
s_os = os.nlargest(10)
os.describe()
os.head()
df_osv.describe()

#Obtenemos todos los usuarios que tienen algun tipo OS

os_per = df_osv.groupby('person')['operating_system_version'].nunique().value_counts()
os_per

person_osv.describe()

df_windows = df_osv.loc[df_osv['operating_system_version'].str.contains('Windows')]
df_ios = df_osv.loc[df_osv['operating_system_version'].str.contains('iOS')]
df_android = df_osv.loc[df_osv['operating_system_version'].str.contains('Android')]
df_mac = df_osv.loc[df_osv['operating_system_version'].str.contains('Mac')]
df_linux = df_osv.loc[df_osv['operating_system_version'].str.contains('Linux')]

df_windows['OS'] = 'Windows'
df_ios['OS'] = 'iOS'
df_android['OS'] = 'Android'
df_mac['OS'] = 'Mac'
df_linux

df_windows = df_windows[['timestamp','OS','person']]
df_ios = df_ios[['timestamp','OS','person']]
df_android = df_android[['timestamp','OS','person']]
df_mac = df_mac[['timestamp','OS','person']]




frames = [df_windows, df_ios, df_android, df_mac]
df_os = pd.concat(frames)

#Separamos los operating system en las 4 categorias mas comunes hoy en dia (Android, Windows , iOS , OSx)

df_os['date'] = pd.to_datetime(df_os['timestamp'])
df_os['month'] = df_os['date'].dt.month
ct = pd.crosstab(df_os.month, df_os.OS)
ct.head(10)
ct
plt.subplots(figsize=(8,8))
grafico_dia_mes=sns.heatmap(ct,linewidths=.5,fmt="d",annot=True,cmap="BuPu")
grafico_dia_mes.set_title("Ingresos de Sistemas Operativos Por Mes",fontsize=20)
grafico_dia_mes.set_xlabel("Sistema Operativos",fontsize=12)
grafico_dia_mes.set_ylabel("Mes",fontsize=12)
#Obtuvimos que el gran flujo de usuarios es proveniente de usuarios de Windows y Android 

df_os['OS'].value_counts()
g = sns.barplot(x=df_os['OS'].value_counts().index, y=df_os['OS'].value_counts().values, orient = 'v')
plt.xticks(rotation=90)
g.set_title("Ingresos al sitio mediante los sistemas operativos mas comunes", fontsize = 15)
g.set_xlabel("Sistema Operativo", fontsize = 12)
g.set_ylabel("Frecuencia", fontsize = 12)
## Android

df_android_full= df[df['person'].isin(df_android['person'])]
df_android_viewed = df_android_full[df_android_full['event'] == 'viewed product']
top_10_models_viewed_android = df_android_viewed['model'].value_counts().head(10)
top_10_models_viewed_android


g = sns.barplot(x=top_10_models_viewed_android.index, y=top_10_models_viewed_android.values, orient = 'v')
plt.xticks(rotation=90)
g.set_title("Los 10 teléfonos mas vistos por usuarios de Android", fontsize = 15)
g.set_xlabel("Modelo", fontsize = 12)
g.set_ylabel("Frecuencia", fontsize = 12)
#Obtenemos los modelos mas vistos por usuarios Android


df_android_conversion = df_android_full[df_android_full['event'] == 'conversion']
top_10_models_bought_android = df_android_conversion['model'].value_counts().head(10)
top_10_models_bought_android


g = sns.barplot(x=top_10_models_bought_android.index, y=top_10_models_bought_android.values, orient = 'v')
plt.xticks(rotation=90)
g.set_title("Los 10 teléfonos mas comprados por usuarios de Android", fontsize = 15)
g.set_xlabel("Modelo", fontsize = 12)
g.set_ylabel("Frecuencia", fontsize = 12)

## Windows

df_windows_full= df[df['person'].isin(df_windows['person'])]
df_windows_viewed = df_windows_full[df_windows_full['event'] == 'viewed product']
top_10_models_viewed_windows = df_windows_viewed['model'].value_counts().head(10)

g = sns.barplot(x=top_10_models_viewed_windows.index, y=top_10_models_viewed_windows.values, orient = 'v')
plt.xticks(rotation=90)
g.set_title("Los 10 teléfonos mas vistos por usuarios de Windows", fontsize = 15)
g.set_xlabel("Modelo", fontsize = 12)
g.set_ylabel("Frecuencia", fontsize = 12)
df_windows_conversion = df_windows_full[df_windows_full['event'] == 'conversion']
top_10_models_bought_windows = df_windows_conversion['model'].value_counts().head(10)
top_10_models_bought_windows


g = sns.barplot(x=top_10_models_bought_windows.index, y=top_10_models_bought_windows.values, orient = 'v')
plt.xticks(rotation=90)
g.set_title("Los 10 teléfonos mas comprados por usuarios de Windows", fontsize = 15)
g.set_xlabel("Modelo", fontsize = 12)
g.set_ylabel("Frecuencia", fontsize = 12)

## iOS

df_iOS_full= df[df['person'].isin(df_ios['person'])]
df_iOS_viewed = df_iOS_full[df_iOS_full['event'] == 'viewed product']
top_10_models_viewed_iOS = df_iOS_viewed['model'].value_counts().head(10)

g = sns.barplot(x=top_10_models_viewed_iOS.index, y=top_10_models_viewed_iOS.values, orient = 'v')
plt.xticks(rotation=90)
g.set_title("Los 10 teléfonos mas vistos por usuarios de iOS", fontsize = 15)
g.set_xlabel("Modelo", fontsize = 12)
g.set_ylabel("Frecuencia", fontsize = 12)

df_iOS_conversion = df_iOS_full[df_iOS_full['event'] == 'conversion']
top_10_models_bought_iOS = df_iOS_conversion['model'].value_counts().head(10)
top_10_models_bought_iOS


g = sns.barplot(x=top_10_models_bought_iOS.index, y=top_10_models_bought_iOS.values, orient = 'v')
plt.xticks(rotation=90)
g.set_title("Los 10 teléfonos mas comprados por usuarios de iOS", fontsize = 15)
g.set_xlabel("Modelo", fontsize = 12)
g.set_ylabel("Frecuencia", fontsize = 12)




df_model = df.loc[df['model'].notnull()]
df_model['event'].value_counts()
df['event'].shape
df_model_checkout = df_model.loc[df_model['event'] == 'checkout']
top_checkout_models = (df_model_checkout['model'].value_counts()).nlargest(15)
df_model_checkout = df_model_checkout.loc[df_model_checkout['model'].isin(top_checkout_models.index)]
g = sns.countplot(x="model", hue="condition", data=df_model_checkout, palette="hls")
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("Condicion de top modelos elegidos en checkout", fontsize=18)
g.set_xlabel("Modelo", fontsize=12)
g.set_ylabel("Cantidad de Modelos", fontsize=18)
df_model_conversion = df_model.loc[df_model['event'] == 'conversion']
top_conversion_models = (df_model_conversion['model'].value_counts()).nlargest(15)
df_model_conversion = df_model_conversion.loc[df_model_conversion['model'].isin(top_checkout_models.index)]
g = sns.countplot(x="model", hue="condition", data=df_model_conversion, palette="hls")
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("Condicion de los 15 modelos mas comprados", fontsize=18)
g.set_xlabel("Modelo", fontsize=12)
g.set_ylabel("Cantidad de Modelos", fontsize=18)
df_model_viewed = df_model.loc[df_model['event'] == 'viewed product']
top_viewed_models = (df_model_viewed['model'].value_counts()).nlargest(15)
df_model_viewed = df_model_viewed.loc[df_model_viewed['model'].isin(top_checkout_models.index)]
g = sns.countplot(x="model", hue="condition", data=df_model_viewed, palette="hls")
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("Condicion de top modelos elegidos en viewed product", fontsize=18)
g.set_xlabel("Modelo", fontsize=12)
g.set_ylabel("Cantidad de Modelos", fontsize=18)
df_conversions = df.loc[df['event'] == 'conversion', : ]
df_conversions = df_conversions[['timestamp', 'event', 'condition', 'model']]
df_conversions.head()
df_conversions['date'] = pd.to_datetime(df_conversions['timestamp'])
df_conversions['counter'] = 1
df_conversions['month'] = df_conversions['date'].dt.month
df_conversions['day'] = df_conversions['date'].dt.weekday_name
ct = pd.crosstab(df_conversions.month, df_conversions.day)
ct.head(10)
plt.subplots(figsize=(8,8))
grafico_dia_mes=sns.heatmap(ct,linewidths=.5,fmt="d",annot=True,cmap="BuPu")
grafico_dia_mes.set_title("Sales per day",fontsize=20)
grafico_dia_mes.set_xlabel("Day",fontsize=12)
grafico_dia_mes.set_ylabel("Month",fontsize=12)
used_articles = df_conversions.groupby('model')
used_articles = used_articles['counter'].sum()
used_articles = used_articles.sort_values(ascending = False)
used_articles = used_articles.head(10)
articles_conditions = df_conversions.loc[df_conversions['condition'].isin(['Bom', 'Excelente', 'Muito Bom', 'Novo']), : ]
crosstab_aux = articles_conditions.loc[df_conversions['model'].isin(used_articles.index), :]
model_condition = pd.crosstab(crosstab_aux.condition, crosstab_aux.model)
model_condition.head()
plt.subplots(figsize=(8,8))
grafico_modelo_condicion=sns.heatmap(model_condition,linewidths=.5,fmt="d",annot=True,cmap="nipy_spectral_r")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
grafico_modelo_condicion.set_title("Condition of top 10 smartphones sold",fontsize=20)
grafico_modelo_condicion.set_xlabel("Model",fontsize=12)
grafico_modelo_condicion.set_ylabel("Condition",fontsize=12)
import geopandas
import geoplot
#GeoDataFrame global
path = geopandas.datasets.get_path('naturalearth_lowres')
df_geo = geopandas.read_file(path)

df_countries = df[['country']].dropna()
df_countries['counter'] = 1

aux = df_countries.groupby('country')
aux = aux.agg('sum')

df_geo = df_geo.rename(columns = {'name' : 'country'})

df_geo = df_geo.merge(aux, on='country', how='left')
df_geo = df_geo.fillna(0)

#Ploteamos el logaritmo de la cantidad de registros por la amplia diferencia
#entre Brasil y el resto del mundo
df_geo['counter_log'] = np.log(df_geo["counter"]+1)

graph = df_geo.plot(figsize=(16,10), column='counter_log', \
                   cmap='nipy_spectral_r', legend=True, k=100)
graph.set_title('Cantidad de registros (log) a nivel mundial', fontsize=18)
graph.set_xlabel("Longitud",fontsize=18)
graph.set_ylabel("Latitud", fontsize=18)
#Cargo el GeoDataFrame de Brasil
brazil_gdf = geopandas.read_file('../input/shapefiles/gadm36_BRA_1.shp') #gdf = GeoDataFrame

#Normalizo los str de las regiones ya que contienen tildes y otros caracteres
brazil_gdf['NAME_1'] = brazil_gdf['NAME_1'].str.normalize('NFKD')\
            .str.encode('ascii', errors='ignore').str.decode('utf-8')


df_brazil = df.loc[df['country'] == 'Brazil', ['region','country']]
df_brazil = df_brazil.loc[df_brazil['region'].notnull(), :]
df_brazil = df_brazil.loc[~df_brazil['region'].isin(['Unknown']), :]
df_brazil['region'] = df_brazil['region'].str.normalize('NFKD')\
                .str.encode('ascii',errors='ignore').str.decode('utf-8')

#Pequeña corrección para luego joinear. Federal District --> Distrito Federal
df_brazil = df_brazil.replace(to_replace='Federal District', \
                              value='Distrito Federal')
df_brazil['counter'] = df_brazil['region'].isin(brazil_gdf.NAME_1)
df_brazil = df_brazil.groupby('region').sum()
df_brazil = df_brazil.reset_index()
df_brazil = df_brazil.rename(columns={'region':'NAME_1'})
brazil_gdf = brazil_gdf.merge(df_brazil, on='NAME_1', how='left')

#Debido a la gran diferencia de registros de Sao Paulo con el resto, ploteamos log
brazil_gdf['counter_log'] = np.log(brazil_gdf["counter"]+1)

graph = brazil_gdf.plot(column='counter_log',edgecolor='black', legend=True, \
                 cmap='BuPu', figsize=(16,10))

graph.set_title('Cantidad de registros (log) por región en Brasil', fontsize=18)
graph.set_xlabel("Longitud",fontsize=18)
graph.set_ylabel("Latitud", fontsize=18)
usa_gdf = geopandas.read_file('../input/shapefiles/cb_2017_us_state_5m.shp')

df_just_usa = df.loc[df['country'] == 'United States', ['region', 'country']]
df_just_usa['counter'] = 1
df_just_usa = df_just_usa.groupby('region').agg('sum')
df_just_usa = df_just_usa.reset_index()
df_just_usa = df_just_usa.rename(columns = {'region' : 'NAME'})
usa_gdf = usa_gdf.merge(df_just_usa,on='NAME', how='left')
usa_gdf = usa_gdf.fillna(0)

#Quitamos algunos sectores del mapa, como Hawaii o Alaska, para que el plot
#sea mas prolijo. Ninguno de estos sectores aparece en el dataset
usa_gdf = usa_gdf[usa_gdf.STATEFP.astype(int) < 60]
usa_gdf = usa_gdf[~usa_gdf.NAME.isin(['Hawaii','Alaska'])]


graph = usa_gdf.plot(column='counter',edgecolor='black', legend=True,\
                 cmap='nipy_spectral_r', figsize=(16,10))
graph.set_title('Cantidad de registros por región en Estados Unidos',\
                fontsize=18)
graph.set_xlabel("Longitud",fontsize=18)
graph.set_ylabel("Latitud", fontsize=18)
user_info = df.loc[df['event'] == 'visited site', ['event','person','city',\
                   'region', 'country', 'device_type', 'operating_system_version',\
                   'browser_version']]

df_mobile = user_info.loc[~user_info['device_type'].str.contains('Computer'), :]
df_not_mobile = user_info.loc[user_info['device_type'].str.contains('Computer'), :]


df_mobile['device'] = 'Mobile'
df_mobile = df_mobile[['person','device']]
df_not_mobile['device'] = 'Not Mobile'
df_not_mobile = df_not_mobile[['person','device']]

df_user_device = df_mobile.merge(df_not_mobile, how='outer')
df_user_device = df_user_device.drop_duplicates()

user_info = user_info.merge(df_user_device, how='outer')
user_info = user_info.drop_duplicates()

device_percentage = (user_info.groupby(['country'])['device']
                     .value_counts(normalize=True)
                     .rename('percentage')
                     .mul(100)
                     .reset_index()
                     .sort_values('country'))
device_percentage = device_percentage.loc[device_percentage['percentage'] != 100.0, :]
plt.figure(figsize=(16,10))
g = sns.barplot(x="country", y='percentage', hue="device", data=device_percentage,\
                palette="hls")
g.set_title("Porcentaje de dispositivos utilizados por pais", fontsize=18)
g.set_xlabel("Pais", fontsize=18)
g.set_ylabel("Porcentaje", fontsize=18)
plt.xticks(rotation=90)
#Informacion de Pais y región de cada usuario
users_locale = df[['person','country','region']]
users_locale = users_locale.dropna().drop_duplicates('person')

users_models = df.loc[df['person'].isin(users_locale['person']), ['person','model','event']]
users_models = users_models.dropna()
users_models = users_models.loc[users_models['event'] == 'viewed product',:]
users_models = users_models.merge(users_locale, on='person', how='left')
brazil = users_models.loc[users_models['country'] == 'Brazil', :]
top_models = brazil['model'].value_counts().head(5).index
brazil = brazil.loc[brazil['model'].isin(top_models), :]
brazil = brazil.loc[brazil['region'] != 'Unknown', :]

plt.figure(figsize=(16,10))
g = sns.countplot(x="region", hue="model", data=brazil, palette="hls")
g.set_title("Top 5 viewed products by region in Brazil", fontsize=18)
g.set_xlabel("Region", fontsize=18)
g.set_ylabel("Frecuency", fontsize=18)
plt.xticks(rotation=90)
usa = users_models.loc[users_models['country'] == 'United States', :]
top_models = usa['model'].value_counts().head(5).index
usa = usa.loc[usa['model'].isin(top_models), :]
usa = usa.loc[usa['region'] != 'Unknown', :]

plt.figure(figsize=(16,10))
g = sns.countplot(x="region", hue="model", data=usa, palette="hls")
g.set_title("Top 5 viewed products by region in United States", fontsize=18)
g.set_xlabel("Region", fontsize=18)
g.set_ylabel("Frecuency", fontsize=18)
plt.xticks(rotation=90)
arg = users_models.loc[users_models['country'] == 'Argentina', :]
top_models = arg['model'].value_counts().head(5).index
arg = arg.loc[arg['model'].isin(top_models), :]
arg = arg.loc[arg['region'] != 'Unknown', :]

plt.figure(figsize=(16,10))
g = sns.countplot(x="region", hue="model", data=arg, palette="hls")
g.set_title("Top 5 viewed products by region in United States", fontsize=18)
g.set_xlabel("Region", fontsize=18)
g.set_ylabel("Frecuency", fontsize=18)
plt.xticks(rotation=90)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline


pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 23)
pd.set_option('display.max_rows', 100)

pd.options.mode.chained_assignment = None

plt.style.use('default') # haciendo los graficos un poco mas bonitos en matplotlib
#plt.rcParams['figure.figsize'] = (20, 10)

sns.set(style="whitegrid") # seteando tipo de grid en seaborn


df_model = df[df.model.notnull()][['person', 'model']]
model_persons = df_model['person']                #  SERIE CON PERSONAS QEUE ESTA
df_model_users = df.loc[df['person'].isin(model_persons), :]
df_model_users_buys = df_model_users.loc[df['event'] == 'conversion']
models_counts = df_model_users_buys['model'].value_counts().head(8)
models_counts
sales_Samsung = sum(df_model_users_buys['model'].loc[df_model_users_buys['model'].str.contains('Samsung')].value_counts())
sales_iPhone = sum(df_model_users_buys['model'].loc[df_model_users_buys['model'].str.contains('iPhone')].value_counts())
sales_Motorola = sum(df_model_users_buys['model'].loc[df_model_users_buys['model'].str.contains('Motorola')].value_counts())
sales_LG = sum(df_model_users_buys['model'].loc[df_model_users_buys['model'].str.contains('LG')].value_counts())
sales_Sony = sum(df_model_users_buys['model'].loc[df_model_users_buys['model'].str.contains('Sony')].value_counts())
sales_Asus = sum(df_model_users_buys['model'].loc[df_model_users_buys['model'].str.contains('Asus')].value_counts())
sales_Lenovo = sum(df_model_users_buys['model'].loc[df_model_users_buys['model'].str.contains('Lenovo')].value_counts())

df_brand_sales = pd.DataFrame({'brand': ['Samsung', 'iPhone', 'Motorola', 'LG', 'Sony', 'Asus', 'Lenovo'],
                   'sales': [sales_Samsung, sales_iPhone, sales_Motorola, sales_LG, sales_Sony, sales_Asus, sales_Lenovo]})
df_brand_sales.sort_values('sales', ascending=False)

sales_Samsung = df_model_users_buys['model'].loc[df_model_users_buys['model'].str.contains('Samsung')].value_counts().head(15)

g2 = sns.barplot(x=sales_Samsung.index, y=sales_Samsung.values, orient = 'v')
plt.xticks(rotation=90)
g2.set_title("Samsung model sales", fontsize = 15)
g2.set_xlabel("Model", fontsize = 12)
g2.set_ylabel("Frequency", fontsize = 12)
sales_iPhone = df_model_users_buys['model'].loc[df_model_users_buys['model'].str.contains('iPhone')].value_counts().head(15)

g2 = sns.barplot(x=sales_iPhone.index, y=sales_iPhone.values, orient = 'v')
plt.xticks(rotation=90)
g2.set_title("iPhone model sales", fontsize = 15)
g2.set_xlabel("Model", fontsize = 12)
g2.set_ylabel("Frequency", fontsize = 12)
sales_Motorola = df_model_users_buys['model'].loc[df_model_users_buys['model'].str.contains('Motorola')].value_counts().head(15)

g2 = sns.barplot(x=sales_Motorola.index, y=sales_Motorola.values, orient = 'v')
plt.xticks(rotation=90)
g2.set_title("Motorola model sales", fontsize = 15)
g2.set_xlabel("Model", fontsize = 12)
g2.set_ylabel("Frequency", fontsize = 12)
# Viewed product. Análisis por país
#Informacion de Pais y región de cada usuario
users_locale = df[['person','country','region']]
users_locale = users_locale.dropna().drop_duplicates('person')

users_models = df.loc[df['person'].isin(users_locale['person']), ['person','model','event']]
users_models = users_models.dropna()
users_models = users_models.loc[users_models['event'] == 'viewed product',:]
users_models = users_models.merge(users_locale, on='person', how='left')

## Brasil

brazil = users_models.loc[users_models['country'] == 'Brazil', :]
top_models = brazil['model'].value_counts().head(5).index
brazil = brazil.loc[brazil['model'].isin(top_models), :]
brazil = brazil.loc[brazil['region'] != 'Unknown', :]

plt.figure(figsize=(16,10))
g = sns.countplot(x="region", hue="model", data=brazil, palette="hls")
g.set_title("Top 5 viewed products by region in Brazil", fontsize=18)
g.set_xlabel("Region", fontsize=18)
g.set_ylabel("Frecuency", fontsize=18)
plt.xticks(rotation=90)

## Estados Unidos

usa = users_models.loc[users_models['country'] == 'United States', :]
top_models = usa['model'].value_counts().head(5).index
usa = usa.loc[usa['model'].isin(top_models), :]
usa = usa.loc[usa['region'] != 'Unknown', :]

plt.figure(figsize=(16,10))
g = sns.countplot(x="region", hue="model", data=usa, palette="hls")
g.set_title("Top 5 viewed products by region in United States", fontsize=18)
g.set_xlabel("Region", fontsize=18)
g.set_ylabel("Frecuency", fontsize=18)
plt.xticks(rotation=90)

## Argentina

arg = users_models.loc[users_models['country'] == 'Argentina', :]
top_models = arg['model'].value_counts().head(5).index
arg = arg.loc[arg['model'].isin(top_models), :]
arg = arg.loc[arg['region'] != 'Unknown', :]

plt.figure(figsize=(16,10))
g = sns.countplot(x="region", hue="model", data=arg, palette="hls")
g.set_title("Top 5 viewed products by region in United States", fontsize=18)
g.set_xlabel("Region", fontsize=18)
g.set_ylabel("Frecuency", fontsize=18)
plt.xticks(rotation=90)
lead = (df.loc[df['event'] == 'lead'])['person']
df_lead_users = df.loc[df.person.isin(lead)]
df_lead_users.describe()
event_counts_lead = df_lead_users.event.value_counts()
checkout_vs_conversion = event_counts_lead.drop(labels=['ad campaign hit', 'viewed product','staticpage','lead', 'brand listing'])
event_counts_lead
conversion_person_lead =(df_lead_users.loc[df_lead_users['event'] == 'conversion'])['person'].drop_duplicates()
df_lead_users_conversion = df_lead_users.loc[df_lead_users['person'].isin(conversion_person_lead)]
event_counts_lead_conversion= df_lead_users_conversion.event.value_counts()
event_counts_lead= df_lead_users.event.value_counts()
checkout_vs_conversion_lead = event_counts_lead_conversion.drop(labels=['ad campaign hit', 'viewed product','staticpage','lead', 'brand listing'])
event_counts_lead

conversion_person_lead.describe()
data_from_lead = pd.DataFrame({"Personas que usaron el evento lead":[291], "Personas que pidieron stock y compraron":[58], "Compras realizadas totales":[124]})
data_from_lead
g = sns.barplot(x=data_from_lead.iloc[0], y=data_from_lead.columns, orient='h')
plt.xticks(rotation=90)
g.set_title("Analisis del feature lead", fontsize=15)
g.set_xlabel("Cantidad", fontsize=12)

df_lead_users_conversions = df_lead_users.loc[df_lead_users['event']== 'conversion']


df_lead_models = df_lead_users.loc[df_lead_users['person'].isin(df_lead_users_conversions['person'])]
df_lead_models = df_lead_models.loc[df_lead_models['event'] == 'lead']


values1 = df_lead_models['model'].value_counts().head(15)
values2 =df_lead_users_conversions['model'].value_counts().head(15)
ind = values1.index
ind2 = values2.index
g = sns.barplot(x=ind, y=values1, orient='v')
plt.xticks(rotation=90)
g.set_title("Modelos pedidos en evento lead", fontsize=15)
g.set_xlabel("Modelo", fontsize=12)
g.set_ylabel("Cantidad", fontsize=12)

g = sns.barplot(x=ind2, y=values2, orient='v')
plt.xticks(rotation=90)
g.set_title("Compras de los usuarios lead", fontsize=15)
g.set_xlabel("Modelo", fontsize=12)
g.set_ylabel("Cantidad", fontsize=12)
df_conversions = df.loc[df['event'] == 'conversion', : ]
df_conversions = df_conversions[['timestamp', 'event', 'condition', 'model']]
df_conversions.head()
model_most_sold = df_conversions['model'].value_counts().head(20)
plt.figure(figsize=(16,10))
g = sns.barplot(x=model_most_sold.index, y=model_most_sold.values, orient='v')
plt.xticks(rotation=90, fontsize = 12)
    
g.set_title("Productos más vendidos a nivel mundial", fontsize=18)
g.set_xlabel("Productos", fontsize=14)
g.set_ylabel("Cantidad", fontsize=14)
model_most_sold
df_campaign_entries = df.loc[df['campaign_source'].notnull()]
df_campaign_person = df_campaign_entries.drop_duplicates('person')
df_campaign_person.describe()
df_campaign = df.loc[df['campaign_source'].notnull()]
df_campaign.describe()
campaign_users = df_campaign['person'].drop_duplicates()
campaign_users.describe()
df_conversion_campaign = df.loc[df['person'].isin(campaign_users)]
df_conversion_campaign = df_conversion_campaign.loc[df_conversion_campaign['event'] == 'conversion']
conversion_person = df_conversion_campaign['person'].drop_duplicates()
df_conversion_campaign.describe()
df_conversion_campaign['timestamp'] = pd.to_datetime(df_conversion_campaign['timestamp'])
df_conversion_campaign['month'] = df_conversion_campaign['timestamp'].dt.month
df_conversion_campaign['count']=1
df_conversion_month = df_conversion_campaign.groupby('month').agg({'count':'sum'}).reset_index()

df_conversion_month.columns = ['month','conversions']
df_campaign['timestamp'] = pd.to_datetime(df_campaign['timestamp'])
df_campaign['month'] = df_campaign['timestamp'].dt.month
df_campaign['count'] = 1
df_campaign_month = df_campaign.groupby('month').agg({'count':'sum'}).reset_index()
df_campaign_month.columns = ['month','people of campaign']
month_conversions_people = pd.merge(df_campaign_month, df_conversion_month, on='month', how='inner')
month_conversions_people
g =sns.barplot(x="month", y="people of campaign", data=month_conversions_people)
g.set_xticklabels(g.get_xticklabels())
g.set_xlabel("Mes", fontsize=15)
g.set_ylabel("Cantidad de gente", fontsize=15)
g.set_title("Ingreso de personas involucradas en campaña al sitio", fontsize=15)

g =sns.barplot(x="month", y="conversions", data=month_conversions_people)
g.set_xlabel("Mes", fontsize=15)
g.set_ylabel("Ventas", fontsize=15)
g.set_title("Compras de personas involucradas en campaña al sitio", fontsize=15)

df_campaign_entries = df.loc[df['campaign_source'].notnull()]
df_campaign_person = df_campaign_entries.drop_duplicates('person')
df_campaign_person.describe()

#La cantidad de gente que entro a la pagina por campaña por lo menos una vez es 21306

#De los entries que son de campaign_source tenemos 82796

df_campaign = df.loc[df['campaign_source'].notnull()]
df_campaign.describe()

#Actividad de usuarios relacionados con alguna campaña

campaign_users = df_campaign['person'].drop_duplicates()
campaign_users.describe()

df_conversion_campaign = df.loc[df['person'].isin(campaign_users)]
df_conversion_campaign = df_conversion_campaign.loc[df_conversion_campaign['event'] == 'conversion']
conversion_person = df_conversion_campaign['person'].drop_duplicates()
df_conversion_campaign.describe()

#Compra de usuarios que tuvieron relacion con alguna campaña

df_conversion_campaign['timestamp'] = pd.to_datetime(df_conversion_campaign['timestamp'])
df_conversion_campaign['month'] = df_conversion_campaign['timestamp'].dt.month
df_conversion_campaign['count']=1
df_conversion_month = df_conversion_campaign.groupby('month').agg({'count':'sum'}).reset_index()

df_conversion_month.columns = ['month','conversions']

#Estas son las compras en la pagina dado los meses por usuarios que alguna vez entraron a la pagina via campaña

df_campaign['timestamp'] = pd.to_datetime(df_campaign['timestamp'])
df_campaign['month'] = df_campaign['timestamp'].dt.month
df_campaign['count'] = 1
df_campaign_month = df_campaign.groupby('month').agg({'count':'sum'}).reset_index()
df_campaign_month.columns = ['month','people of campaign']

month_conversions_people = pd.merge(df_campaign_month, df_conversion_month, on='month', how='inner')
month_conversions_people

g =sns.barplot(x="month", y="people of campaign", data=month_conversions_people)
g.set_xticklabels(g.get_xticklabels())
g.set_xlabel("Mes", fontsize=15)
g.set_ylabel("Cantidad de gente", fontsize=15)
g.set_title("Ingreso de personas involucradas en campaña al sitio", fontsize=15)


#Viendo este grafico diriamos que los meses con mayores ventas serian 5 y 6 ya que tiene mas eventos en la pagina

g =sns.barplot(x="month", y="conversions", data=month_conversions_people)
g.set_xlabel("Mes", fontsize=15)
g.set_ylabel("Ventas", fontsize=15)
g.set_title("Compras de personas involucradas en campaña al sitio", fontsize=15)


#Podemos ver que este no fue el caso ya que el mes 4 fue el que consigio mas ventas