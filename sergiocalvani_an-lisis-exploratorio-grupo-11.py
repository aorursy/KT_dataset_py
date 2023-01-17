# Alumnos:
#Bravo Arroyo, Víctor Manuel - 98882
#Calvani, Sergio Alejandro - 98588
#Montes, Gastón - 89397
#Pérez Ondarts, Flavio - 96786
#Link de GitHub: https://github.com/GastonMontes/75.06-Datos-TP1
#Importamos las librerias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as datetime
import sys

%run double-pendulum.py
pd.options.mode.chained_assignment = None
%matplotlib inline
plt.style.use('default')
sns.set(style="whitegrid")
#Cargamos los datos en dataframes
educacion=pd.read_csv('../input/fiuba_1_postulantes_educacion.csv')
genero_edad=pd.read_csv('../input/fiuba_2_postulantes_genero_y_edad.csv')
vistas=pd.read_csv('../input/fiuba_3_vistas.csv')
postulaciones=pd.read_csv('../input/fiuba_4_postulaciones.csv')
avisos_online=pd.read_csv('../input/fiuba_5_avisos_online.csv')
avisos_detalle=pd.read_csv('../input/fiuba_6_avisos_detalle.csv')
#Verificamos si hay algún elemento nulo
#En el primer archivo no hay elementos nulos
educacion.info()
#En 'fechanacimiento' hay elementos nulos
genero_edad.info()
#En vistas no hay elementos nulos
vistas.info()
#En postulaciones no hay elementos nulos
postulaciones.info()
#En este archivo no hay elementos nulos
avisos_online.info()
#Hay elementos nulos en 'ciudad' , 'mapacalle' y 'denominacion_empresa'
avisos_detalle.info()
#Cambiamos los nombres de las columnas para que sea más sencillo identificarlas
educacion.rename(columns={'nombre':'Nivel','estado':'Estado'},inplace=True)
genero_edad.rename(columns={'fechanacimiento':'Fecha_Nacimiento','sexo':'Sexo'},inplace=True)
vistas.rename(columns={'idAviso':'idaviso'},inplace=True)
postulaciones.rename(columns={'fechapostulacion':'Fecha_Postulacion'},inplace=True)
avisos_detalle.rename(columns={'titulo':'Titulo','descripcion':'Descripcion','nombre_zona':'Zona','ciudad':'Ciudad','tipo_de_trabajo':'Tipo_de_Trabajo','nivel_laboral':'Nivel_Laboral','nombre_area':'Nombre_Area'},inplace=True)
#Realizamos la convesion de las fechas
postulaciones['Fecha_Postulacion']=pd.to_datetime(postulaciones['Fecha_Postulacion'])
vistas['timestamp']=pd.to_datetime(vistas['timestamp'])
genero_edad['Fecha_Nacimiento'] = pd.to_datetime(genero_edad['Fecha_Nacimiento'],errors='coerce')
#Calculamos cuales son las areas de trabajo con mayor cantidad de avisos publicados
nombre_area=avisos_detalle[['idaviso','Nombre_Area']]
top_avisos=nombre_area['Nombre_Area'].value_counts().head(10)
plt.subplots(figsize=(8,8))
grafico_top_avisos=sns.barplot(x=top_avisos.values,y=top_avisos.index,orient='h',palette="magma")
grafico_top_avisos.set_title("Areas con mayor cantidad de Avisos",fontsize=20)
grafico_top_avisos.set_xlabel("Cantidad de Avisos",fontsize=12)
grafico_top_avisos.set_ylabel("Areas de Trabajo",fontsize=12)
postulaciones_area=pd.merge(postulaciones,nombre_area,on='idaviso',how='inner')
vistas_area=pd.merge(vistas,nombre_area,on='idaviso',how='inner')
top_postulaciones=postulaciones_area['Nombre_Area'].value_counts().head(10)
plt.subplots(figsize=(8,8))
grafico_top_postulaciones=sns.barplot(x=top_postulaciones.values,y=top_postulaciones.index,orient='h')
grafico_top_postulaciones.set_title("Areas con mas Postulaciones",fontsize=20)
grafico_top_postulaciones.set_xlabel("Cantidad de Postulaciones",fontsize=12)
grafico_top_postulaciones.set_ylabel("Areas de Trabajo",fontsize=12)
df_post=top_postulaciones.reset_index()
df_top_post=df_post[['index']]
df_top_post.rename(columns={'index':'Nombre_Area'},inplace=True)
comp_post_vis=top_postulaciones.reset_index()
comp_post_vis.rename(columns={'index':'Nombre_Area','Nombre_Area':'Postulaciones'},inplace=True)
top_vistas=vistas_area['Nombre_Area'].value_counts().head(10)
df_top_vistas=top_vistas.reset_index()
comp_post_vis['Vistas']=df_top_vistas['Nombre_Area']
comp_post_vis
grafico_comp_post_vista=comp_post_vis.plot(kind='bar',x='Nombre_Area',fontsize=12,figsize=(8,8),rot=70)
grafico_comp_post_vista.set_title("Cantidad de Postulaciones y Visitas de Principales Area",fontsize=20)
grafico_comp_post_vista.set_xlabel("Area de Trabajo",fontsize=12)
grafico_comp_post_vista.set_ylabel("Cantidad de Postulaciones/Visitas",fontsize=12)
leyenda=plt.legend(['Postulaciones','Vistas'],fontsize=12,title='Tipo',frameon=True,facecolor='white',edgecolor='black')
leyenda.get_frame().set_linewidth(1.0)
genero=genero_edad[['idpostulante','Sexo']]
genero_femenino=genero[(genero_edad['Sexo']=='FEM')]
genero_masculino=genero[(genero_edad['Sexo']=='MASC')]
#Agregamos a las postulaciones y a las vistas el género del usuario
postulantes_femenino=pd.merge(postulaciones_area,genero_femenino,on='idpostulante',how='inner')
postulantes_masculino=pd.merge(postulaciones_area,genero_masculino,on='idpostulante',how='inner')
vistas_femenino=pd.merge(vistas_area,genero_femenino,on='idpostulante',how='inner')
vistas_masculino=pd.merge(vistas_area,genero_masculino,on='idpostulante',how='inner')
#Filtramos en postulaciones y vistas por genero, las áreas de trabajo más importantes
top_post_fem=pd.merge(df_top_post,postulantes_femenino,on='Nombre_Area',how='inner')
top_post_masc=pd.merge(df_top_post,postulantes_masculino,on='Nombre_Area',how='inner')
top_vistas_fem=pd.merge(df_top_post,vistas_femenino,on='Nombre_Area',how='inner')
top_vistas_masc=pd.merge(df_top_post,vistas_masculino,on='Nombre_Area',how='inner')
#Vemos la cantidad de postulaciones por genero que hay en las áreas más importantes
post_genero=top_post_masc.groupby(['Nombre_Area']).agg({'idpostulante':'count'})
post_fem=top_post_fem.groupby(['Nombre_Area']).agg({'idpostulante':'count'})
post_genero['Femenino']=post_fem['idpostulante']
post_genero.rename(columns={'idpostulante':'Masculino'},inplace=True)
post_genero
grafico_post_genero=post_genero.plot(kind='bar',color=['royalblue','salmon'],fontsize=12,figsize=(8,8),rot=70)
grafico_post_genero.set_title("Cantidad de Postulaciones por genero en Areas principales",fontsize=20)
grafico_post_genero.set_xlabel("Area de Trabajo",fontsize=12)
grafico_post_genero.set_ylabel("Cantidad de Postulaciones",fontsize=12)
leyenda=plt.legend(['Masculino','Femenino'],fontsize=12,title='Genero',frameon=True,facecolor='white',edgecolor='black')
leyenda.get_frame().set_linewidth(1.0)
vistas_genero=top_vistas_masc.groupby(['Nombre_Area']).agg({'idpostulante':'count'})
vistas_fem=top_vistas_fem.groupby(['Nombre_Area']).agg({'idpostulante':'count'})
vistas_genero['Femenino']=vistas_fem['idpostulante']
vistas_genero.rename(columns={'idpostulante':'Masculino'},inplace=True)
vistas_genero
grafico_vistas_genero=vistas_genero.plot(kind='bar',color=['royalblue','salmon'],figsize=(8,8),fontsize=12,rot=70)
grafico_vistas_genero.set_title("Cantidad de Vistas por genero en Areas principales",fontsize=20)
grafico_vistas_genero.set_xlabel("Area de Trabajo",fontsize=12)
grafico_vistas_genero.set_ylabel("Cantidad de Vistas",fontsize=12)
leyenda=plt.legend(['Masculino','Femenino'],fontsize=12,title='Genero',frameon=True,facecolor='white',edgecolor='black')
leyenda.get_frame().set_linewidth(1.0)
#Primero calculamos la edad de cada postulante
#La fecha actual es la del día 12-4-2018
fechaActual=pd.Timestamp(datetime.datetime.now())
genero_edad['Edad']=(fechaActual - genero_edad['Fecha_Nacimiento']).astype('<m8[Y]')
genero_declarado=genero_edad[(genero_edad['Sexo'] != 'NO_DECLARA')]
genero_declarado_filtrado=genero_declarado[(genero_declarado['Edad']>=18) & (genero_declarado['Edad']<=66)]
edad=genero_declarado_filtrado[['idpostulante','Sexo','Edad']]
postulaciones_edad=pd.merge(postulaciones_area,edad,on='idpostulante',how='inner')
top_postulaciones_edad=pd.merge(postulaciones_edad,df_top_post,on='Nombre_Area',how='inner')
edad_post_general=top_postulaciones_edad.groupby(['Nombre_Area']).agg({'Edad':'mean'})
edad_post_general
plt.subplots(figsize=(8,8))
grafico_edad_post_general=sns.barplot(y=edad_post_general.index,x=edad_post_general['Edad'],orient='h',palette="rocket")
grafico_edad_post_general.set_title("Edad promedio de Postulantes en Areas Principales",fontsize=20)
grafico_edad_post_general.set_ylabel("Área de Trabajo",fontsize=12)
grafico_edad_post_general.set_xlabel("Edad",fontsize=12)
plt.xlim([0,31])
top_postulaciones_edad_fem=top_postulaciones_edad[(top_postulaciones_edad['Sexo']=='FEM')]
edad_post_fem=top_postulaciones_edad_fem.groupby(['Nombre_Area']).agg({'Edad':'mean'})
edad_post_fem
plt.subplots(figsize=(8,8))
grafico_edad_post_fem=sns.barplot(y=edad_post_fem.index,x=edad_post_fem['Edad'],orient='h',palette="Dark2_r")
grafico_edad_post_fem.set_title("Edad promedio de Postulantes en Areas Principales (Femenino)",fontsize=20)
grafico_edad_post_fem.set_xlabel("Edad",fontsize=12)
grafico_edad_post_fem.set_ylabel("Área de Trabajo",fontsize=12)
plt.xlim([0,31])
top_postulaciones_edad_masc=top_postulaciones_edad[(top_postulaciones_edad['Sexo']=='MASC')]
edad_post_masc=top_postulaciones_edad_masc.groupby(['Nombre_Area']).agg({'Edad':'mean'})
edad_post_masc
plt.subplots(figsize=(8,8))
grafico_edad_post_masc=sns.barplot(y=edad_post_masc.index,x=edad_post_masc['Edad'],orient='h',palette="nipy_spectral_r")
grafico_edad_post_masc.set_title("Edad promedio de Postulantes en Areas Principales (Masculino)",fontsize=20)
grafico_edad_post_masc.set_xlabel("Edad",fontsize=12)
grafico_edad_post_masc.set_ylabel("Área de Trabajo",fontsize=12)
plt.xlim([0,33])
en_curso=educacion[(educacion['Estado']=='En Curso')]
postulaciones_en_curso=pd.merge(postulaciones_area,en_curso,on='idpostulante',how='inner')
top_postulaciones_en_curso=pd.merge(postulaciones_en_curso,df_top_post,on='Nombre_Area',how='inner')
top_postulaciones_en_curso['Valor']=1
nivel_top_postulaciones_en_curso=top_postulaciones_en_curso.pivot_table(index='Nombre_Area',columns='Nivel',values='Valor',aggfunc='count')
nivel_top_postulaciones_en_curso
plt.subplots(figsize=(8,8))
grafico_postulaciones_en_curso=sns.heatmap(nivel_top_postulaciones_en_curso,linewidths=.5,fmt="d",annot=True,cmap="nipy_spectral_r")
grafico_postulaciones_en_curso.set_title("Cantidad de Postulaciones Estado Cursando segun Area de Trabajo",fontsize=20)
grafico_postulaciones_en_curso.set_xlabel("Nivel de Estudio",fontsize=12)
grafico_postulaciones_en_curso.set_ylabel("Area de Trabajo",fontsize=12)
grafico_postulaciones_en_curso.set_xticklabels(grafico_postulaciones_en_curso.get_xticklabels(),rotation=60)
graduados=educacion[(educacion['Estado']=='Graduado')]
postulaciones_graduados=pd.merge(postulaciones_area,graduados,on='idpostulante',how='inner')
top_postulaciones_graduados=pd.merge(postulaciones_graduados,df_top_post,on='Nombre_Area',how='inner')
top_postulaciones_graduados['Valor']=1
nivel_top_postulaciones_graduados=top_postulaciones_graduados.pivot_table(index='Nombre_Area',columns='Nivel',values='Valor',aggfunc='count')
nivel_top_postulaciones_graduados
plt.subplots(figsize=(8,8))
grafico_postulaciones_graduados=sns.heatmap(nivel_top_postulaciones_graduados,linewidths=.5,fmt="d",annot=True,cmap="nipy_spectral_r")
grafico_postulaciones_graduados.set_title("Cantidad de Postulaciones Estado Graduados segun Area de Trabajo",fontsize=20)
grafico_postulaciones_graduados.set_xlabel("Nivel de Estudio",fontsize=12)
grafico_postulaciones_graduados.set_ylabel("Area de Trabajo",fontsize=12)
grafico_postulaciones_graduados.set_xticklabels(grafico_postulaciones_graduados.get_xticklabels(),rotation=60)
segunda_enero=postulaciones[(postulaciones['Fecha_Postulacion']>='2018-01-15') & (postulaciones['Fecha_Postulacion']<='2018-01-31')]
segunda_enero['Dia']=segunda_enero['Fecha_Postulacion'].dt.day
quincenas=segunda_enero['Dia'].value_counts().sort_index().reset_index()
quincenas.replace({15:1,16:2,17:3,18:4,19:5,20:6,21:7,22:8,23:9,24:10,25:11,26:12,27:13,28:14,29:15,30:16},inplace=True)
quincenas.rename(columns={'index':'Dia','Dia':'2° Enero'},inplace=True)
primera_febrero=postulaciones[(postulaciones['Fecha_Postulacion']>='2018-02-01') & (postulaciones['Fecha_Postulacion']<'2018-02-16')]
primera_febrero['Solo_Dia']=primera_febrero['Fecha_Postulacion'].dt.day
primera_quincena_feb=primera_febrero['Solo_Dia'].value_counts().sort_index().reset_index()
quincenas['1° Febrero']=primera_quincena_feb['Solo_Dia']
segunda_febrero=postulaciones[(postulaciones['Fecha_Postulacion']>'2018-02-15') & (postulaciones['Fecha_Postulacion']<'2018-02-28')]
segunda_febrero['Solo_Dia']=postulaciones['Fecha_Postulacion'].dt.day
segunda_quincena_feb=segunda_febrero['Solo_Dia'].value_counts().sort_index().reset_index()
quincenas['2° Febrero']=segunda_quincena_feb['Solo_Dia']
group_quincenas=quincenas.groupby(['Dia']).agg('sum')
group_quincenas
grafico_quincena=group_quincenas.plot(kind='line',figsize=(8,8),fontsize=15)
grafico_quincena.set_title("Cantidad de Postulaciones por Quincena",fontsize=20)
grafico_quincena.set_xlabel("Dia",fontsize=12)
grafico_quincena.set_ylabel("Cantidad de Postulaciones",fontsize=12)
plt.legend(['2° Enero','1° Febrero','2° Febrero'],fontsize=15)
semana=postulaciones[['Fecha_Postulacion']]
semana['Dia_Semana']=semana['Fecha_Postulacion'].dt.weekday
semana['Dia_Semana']=(semana['Dia_Semana']+1)
semana['Semana']=semana['Fecha_Postulacion'].dt.week
semana['Valor']=1
pivot_dia_semana=semana.pivot_table(index='Semana',columns='Dia_Semana',values='Valor',aggfunc='sum')
semana.drop(columns={'Valor'},inplace=True)
pivot_dia_semana.rename(index={3:'3° Enero',4:'4° Enero',5:'5° Enero',6:'1° Febrero',7:'2° Febrero',8:'3° Febrero',9:"4° Febrero"},inplace=True)
pivot_dia_semana.rename(columns={1:'Lunes',2:'Martes',3:'Miercoles',4:'Jueves',5:'Viernes',6:'Sabado',7:'Domingo'},inplace=True)
pivot_dia_semana
plt.subplots(figsize=(8,8))
grafico_semana=sns.heatmap(pivot_dia_semana,linewidths=.5,cmap="magma_r")
grafico_semana.set_title("Postulaciones por Dia de Semana",fontsize=22)
grafico_semana.set_xlabel("Dia de Semana",fontsize=15)
grafico_semana.set_ylabel("Semana del Mes",fontsize=15)
grafico_semana.set_yticklabels(grafico_semana.get_yticklabels(),rotation=0)
#Extraigo las horas de 'Fecha_Postulacion'
postulaciones['Hora']=postulaciones['Fecha_Postulacion'].dt.hour
hora_general=postulaciones['Hora'].value_counts().sort_index()
grafico_hora_general=hora_general.plot(rot=0,figsize=(8,8),kind='bar',color='purple')
grafico_hora_general.set_title("Cantidad de Postulaciones por Hora",fontsize=20)
grafico_hora_general.set_xlabel("Horas de Dia",fontsize=12)
grafico_hora_general.set_ylabel("Cantidad de Postulaciones",fontsize=12)
#Ahora vamos a juntar la hora de Postulacion con la edad de los postulantes
hora_edad=postulaciones[['idpostulante','Hora']]
hora_edad_general=pd.merge(hora_edad,edad,on='idpostulante',how='inner')
group_hora_edad=hora_edad_general.groupby(['Hora']).agg({'Edad':'mean'})
grafico_group_edad=group_hora_edad.plot(kind='bar',rot=0,figsize=(8,8),fontsize=12,legend=False)
grafico_group_edad.set_title("Edad promedio de Postulaciones por Hora",fontsize=20)
grafico_group_edad.set_xlabel("Horas del Dia",fontsize=12)
grafico_group_edad.set_ylabel("Edad",fontsize=12)
plt.ylim([0,40])
vistas['Hora']=vistas['timestamp'].dt.hour
hora_vistas=vistas['Hora'].value_counts().sort_index()
grafico_horas_vistas=hora_vistas.plot(kind='bar',figsize=(8,8),fontsize=12,rot=0,color='teal')
grafico_horas_vistas.set_title("Cantidad de Vistas por Hora",fontsize=20)
grafico_horas_vistas.set_xlabel("Horas",fontsize=12)
grafico_horas_vistas.set_ylabel("Cantidad de Vistas",fontsize=12)
#Separo las fechas con la hora y las dejo solo con la fecha
vistas['Fecha_Vista']=vistas['timestamp'].dt.date
postulaciones['Fecha_sin_hora']=postulaciones['Fecha_Postulacion'].dt.date
postulacion_igual_vista=postulaciones[(postulaciones['Fecha_Postulacion']>='2018-02-23') & (postulaciones['Fecha_Postulacion']<'2018-03-01')]
vistas_igual_postulacion=vistas[(vistas['timestamp']<'2018-03-01')]
fin_feb=postulacion_igual_vista['Fecha_sin_hora'].value_counts().sort_index().reset_index()
vistas_fin_feb=vistas_igual_postulacion['Fecha_Vista'].value_counts().sort_index().reset_index()
fin_feb['Vistas']=vistas_fin_feb['Fecha_Vista']
fin_feb.rename(columns={'index':'Dia','Fecha_sin_hora':'Postulaciones'},inplace=True)
fin_feb.set_index('Dia',inplace=True)
fin_feb
grafico_fin_feb=fin_feb.plot(kind='line',figsize=(8,8),fontsize=12)
grafico_fin_feb.set_title("Cantidad de Vistas/Postulaciones por Dia",fontsize=20)
grafico_fin_feb.set_xlabel("Dia",fontsize=12)
grafico_fin_feb.set_ylabel("Cantidad de Vistas/Postulaciones",fontsize=12)
plt.legend(['Postulaciones','Vistas'],fontsize=12)
tipo_avisos=avisos_detalle['Tipo_de_Trabajo'].value_counts()
tipo_avisos_top=tipo_avisos.head(2)
tipo_avisos_resto=tipo_avisos.tail(7)
tipo_avisos
plt.subplots(figsize=(8,8))
grafico_tipo_avisos_top=sns.barplot(x=tipo_avisos_top.values,y=tipo_avisos_top.index,orient='h',palette="binary_r")
grafico_tipo_avisos_top.set_title("Cantidad de Tipos de Trabajo en los avisos (Principales)",fontsize=20)
grafico_tipo_avisos_top.set_xlabel("Cantidad de Avisos",fontsize=12)
grafico_tipo_avisos_top.set_ylabel("Tipo de Trabajo",fontsize=12)
plt.subplots(figsize=(8,8))
grafico_tipo_avisos_resto=sns.barplot(x=tipo_avisos_resto.values,y=tipo_avisos_resto.index,orient='h',palette="gist_earth")
grafico_tipo_avisos_resto.set_title("Cantidad de Tipos de Trabajo en los avisos (Resto)",fontsize=20)
grafico_tipo_avisos_resto.set_xlabel("Cantidad de Avisos",fontsize=12)
grafico_tipo_avisos_resto.set_ylabel("Tipo de Trabajo",fontsize=12)
tipo_trabajo=avisos_detalle[['idaviso','Tipo_de_Trabajo']]
postulantes_tipo=pd.merge(tipo_trabajo,postulaciones,on='idaviso',how='inner')
cantidad_tipo_post=postulantes_tipo['Tipo_de_Trabajo'].value_counts()
cantidad_tipo_post_resto=cantidad_tipo_post.tail(7)
cantidad_tipo_post_top=cantidad_tipo_post.head(2)
cantidad_tipo_post
plt.subplots(figsize=(8,8))
grafico_cantidad_tipo_post_top=sns.barplot(x=cantidad_tipo_post_top.values,y=cantidad_tipo_post_top.index,orient='h')
grafico_cantidad_tipo_post_top.set_title("Cantidad de Postulaciones por Tipo de Trabajo (Principales)",fontsize=20)
grafico_cantidad_tipo_post_top.set_xlabel("Cantidad de Postulaciones",fontsize=12)
grafico_cantidad_tipo_post_top.set_ylabel("Tipo de Trabajo",fontsize=12)
plt.subplots(figsize=(8,8))
grafico_cantidad_tipo_post_resto=sns.barplot(x=cantidad_tipo_post_resto.values,y=cantidad_tipo_post_resto.index,orient='h',palette="tab20b_r")
grafico_cantidad_tipo_post_resto.set_title("Cantidad de Postulaciones por Tipo de Trabajo (Resto)",fontsize=20)
grafico_cantidad_tipo_post_resto.set_xlabel("Cantidad de Postulaciones",fontsize=12)
grafico_cantidad_tipo_post_resto.set_ylabel("Tipo de Trabajo",fontsize=12)
vistas_tipo=pd.merge(tipo_trabajo,vistas,on='idaviso',how='inner')
cantidad_tipo_vistas=vistas_tipo['Tipo_de_Trabajo'].value_counts()
cantidad_tipo_vistas_top=cantidad_tipo_vistas.head(2)
cantidad_tipo_vistas_resto=cantidad_tipo_vistas.tail(7)
cantidad_tipo_vistas
plt.subplots(figsize=(8,8))
grafico_cantidad_tipo_vistas_top=sns.barplot(x=cantidad_tipo_vistas_top.values,y=cantidad_tipo_vistas_top.index,orient='h',palette="gist_heat_r")
grafico_cantidad_tipo_vistas_top.set_title("Cantidad de Vistas por Tipo de Trabajo (Principales)",fontsize=20)
grafico_cantidad_tipo_vistas_top.set_xlabel("Cantidad de Vistas",fontsize=12)
grafico_cantidad_tipo_vistas_top.set_ylabel("Tipo de Trabajo",fontsize=12)
plt.subplots(figsize=(8,8))
grafico_cantidad_tipo_vistas_resto=sns.barplot(x=cantidad_tipo_vistas_resto.values,y=cantidad_tipo_vistas_resto.index,orient='h',palette='BuGn_d')
grafico_cantidad_tipo_vistas_resto.set_title("Cantidad de Vistas por Tipo de Trabajo (Resto)",fontsize=20)
grafico_cantidad_tipo_vistas_resto.set_xlabel("Cantidad de Vistas",fontsize=12)
grafico_cantidad_tipo_vistas_resto.set_ylabel("Tipo de Trabajo",fontsize=12)
postulantes_tipo_femenino=pd.merge(postulantes_tipo,genero_femenino,on='idpostulante',how='inner')
postulantes_tipo_masculino=pd.merge(postulantes_tipo,genero_masculino,on='idpostulante',how='inner')
cantidad_tipo_femenino=postulantes_tipo_femenino['Tipo_de_Trabajo'].value_counts()
cantidad_tipo_femenino_top=cantidad_tipo_femenino.head(2).sort_index()
cantidad_tipo_femenino_resto=cantidad_tipo_femenino.tail(7).sort_index()
cantidad_tipo_masculino=postulantes_tipo_masculino['Tipo_de_Trabajo'].value_counts()
cantidad_tipo_masculino_top=cantidad_tipo_masculino.head(2).sort_index()
cantidad_tipo_masculino_resto=cantidad_tipo_masculino.tail(7).sort_index()
df_cantidad_masculino_top=cantidad_tipo_masculino_top.reset_index()
cantidad_top_general=cantidad_tipo_femenino_top.reset_index()
cantidad_top_general['Masculino']=df_cantidad_masculino_top['Tipo_de_Trabajo']
cantidad_top_general.rename(columns={'index':'Tipo_de_Trabajo','Tipo_de_Trabajo':'Femenino'},inplace=True)
cantidad_top_general.set_index('Tipo_de_Trabajo',inplace=True)
cantidad_top_general
grafico_cantidad_top_general=cantidad_top_general.plot(kind='bar',figsize=(8,8),fontsize=12,color=['salmon','royalblue'],rot=70)
grafico_cantidad_top_general.set_title("Cantidad de Postulaciones por Tipo de Trabajo por género (Principales)",fontsize=20)
grafico_cantidad_top_general.set_xlabel("Tipo de Trabajo",fontsize=12)
grafico_cantidad_top_general.set_ylabel("Cantidad de Postulaciones",fontsize=12)
leyenda=plt.legend(['Femenino','Masculino'],fontsize=12,title='Genero',frameon=True,facecolor='white',edgecolor='black')
leyenda.get_frame().set_linewidth(1.0)
df_cantidad_masculino_resto=cantidad_tipo_masculino_resto.reset_index()
cantidad_resto_general=cantidad_tipo_femenino_resto.reset_index()
cantidad_resto_general['Masculino']=df_cantidad_masculino_resto['Tipo_de_Trabajo']
cantidad_resto_general.rename(columns={'index':'Tipo_de_Trabajo','Tipo_de_Trabajo':'Femenino'},inplace=True)
cantidad_resto_general.set_index('Tipo_de_Trabajo',inplace=True)
cantidad_resto_general
grafico_cantidad_resto_general=cantidad_resto_general.plot(kind='bar',figsize=(8,8),fontsize=12,color=['salmon','royalblue'],rot=70)
grafico_cantidad_resto_general.set_title("Cantidad de Postulaciones por Tipo de Trabajo por género (Resto)",fontsize=20)
grafico_cantidad_resto_general.set_xlabel("Tipo de Trabajo",fontsize=12)
grafico_cantidad_resto_general.set_ylabel("Cantidad de Postulaciones",fontsize=12)
leyenda=plt.legend(['Femenino','Masculino'],fontsize=12,title='Genero',frameon=True,facecolor='white',edgecolor='black')
leyenda.get_frame().set_linewidth(1.0)
postulantes_tipo_edad=pd.merge(postulantes_tipo,edad,on='idpostulante',how='inner')
tipo_edad_general=postulantes_tipo_edad.groupby(['Tipo_de_Trabajo']).agg({'Edad':'mean'})
tipo_edad_general
plt.subplots(figsize=(8,8))
grafico_tipo_edad_general=sns.barplot(y=tipo_edad_general.index,x=tipo_edad_general['Edad'],orient='h',palette="Blues_d")
grafico_tipo_edad_general.set_title("Edad Promedio de Postulaciones por Tipo de Trabajo",fontsize=20)
grafico_tipo_edad_general.set_ylabel("Tipo de Trabajo",fontsize=12)
grafico_tipo_edad_general.set_xlabel("Edad",fontsize=12)
plt.xlim([0,34])
postulantes_tipo_edad_femenino=postulantes_tipo_edad[(postulantes_tipo_edad['Sexo']=='FEM')]
tipo_edad_femenino=postulantes_tipo_edad_femenino.groupby(['Tipo_de_Trabajo']).agg({'Edad':'mean'})
tipo_edad_femenino
plt.subplots(figsize=(8,8))
grafico_tipo_edad_femenino=sns.barplot(y=tipo_edad_femenino.index,x=tipo_edad_femenino['Edad'],orient='h',palette="icefire_r")
grafico_tipo_edad_femenino.set_title("Edad Promedio de Postulaciones por Tipo de Trabajo (Femenino)",fontsize=20)
grafico_tipo_edad_femenino.set_ylabel("Tipo de Trabajo",fontsize=12)
grafico_tipo_edad_femenino.set_xlabel("Edad",fontsize=12)
plt.xlim([0,32])
postulantes_tipo_edad_masculino=postulantes_tipo_edad[(postulantes_tipo_edad['Sexo']=='MASC')]
tipo_edad_masculino=postulantes_tipo_edad_masculino.groupby(['Tipo_de_Trabajo']).agg({'Edad':'mean'})
tipo_edad_masculino
plt.subplots(figsize=(8,8))
grafico_tipo_edad_masculino=sns.barplot(y=tipo_edad_masculino.index,x=tipo_edad_masculino['Edad'],orient='h',palette="copper")
grafico_tipo_edad_masculino.set_title("Edad Promedio de Postulaciones por Tipo de Trabajo (Masculino)",fontsize=20)
grafico_tipo_edad_masculino.set_ylabel("Tipo de Trabajo",fontsize=12)
grafico_tipo_edad_masculino.set_xlabel("Edad",fontsize=12)
plt.xlim([0,34])
tipo_area=avisos_detalle[['idaviso','Tipo_de_Trabajo','Nombre_Area']]
postulaciones_tipo_area=pd.merge(tipo_area,postulaciones,on='idaviso',how='inner')
postulaciones_full_area=postulaciones_tipo_area[(postulaciones_tipo_area['Tipo_de_Trabajo']=='Full-time')]
postulaciones_full_top=pd.merge(postulaciones_full_area,df_top_post,on='Nombre_Area',how='inner')
cantidad_full_top=postulaciones_full_top['Nombre_Area'].value_counts()
cantidad_full_top
plt.subplots(figsize=(8,8))
grafico_cantidad_full_top=sns.barplot(x=cantidad_full_top.values,y=cantidad_full_top.index,orient='h',palette="cool_r")
grafico_cantidad_full_top.set_title("Cantidad de Postulaciones a Full-Time en Áreas Principales",fontsize=20)
grafico_cantidad_full_top.set_xlabel("Cantidad de Postulaciones",fontsize=12)
grafico_cantidad_full_top.set_ylabel("Área de Trabajo",fontsize=12)
postulaciones_part_area=postulaciones_tipo_area[(postulaciones_tipo_area['Tipo_de_Trabajo']=='Part-time')]
postulaciones_part_top=pd.merge(postulaciones_part_area,df_top_post,on='Nombre_Area',how='inner')
cantidad_part_top=postulaciones_part_top['Nombre_Area'].value_counts()
cantidad_part_top
plt.subplots(figsize=(8,8))
grafico_cantidad_part_top=sns.barplot(x=cantidad_part_top.values,y=cantidad_part_top.index,orient='h',palette="gist_heat_r")
grafico_cantidad_part_top.set_title("Cantidad de Postulaciones a Part-Time en Áreas Principales",fontsize=20)
grafico_cantidad_part_top.set_xlabel("Cantidad de Postulaciones",fontsize=12)
grafico_cantidad_part_top.set_ylabel("Área de Trabajo",fontsize=12)
postulaciones_pasantia_area=postulaciones_tipo_area[(postulaciones_tipo_area['Tipo_de_Trabajo']=='Pasantia')]
postulaciones_pasantia_top=pd.merge(postulaciones_pasantia_area,df_top_post,on='Nombre_Area',how='inner')
cantidad_pasantia_top=postulaciones_pasantia_top['Nombre_Area'].value_counts()
cantidad_pasantia_top
plt.subplots(figsize=(8,8))
grafico_cantidad_pasantia_top=sns.barplot(x=cantidad_pasantia_top.values,y=cantidad_pasantia_top.index,orient='h',palette="Spectral_r")
grafico_cantidad_pasantia_top.set_title("Cantidad de Postulaciones a Pasantía en Áreas Principales",fontsize=20)
grafico_cantidad_pasantia_top.set_xlabel("Cantidad de Postulaciones",fontsize=12)
grafico_cantidad_pasantia_top.set_ylabel("Área de Trabajo",fontsize=12)
#Como hay poca cantidad de Pasantias en las áreas principales, veremos en donde hay más postulaciones en Pasantía
pasantias_mejores=postulaciones_pasantia_area['Nombre_Area'].value_counts().head(10)
plt.subplots(figsize=(8,8))
grafico_pasantias_mejores=sns.barplot(x=pasantias_mejores.values,y=pasantias_mejores.index,orient='h')
grafico_pasantias_mejores.set_title("Principales Áreas donde hay más Postulaciones en Pasantías",fontsize=20)
grafico_pasantias_mejores.set_xlabel("Cantidad de Postulaciones",fontsize=12)
grafico_pasantias_mejores.set_ylabel("Área de Trabajo",fontsize=12)
educacion_graduados=educacion[(educacion['Estado']=='Graduado')]
postulaciones_tipo_graduado=pd.merge(postulantes_tipo,educacion_graduados,on='idpostulante',how='inner')
postulaciones_tipo_graduado['Valor']=1
postulaciones_tipo_graduado_top=postulaciones_tipo_graduado[(postulaciones_tipo_graduado['Tipo_de_Trabajo']=='Full-time') | (postulaciones_tipo_graduado['Tipo_de_Trabajo']=='Part-time')]
pivot_postulaciones_tipo_graduado_top=postulaciones_tipo_graduado_top.pivot_table(index='Nivel',columns='Tipo_de_Trabajo',values='Valor',aggfunc='sum')
pivot_postulaciones_tipo_graduado_top
plt.subplots(figsize=(8,8))
grafico_pivot_postulaciones_tipo_graduado_top=sns.heatmap(pivot_postulaciones_tipo_graduado_top,linewidths=.5,fmt="d",annot=True,cmap="tab20c")
grafico_pivot_postulaciones_tipo_graduado_top.set_title("Postulaciones de Graduados según Nivel de Educacion en Full-Time y Part-Time",fontsize=14)
grafico_pivot_postulaciones_tipo_graduado_top.set_xlabel("Tipo de Trabajo",fontsize=12)
grafico_pivot_postulaciones_tipo_graduado_top.set_ylabel("Nivel de Educacion",fontsize=12)
postulaciones_tipo_graduado_resto=postulaciones_tipo_graduado[(postulaciones_tipo_graduado['Tipo_de_Trabajo'] != 'Full-time') & (postulaciones_tipo_graduado['Tipo_de_Trabajo'] != 'Part-time')]
pivot_postulaciones_tipo_graduado_resto=postulaciones_tipo_graduado_resto.pivot_table(index='Nivel',columns='Tipo_de_Trabajo',values='Valor',aggfunc='sum')
pivot_postulaciones_tipo_graduado_resto.fillna(0,inplace=True)
pivot_postulaciones_tipo_graduado_resto
plt.subplots(figsize=(8,8))
grafico_pivot_postulaciones_tipo_graduado_resto=sns.heatmap(pivot_postulaciones_tipo_graduado_resto,linewidths=.5,cmap="rainbow_r")
grafico_pivot_postulaciones_tipo_graduado_resto.set_title("Postulaciones de Graduados según Nivel de Educacion (Resto)",fontsize=14)
grafico_pivot_postulaciones_tipo_graduado_resto.set_xlabel("Tipo de Trabajo",fontsize=12)
grafico_pivot_postulaciones_tipo_graduado_resto.set_ylabel("Nivel de Educacion",fontsize=12)
grafico_pivot_postulaciones_tipo_graduado_resto.set_xticklabels(grafico_pivot_postulaciones_tipo_graduado_resto.get_xticklabels(),rotation=60)
educacion_cursando=educacion[(educacion['Estado']=='En Curso')]
postulaciones_tipo_cursando=pd.merge(postulantes_tipo,educacion_cursando,on='idpostulante',how='inner')
postulaciones_tipo_cursando['Valor']=1
postulaciones_tipo_cursando_top=postulaciones_tipo_cursando[(postulaciones_tipo_cursando['Tipo_de_Trabajo']=='Full-time') | (postulaciones_tipo_cursando['Tipo_de_Trabajo']=='Part-time')]
pivot_postulaciones_tipo_cursando_top=postulaciones_tipo_cursando_top.pivot_table(index='Nivel',columns='Tipo_de_Trabajo',values='Valor',aggfunc='sum')
pivot_postulaciones_tipo_cursando_top
plt.subplots(figsize=(8,8))
grafico_pivot_postulaciones_tipo_cursando_top=sns.heatmap(pivot_postulaciones_tipo_cursando_top,linewidths=.5,fmt="d",annot=True,cmap="tab20c")
grafico_pivot_postulaciones_tipo_cursando_top.set_title("Postulaciones de estado Cursando según Nivel de Educacion en Full-Time y Part-Time",fontsize=14)
grafico_pivot_postulaciones_tipo_cursando_top.set_xlabel("Tipo de Trabajo",fontsize=12)
grafico_pivot_postulaciones_tipo_cursando_top.set_ylabel("Nivel de Educacion",fontsize=12)
postulaciones_tipo_cursando_resto=postulaciones_tipo_cursando[(postulaciones_tipo_cursando['Tipo_de_Trabajo'] != 'Full-time') & (postulaciones_tipo_cursando['Tipo_de_Trabajo'] != 'Part-time')]
pivot_postulaciones_tipo_cursando_resto=postulaciones_tipo_cursando_resto.pivot_table(index='Nivel',columns='Tipo_de_Trabajo',values='Valor',aggfunc='sum')
pivot_postulaciones_tipo_cursando_resto.fillna(0,inplace=True)
pivot_postulaciones_tipo_cursando_resto
plt.subplots(figsize=(8,8))
grafico_pivot_postulaciones_tipo_cursando_resto=sns.heatmap(pivot_postulaciones_tipo_cursando_resto,linewidths=.5,cmap="rainbow_r")
grafico_pivot_postulaciones_tipo_cursando_resto.set_title("Postulaciones de estado Cursando según Nivel de Educacion (Resto)",fontsize=14)
grafico_pivot_postulaciones_tipo_cursando_resto.set_xlabel("Tipo de Trabajo",fontsize=12)
grafico_pivot_postulaciones_tipo_cursando_resto.set_ylabel("Nivel de Educacion",fontsize=12)
grafico_pivot_postulaciones_tipo_cursando_resto.set_xticklabels(grafico_pivot_postulaciones_tipo_cursando_resto.get_xticklabels(),rotation=60)
tipo_nivel_trabajo=avisos_detalle['Nivel_Laboral'].value_counts()
tipo_nivel_trabajo
plt.subplots(figsize=(8,8))
grafico_tipo_nivel_trabajo=sns.barplot(x=tipo_nivel_trabajo.values,y=tipo_nivel_trabajo.index,orient='h',palette="Paired_r")
grafico_tipo_nivel_trabajo.set_title("Cantidad de Avisos por Tipo de Nivel Laboral ",fontsize=20)
grafico_tipo_nivel_trabajo.set_xlabel("Cantidad de Avisos",fontsize=12)
grafico_tipo_nivel_trabajo.set_ylabel("Tipo de Nivel Laboral",fontsize=12)
nivel_trabajo=avisos_detalle[['idaviso','Nivel_Laboral','Nombre_Area']]
postulaciones_nivel_trabajo=pd.merge(postulaciones,nivel_trabajo,on='idaviso',how='inner')
cantidad_nivel_trabajo_postulaciones=postulaciones_nivel_trabajo['Nivel_Laboral'].value_counts()
cantidad_nivel_trabajo_postulaciones
plt.subplots(figsize=(8,8))
grafico_cantidad_nivel_trabajo_postulaciones=sns.barplot(x=cantidad_nivel_trabajo_postulaciones.values,y=cantidad_nivel_trabajo_postulaciones.index,orient='h',palette="CMRmap_r")
grafico_cantidad_nivel_trabajo_postulaciones.set_title("Cantidad de Postulaciones por Nivel Laboral",fontsize=20)
grafico_cantidad_nivel_trabajo_postulaciones.set_xlabel("Cantidad de Postulaciones",fontsize=12)
grafico_cantidad_nivel_trabajo_postulaciones.set_ylabel("Tipo de Nivel Laboral",fontsize=12)
vistas_nivel_trabajo=pd.merge(vistas,nivel_trabajo,on='idaviso',how='inner')
cantidad_nivel_trabajo_vistas=vistas_nivel_trabajo['Nivel_Laboral'].value_counts()
cantidad_nivel_trabajo_vistas
plt.subplots(figsize=(8,8))
grafico_cantidad_nivel_trabajo_vistas=sns.barplot(x=cantidad_nivel_trabajo_vistas.values,y=cantidad_nivel_trabajo_vistas.index,orient='h',palette="terrain_r")
grafico_cantidad_nivel_trabajo_vistas.set_title("Cantidad de Vistas por Nivel Laboral",fontsize=20)
grafico_cantidad_nivel_trabajo_vistas.set_xlabel("Cantidad de Vistas",fontsize=12)
grafico_cantidad_nivel_trabajo_vistas.set_ylabel("Tipo de Nivel Laboral",fontsize=12)
postulaciones_nivel_trabajo_femenino=pd.merge(postulaciones_nivel_trabajo,genero_femenino,on='idpostulante',how='inner')
postulaciones_nivel_trabajo_masculino=pd.merge(postulaciones_nivel_trabajo,genero_masculino,on='idpostulante',how='inner')
cantidad_nivel_trabajo_general=postulaciones_nivel_trabajo_femenino['Nivel_Laboral'].value_counts().sort_index().reset_index()
cantidad_nivel_trabajo_masculino=postulaciones_nivel_trabajo_masculino['Nivel_Laboral'].value_counts().sort_index().reset_index()
cantidad_nivel_trabajo_general.rename(columns={'index':'Nivel_Laboral','Nivel_Laboral':'Femenino'},inplace=True)
cantidad_nivel_trabajo_general['Masculino']=cantidad_nivel_trabajo_masculino['Nivel_Laboral']
cantidad_nivel_trabajo_general.set_index('Nivel_Laboral',inplace=True)
cantidad_nivel_trabajo_general
grafico_cantidad_nivel_trabajo_general=cantidad_nivel_trabajo_general.plot(kind='bar',figsize=(8,8),fontsize=12,color=['salmon','royalblue'],rot=70)
grafico_cantidad_nivel_trabajo_general.set_title("Cantidad de Postulaciones por Nivel Laboral según Género",fontsize=20)
grafico_cantidad_nivel_trabajo_general.set_xlabel("Tipo de Nivel Laboral",fontsize=12)
grafico_cantidad_nivel_trabajo_general.set_ylabel("Cantidad de Postulaciones",fontsize=12)
leyenda=plt.legend(["Femenino","Masculino"],fontsize=12,title='Genero',frameon=True,facecolor='white',edgecolor='black')
leyenda.get_frame().set_linewidth(1.0)
postulaciones_nivel_trabajo_top=pd.merge(postulaciones_nivel_trabajo,df_top_post,on='Nombre_Area',how='inner')
postulaciones_nivel_trabajo_top['Valor']=1
postulaciones_nivel_top_trabajo_top=postulaciones_nivel_trabajo_top[(postulaciones_nivel_trabajo_top['Nivel_Laboral']=='Junior') | (postulaciones_nivel_trabajo_top['Nivel_Laboral']=='Otro') | (postulaciones_nivel_trabajo_top['Nivel_Laboral']=='Senior / Semi-Senior')]
pivot_top_nivel_top=postulaciones_nivel_top_trabajo_top.pivot_table(index='Nombre_Area',columns='Nivel_Laboral',values='Valor',aggfunc='sum')
pivot_top_nivel_top
plt.subplots(figsize=(8,8))
grafico_pivot_top_nivel_top=sns.heatmap(pivot_top_nivel_top,linewidths=.5,fmt="d",annot=True,cmap="tab20c")
grafico_pivot_top_nivel_top.set_title("Postulaciones de los Niveles Laborales en Áreas de Trabajo Principales (Principales)",fontsize=16)
grafico_pivot_top_nivel_top.set_xlabel("Nivel Laboral",fontsize=12)
grafico_pivot_top_nivel_top.set_ylabel("Área de Trabajo",fontsize=12)
postulaciones_nivel_resto_trabajo_top=postulaciones_nivel_trabajo_top[(postulaciones_nivel_trabajo_top['Nivel_Laboral']=='Gerencia / Alta Gerencia / Dirección') | (postulaciones_nivel_trabajo_top['Nivel_Laboral']=='Jefe / Supervisor / Responsable')]
pivot_resto_nivel_top=postulaciones_nivel_resto_trabajo_top.pivot_table(index='Nombre_Area',columns='Nivel_Laboral',values='Valor',aggfunc='sum')
pivot_resto_nivel_top.fillna(0,inplace=True)
pivot_resto_nivel_top
plt.subplots(figsize=(8,8))
grafico_pivot_resto_nivel_top=sns.heatmap(pivot_resto_nivel_top,linewidths=.5,center=0,fmt="f",annot=True,cmap="gist_heat_r")
grafico_pivot_resto_nivel_top.set_title("Postulaciones de los Niveles Laborales en Áreas de Trabajo Principales (Resto)",fontsize=16)
grafico_pivot_resto_nivel_top.set_xlabel("Nivel de Trabajo",fontsize=12)
grafico_pivot_resto_nivel_top.set_ylabel("Área de Trabajo",fontsize=12)
grafico_pivot_resto_nivel_top.set_xticklabels(grafico_pivot_resto_nivel_top.get_xticklabels(),rotation=60)
nivel_laboral_tipo_trabajo=avisos_detalle[['idaviso','Nombre_Area','Tipo_de_Trabajo','Nivel_Laboral']]
nivel_laboral_tipo_trabajo['Valor']=1
nivel_laboral_tipo_trabajo_top=nivel_laboral_tipo_trabajo[(nivel_laboral_tipo_trabajo['Tipo_de_Trabajo']=='Full-time') | (nivel_laboral_tipo_trabajo['Tipo_de_Trabajo']=='Part-time')]
pivot_nivel_laboral_tipo_trabajo_top=nivel_laboral_tipo_trabajo_top.pivot_table(index='Tipo_de_Trabajo',columns='Nivel_Laboral',values='Valor',aggfunc='sum')
pivot_nivel_laboral_tipo_trabajo_top
grafico_pivot_nivel_laboral_tipo_trabajo_top=pivot_nivel_laboral_tipo_trabajo_top.plot(kind='bar',figsize=(8,8),fontsize=12,rot=0)
grafico_pivot_nivel_laboral_tipo_trabajo_top.set_title("Cantidad de Avisos por Nivel Laboral en relación con Full-Time y Part-Time",fontsize=16)
grafico_pivot_nivel_laboral_tipo_trabajo_top.set_xlabel("Tipo de Trabajo",fontsize=12)
grafico_pivot_nivel_laboral_tipo_trabajo_top.set_ylabel("Cantidad de Avisos",fontsize=12)
leyenda=plt.legend(["Gerencia / Alta Gerencia / Dirección","Jefe / Supervisor / Responsable","Junior","Otro","Senior / Semi-Senior"],fontsize=12,title='Nivel Laboral',frameon=True,facecolor='white',edgecolor='black')
leyenda.get_frame().set_linewidth(1.0)
nivel_laboral_tipo_trabajo_resto=nivel_laboral_tipo_trabajo[(nivel_laboral_tipo_trabajo['Tipo_de_Trabajo'] !='Full-time') & (nivel_laboral_tipo_trabajo['Tipo_de_Trabajo'] !='Part-time')]
pivot_nivel_laboral_tipo_trabajo_resto=nivel_laboral_tipo_trabajo_resto.pivot_table(index='Tipo_de_Trabajo',columns='Nivel_Laboral',values='Valor',aggfunc='sum')
pivot_nivel_laboral_tipo_trabajo_resto.fillna(0,inplace=True)
pivot_nivel_laboral_tipo_trabajo_resto
grafico_pivot_nivel_laboral_tipo_trabajo_resto=pivot_nivel_laboral_tipo_trabajo_resto.plot(kind='bar',figsize=(9,9),fontsize=12,rot=70)
grafico_pivot_nivel_laboral_tipo_trabajo_resto.set_title("Cantidad de Avisos por Nivel Laboral en relación con Tipo de Trabajo (Resto)",fontsize=16)
grafico_pivot_nivel_laboral_tipo_trabajo_resto.set_xlabel("Tipo de Trabajo",fontsize=12)
grafico_pivot_nivel_laboral_tipo_trabajo_resto.set_ylabel("Cantidad de Avisos",fontsize=12)
leyenda=plt.legend(["Gerencia / Alta Gerencia / Dirección","Jefe / Supervisor / Responsable","Junior","Otro","Senior / Semi-Senior"],fontsize=12,title='Nivel Laboral',frameon=True,facecolor='white',edgecolor='black',bbox_to_anchor=(1.05, 1))
leyenda.get_frame().set_linewidth(1.0)
postulaciones_nivel_tipo_trabajo=pd.merge(postulaciones,nivel_laboral_tipo_trabajo,on='idaviso',how='inner')
postulaciones_nivel_tipo_trabajo_top=postulaciones_nivel_tipo_trabajo[(postulaciones_nivel_tipo_trabajo['Tipo_de_Trabajo']=='Full-time') | (postulaciones_nivel_tipo_trabajo['Tipo_de_Trabajo']=='Part-time')]
pivot_postulaciones_nivel_tipo_trabajo_top=postulaciones_nivel_tipo_trabajo_top.pivot_table(index='Nivel_Laboral',columns='Tipo_de_Trabajo',values='Valor',aggfunc='sum')
pivot_postulaciones_nivel_tipo_trabajo_top
plt.subplots(figsize=(8,8))
grafico_pivot_postulaciones_nivel_tipo_trabajo_top=sns.heatmap(pivot_postulaciones_nivel_tipo_trabajo_top,linewidths=.5,fmt="d",annot=True,cmap="icefire_r")
grafico_pivot_postulaciones_nivel_tipo_trabajo_top.set_title("Postulaciones por Full-Time y Part-Time en relación con el Nivel Laboral",fontsize=16)
grafico_pivot_postulaciones_nivel_tipo_trabajo_top.set_xlabel("Tipo de Trabajo",fontsize=12)
grafico_pivot_postulaciones_nivel_tipo_trabajo_top.set_ylabel("Nivel Laboral",fontsize=12)
postulaciones_nivel_tipo_trabajo_resto=postulaciones_nivel_tipo_trabajo[(postulaciones_nivel_tipo_trabajo['Tipo_de_Trabajo'] != 'Full-time') & (postulaciones_nivel_tipo_trabajo['Tipo_de_Trabajo'] != 'Part-time')]
pivot_postulaciones_nivel_tipo_trabajo_resto=postulaciones_nivel_tipo_trabajo_resto.pivot_table(index='Nivel_Laboral',columns='Tipo_de_Trabajo',values='Valor',aggfunc='sum')
pivot_postulaciones_nivel_tipo_trabajo_resto.fillna(0,inplace=True)
pivot_postulaciones_nivel_tipo_trabajo_resto
plt.subplots(figsize=(8,8))
grafico_pivot_postulaciones_nivel_tipo_trabajo_resto=sns.heatmap(pivot_postulaciones_nivel_tipo_trabajo_resto,linewidths=.5,cmap="tab20b_r")
grafico_pivot_postulaciones_nivel_tipo_trabajo_resto.set_title("Postulaciones por Tipo de Trabajo en relación con el Nivel Laboral",fontsize=16)
grafico_pivot_postulaciones_nivel_tipo_trabajo_resto.set_xlabel("Tipo de Trabajo",fontsize=12)
grafico_pivot_postulaciones_nivel_tipo_trabajo_resto.set_ylabel("Nivel Laboral",fontsize=12)
grafico_pivot_postulaciones_nivel_tipo_trabajo_resto.set_xticklabels(grafico_pivot_postulaciones_nivel_tipo_trabajo_resto.get_xticklabels(),rotation=60)
postulaciones_nivel_laboral_edad=pd.merge(postulaciones_nivel_tipo_trabajo,edad,on='idpostulante',how='inner')
group_postulaciones_nivel_laboral_edad=postulaciones_nivel_laboral_edad.groupby(['Nivel_Laboral']).agg({'Edad':'mean'})
group_postulaciones_nivel_laboral_edad
plt.subplots(figsize=(8,8))
grafico_postulaciones_nivel_laboral_edad=sns.barplot(x=group_postulaciones_nivel_laboral_edad['Edad'],y=group_postulaciones_nivel_laboral_edad.index,orient='h',palette="rainbow_r")
grafico_postulaciones_nivel_laboral_edad.set_title("Edad Promedio de Postulaciones por Nivel Laboral",fontsize=20)
grafico_postulaciones_nivel_laboral_edad.set_xlabel("Edad",fontsize=12)
grafico_postulaciones_nivel_laboral_edad.set_ylabel("Nivel Laboral",fontsize=12)
plt.xlim([0,40])
postulaciones_nivel_laboral_edad_femenino=postulaciones_nivel_laboral_edad[(postulaciones_nivel_laboral_edad['Sexo']=='FEM')]
group_postulaciones_nivel_laboral_edad_femenino=postulaciones_nivel_laboral_edad_femenino.groupby(['Nivel_Laboral']).agg({'Edad':'mean'})
group_postulaciones_nivel_laboral_edad_femenino
plt.subplots(figsize=(8,8))
grafico_postulaciones_nivel_laboral_edad_femenino=sns.barplot(x=group_postulaciones_nivel_laboral_edad_femenino['Edad'],y=group_postulaciones_nivel_laboral_edad_femenino.index,orient='h',palette="YlOrRd_r")
grafico_postulaciones_nivel_laboral_edad_femenino.set_title("Edad Promedio de Postulaciones por Nivel Laboral (Género Femenino)",fontsize=14)
grafico_postulaciones_nivel_laboral_edad_femenino.set_xlabel("Edad",fontsize=12)
grafico_postulaciones_nivel_laboral_edad_femenino.set_ylabel("Nivel Laboral",fontsize=12)
plt.xlim([0,40])
postulaciones_nivel_laboral_edad_masculino=postulaciones_nivel_laboral_edad[(postulaciones_nivel_laboral_edad['Sexo']=='MASC')]
group_postulaciones_nivel_laboral_edad_masculino=postulaciones_nivel_laboral_edad_masculino.groupby(['Nivel_Laboral']).agg({'Edad':'mean'})
group_postulaciones_nivel_laboral_edad_masculino
plt.subplots(figsize=(8,8))
grafico_postulaciones_nivel_laboral_edad_masculino=sns.barplot(x=group_postulaciones_nivel_laboral_edad_masculino['Edad'],y=group_postulaciones_nivel_laboral_edad_masculino.index,orient='h',palette="spring_r")
grafico_postulaciones_nivel_laboral_edad_masculino.set_title("Edad Promedio de Postulaciones por Nivel Laboral (Género Masculino)",fontsize=14)
grafico_postulaciones_nivel_laboral_edad_masculino.set_xlabel("Edad",fontsize=12)
grafico_postulaciones_nivel_laboral_edad_masculino.set_ylabel("Nivel Laboral",fontsize=12)
plt.xlim([0,40])
postulaciones_nivel_trabajo_graduados=pd.merge(postulaciones_nivel_trabajo,educacion_graduados,on='idpostulante',how='inner')
postulaciones_nivel_trabajo_graduados['Valor']=1
pivot_postulaciones_nivel_trabajo_graduados=postulaciones_nivel_trabajo_graduados.pivot_table(index='Nivel_Laboral',columns='Nivel',values='Valor',aggfunc='sum')
pivot_postulaciones_nivel_trabajo_graduados
plt.subplots(figsize=(9,9))
grafico_pivot_postulaciones_nivel_trabajo_graduados=sns.heatmap(pivot_postulaciones_nivel_trabajo_graduados,linewidths=.5,fmt="d",annot=True,cmap="rainbow_r")
grafico_pivot_postulaciones_nivel_trabajo_graduados.set_title("Postulaciones de Nivel Laboral segun estado Graduado",fontsize=20)
grafico_pivot_postulaciones_nivel_trabajo_graduados.set_xlabel("Nivel de Educacion",fontsize=12)
grafico_pivot_postulaciones_nivel_trabajo_graduados.set_ylabel("Nivel Laboral",fontsize=12)
grafico_pivot_postulaciones_nivel_trabajo_graduados.set_xticklabels(grafico_pivot_postulaciones_nivel_trabajo_graduados.get_xticklabels(),rotation=60)
postulaciones_nivel_trabajo_cursando=pd.merge(postulaciones_nivel_trabajo,educacion_cursando,on='idpostulante',how='inner')
postulaciones_nivel_trabajo_cursando['Valor']=1
pivot_postulaciones_nivel_trabajo_cursando=postulaciones_nivel_trabajo_cursando.pivot_table(index='Nivel_Laboral',columns='Nivel',values='Valor',aggfunc='sum')
pivot_postulaciones_nivel_trabajo_cursando
plt.subplots(figsize=(8,8))
grafico_pivot_postulaciones_nivel_trabajo_cursando=sns.heatmap(pivot_postulaciones_nivel_trabajo_cursando,linewidths=.5,fmt="d",annot=True,cmap="tab20c_r")
grafico_pivot_postulaciones_nivel_trabajo_cursando.set_title("Postulaciones de Nivel Laboral segun estado En Curso",fontsize=20)
grafico_pivot_postulaciones_nivel_trabajo_cursando.set_xlabel("Nivel de Educacion",fontsize=12)
grafico_pivot_postulaciones_nivel_trabajo_cursando.set_ylabel("Nivel Laboral",fontsize=12)
grafico_pivot_postulaciones_nivel_trabajo_cursando.set_xticklabels(grafico_pivot_postulaciones_nivel_trabajo_cursando.get_xticklabels(),rotation=60)
zona_avisos=avisos_detalle['Zona'].value_counts().head(2)
zona_avisos
grafico_zona_avisos=zona_avisos.plot(kind='pie',autopct='%1.1f%%',figsize=(7,7),fontsize=12,colors=['orange','purple'],explode=(0.1, 0))
grafico_zona_avisos.set_title("Cantidad de Avisos por Zona",fontsize=20)
grafico_zona_avisos.set_ylabel("")
avisos_zona=avisos_detalle[['idaviso','Zona']]
postulaciones_zona=pd.merge(postulaciones,avisos_zona,on='idaviso',how='inner')
cantidad_postulaciones_zona=postulaciones_zona['Zona'].value_counts().head(2)
cantidad_postulaciones_zona
grafico_cantidad_postulaciones_zona=sns.barplot(x=cantidad_postulaciones_zona.values,y=cantidad_postulaciones_zona.index,orient='h')
grafico_cantidad_postulaciones_zona.set_title("Postulaciones por Zona",fontsize=20)
grafico_cantidad_postulaciones_zona.set_xlabel("Cantidad de Postulaciones",fontsize=12)
grafico_cantidad_postulaciones_zona.set_ylabel("Zona",fontsize=12)
zona_area_trabajo=avisos_detalle[['idaviso','Zona','Nombre_Area']]
postulaciones_zona_area=pd.merge(postulaciones,zona_area_trabajo,on='idaviso',how='inner')
postulaciones_zona_area_top=pd.merge(postulaciones_zona_area,df_top_post,on='Nombre_Area',how='inner')
postulaciones_zona_area_top['Valor']=1
pivot_postulaciones_zona_area_top=postulaciones_zona_area_top.pivot_table(index='Nombre_Area',columns='Zona',values='Valor',aggfunc='sum')
pivot_postulaciones_zona_area_top
plt.subplots(figsize=(8,8))
grafico_postulaciones_zona_area_top=sns.heatmap(pivot_postulaciones_zona_area_top,linewidths=.5,fmt="d",annot=True,cmap="tab20b_r")
grafico_postulaciones_zona_area_top.set_title("Postulaciones por Zona según Área de Trabajo (Principales)",fontsize=20)
grafico_postulaciones_zona_area_top.set_xlabel("Zona",fontsize=12)
grafico_postulaciones_zona_area_top.set_ylabel("Área de Trabajo",fontsize=12)
postulaciones_zona_genero=pd.merge(postulaciones_zona_area,genero,on='idpostulante',how='inner')
postulaciones_zona_genero['Valor']=1
postulaciones_zona_declarado=postulaciones_zona_genero[(postulaciones_zona_genero['Sexo'] != 'NO_DECLARA')]
postulaciones_zona_top_genero=postulaciones_zona_declarado[(postulaciones_zona_declarado['Zona'] != 'GBA Oeste')]
pivot_postulaciones_zona_genero=postulaciones_zona_top_genero.pivot_table(index='Zona',columns='Sexo',values='Valor',aggfunc='sum')
pivot_postulaciones_zona_genero
grafico_pivot_postulaciones_zona_genero=pivot_postulaciones_zona_genero.plot(kind='bar',figsize=(8,8),fontsize=12,color=['salmon','royalblue'],rot=0)
grafico_pivot_postulaciones_zona_genero.set_title("Postulaciones por Zona según Genero",fontsize=20)
grafico_pivot_postulaciones_zona_genero.set_xlabel("Zona",fontsize=12)
grafico_pivot_postulaciones_zona_genero.set_ylabel("Cantidad de Postulaciones",fontsize=12)
leyenda=plt.legend(['Femenino','Masculino'],fontsize=12,title='Genero',frameon=True,facecolor='white',edgecolor='black')
leyenda.get_frame().set_linewidth(1.0)
postulaciones_zona_edad=pd.merge(postulaciones_zona_area,edad,on='idpostulante',how='inner')
postulaciones_zona_top_edad=postulaciones_zona_edad[(postulaciones_zona_edad['Zona'] != 'GBA Oeste')]
postulaciones_zona_edad_general=postulaciones_zona_top_edad.groupby(['Zona']).agg({'Edad':'mean'})
postulaciones_zona_edad_genero=postulaciones_zona_top_edad.pivot_table(index='Zona',columns='Sexo',values='Edad',aggfunc='mean')
postulaciones_zona_edad_genero['General']=postulaciones_zona_edad_general['Edad']
postulaciones_zona_edad_genero
grafico_postulaciones_zona_edad_genero=postulaciones_zona_edad_genero.plot(kind='bar',figsize=(8,8),fontsize=12,rot=0,color=['salmon','royalblue','limegreen'])
grafico_postulaciones_zona_edad_genero.set_title("Edad Promedio de Postulaciones por Zona según genero",fontsize=20)
grafico_postulaciones_zona_edad_genero.set_xlabel("Zona",fontsize=12)
grafico_postulaciones_zona_edad_genero.set_ylabel("Edad",fontsize=12)
leyenda=plt.legend(['Femenino','Masculino','General'],fontsize=12,title='Genero',frameon=True,facecolor='white',edgecolor='black')
leyenda.get_frame().set_linewidth(1.0)
tipo_nivel_zona=avisos_detalle[['idaviso','Zona','Tipo_de_Trabajo','Nivel_Laboral']]
postulaciones_zona_tipo_nivel=pd.merge(postulaciones,tipo_nivel_zona,on='idaviso',how='inner')
postulaciones_zona_tipo_nivel['Valor']=1
postulaciones_zona_top_tipo_nivel=postulaciones_zona_tipo_nivel[(postulaciones_zona_tipo_nivel['Zona'] != 'GBA Oeste')]
postulaciones_zona_top_tipo_top=postulaciones_zona_top_tipo_nivel[(postulaciones_zona_top_tipo_nivel['Tipo_de_Trabajo']=='Full-time') | (postulaciones_zona_top_tipo_nivel['Tipo_de_Trabajo']=='Part-time')]
pivot_postulaciones_zona_top_tipo_top=postulaciones_zona_top_tipo_top.pivot_table(index='Zona',columns='Tipo_de_Trabajo',values='Valor',aggfunc='sum')
pivot_postulaciones_zona_top_tipo_top
grafico_pivot_postulaciones_zona_top_tipo_top=pivot_postulaciones_zona_top_tipo_top.plot(kind='bar',figsize=(8,8),fontsize=12,rot=0)
grafico_pivot_postulaciones_zona_top_tipo_top.set_title("Postulaciones por Zona según Tipo de Trabajo (Principales)",fontsize=20)
grafico_pivot_postulaciones_zona_top_tipo_top.set_xlabel("Zona",fontsize=12)
grafico_pivot_postulaciones_zona_top_tipo_top.set_ylabel("Cantidad de Postulaciones",fontsize=12)
leyenda=plt.legend(['Full-Time','Part-Time'],fontsize=12,title='Tipo de Trabajo',frameon=True,facecolor='white',edgecolor='black')
leyenda.get_frame().set_linewidth(1.0)
postulaciones_zona_top_tipo_resto=postulaciones_zona_top_tipo_nivel[(postulaciones_zona_top_tipo_nivel['Tipo_de_Trabajo'] != 'Full-time') & (postulaciones_zona_top_tipo_nivel['Tipo_de_Trabajo'] != 'Part-time')]
pivot_postulaciones_zona_top_tipo_resto=postulaciones_zona_top_tipo_resto.pivot_table(index='Zona',columns='Tipo_de_Trabajo',values='Valor',aggfunc='sum')
pivot_postulaciones_zona_top_tipo_resto
grafico_pivot_postulaciones_zona_top_tipo_resto=pivot_postulaciones_zona_top_tipo_resto.plot(kind='bar',figsize=(8,8),color=['royalblue','orchid','gray','tomato','c','chocolate','yellowgreen'],fontsize=12,rot=0)
grafico_pivot_postulaciones_zona_top_tipo_resto.set_title("Postulaciones por Zona según Tipo de Trabajo (Resto)",fontsize=20)
grafico_pivot_postulaciones_zona_top_tipo_resto.set_xlabel("Zona",fontsize=12)
grafico_pivot_postulaciones_zona_top_tipo_resto.set_ylabel("Cantidad de Postulaciones",fontsize=12)
leyenda=plt.legend(['Fines de Semana','Pasantía','Por Contrato','Por Horas','Primer Empleo','Teletrabajo','Temporario'],fontsize=12,title='Tipo de Trabajo',frameon=True,facecolor='white',edgecolor='black')
leyenda.get_frame().set_linewidth(1.0)
pivot_postulaciones_zona_top_nivel=postulaciones_zona_top_tipo_nivel.pivot_table(index='Zona',columns='Nivel_Laboral',values='Valor',aggfunc='sum')
pivot_postulaciones_zona_top_nivel
grafico_postulaciones_zona_top_nivel=pivot_postulaciones_zona_top_nivel.plot(kind='bar',figsize=(8,8),fontsize=12,rot=0)
grafico_postulaciones_zona_top_nivel.set_title("Postulaciones según Zona por Nivel Laboral",fontsize=20)
grafico_postulaciones_zona_top_nivel.set_xlabel("Zona",fontsize=12)
grafico_postulaciones_zona_top_nivel.set_ylabel("Cantidad de Postulaciones",fontsize=12)
leyenda=plt.legend(['Gerencia / Alta Gerencia / Dirección','Jefe / Supervisor / Responsable','Junior','Otro','Senior / Semi-Senior'],fontsize=12,title='Nivel Laboral',frameon=True,facecolor='white',edgecolor='black')
leyenda.get_frame().set_linewidth(1.0)
genero_usuarios=genero_edad['Sexo'].value_counts()
genero_usuarios
generos = ['Femenino', 'Masculino','No Declara']
grafico_genero_usuarios=genero_usuarios.plot(kind='pie',labels=generos,figsize=(7,7),fontsize=12,colors=['salmon','royalblue','tomato'],autopct='%1.1f%%')
grafico_genero_usuarios.set_title("Cantidad de Usuarios registrados segun Genero",fontsize=20)
grafico_genero_usuarios.set_ylabel("")
postulaciones_genero=pd.merge(postulaciones,genero_edad,on='idpostulante',how='inner')
cantidad_postulaciones_genero=postulaciones_genero['Sexo'].value_counts()
cantidad_postulaciones_genero.rename(index={'FEM':'Femenino','MASC':'Masculino','NO_DECLARA':'No Declara'},inplace=True)
cantidad_postulaciones_genero
pivot_cantidad_postulaciones_genero=cantidad_postulaciones_genero.plot(kind='bar',figsize=(8,8),color=['salmon','royalblue','tomato'],fontsize=12,rot=0)
pivot_cantidad_postulaciones_genero.set_title("Cantidad de Postulaciones por Genero",fontsize=20)
pivot_cantidad_postulaciones_genero.set_xlabel("Genero",fontsize=12)
pivot_cantidad_postulaciones_genero.set_ylabel("Cantidad de Postulaciones",fontsize=12)
postulaciones_edad_filtado=postulaciones_genero[(postulaciones_genero['Edad']>=18) & (postulaciones_genero['Edad']<=66)]
group_postulaciones_edad_filtado=postulaciones_edad_filtado.groupby(['Sexo']).agg({'Edad':'mean'})
group_postulaciones_edad_filtado.rename(index={'FEM':'Femenino','MASC':'Masculino','NO_DECLARA':'No Declara'},inplace=True)
group_postulaciones_edad_filtado
grafico_postulaciones_edad_filtado=sns.barplot(x=group_postulaciones_edad_filtado['Edad'],y=group_postulaciones_edad_filtado.index,palette="YlGnBu")
grafico_postulaciones_edad_filtado.set_title("Edad General de las Postulaciones",fontsize=20)
grafico_postulaciones_edad_filtado.set_xlabel("Edad",fontsize=12)
grafico_postulaciones_edad_filtado.set_ylabel("Genero",fontsize=12)
edad_fem=edad[(edad['Sexo']=='FEM')]
edad_masc=edad[(edad['Sexo']=='MASC')]
cantidad_edad_fem=edad_fem['Edad'].value_counts().sort_index()
cantidad_edad_masc=edad_masc['Edad'].value_counts().sort_index()
grafico_cantidad_edad_fem=cantidad_edad_fem.plot(kind='bar',figsize=(9,9),fontsize=12,color='salmon')
grafico_cantidad_edad_fem.set_title("Edades de Registrados (Femenino)",fontsize=20)
grafico_cantidad_edad_fem.set_xlabel("Edades",fontsize=12)
grafico_cantidad_edad_fem.set_ylabel("Cantidad de Usuarios",fontsize=12)
grafico_cantidad_edad_masc=cantidad_edad_masc.plot(kind='bar',figsize=(9,9),fontsize=12,color='royalblue')
grafico_cantidad_edad_masc.set_title("Edades de Registrados (Masculino)",fontsize=20)
grafico_cantidad_edad_masc.set_xlabel("Edades",fontsize=12)
grafico_cantidad_edad_masc.set_ylabel("Cantidad de Usuarios",fontsize=12)
educacion['Valor']=1
pivot_educacion=educacion.pivot_table(index='Nivel',columns='Estado',values='Valor',aggfunc='sum')
pivot_educacion
plt.subplots(figsize=(8,8))
grafico_pivot_educacion=sns.heatmap(pivot_educacion,linewidths=.5,fmt="d",annot=True,cmap="vlag_r")
grafico_pivot_educacion.set_title("Cantidad de Usuarios por Nivel de Educacion segun Estado",fontsize=20)
grafico_pivot_educacion.set_xlabel("Estado",fontsize=12)
grafico_pivot_educacion.set_ylabel("Nivel",fontsize=12)
educacion_fem=pd.merge(educacion,edad_fem,on='idpostulante',how='inner')
educacion_masc=pd.merge(educacion,edad_masc,on='idpostulante',how='inner')
pivot_educacion_fem=educacion_fem.pivot_table(index='Nivel',columns='Estado',values='Valor',aggfunc='sum')
pivot_educacion_masc=educacion_masc.pivot_table(index='Nivel',columns='Estado',values='Valor',aggfunc='sum')
pivot_educacion_fem
plt.subplots(figsize=(8,8))
grafico_pivot_educacion_fem=sns.heatmap(pivot_educacion_fem,linewidths=.5,fmt="d",annot=True,cmap="gist_ncar")
grafico_pivot_educacion_fem.set_title("Cantidad de Usuarios por Nivel de Educacion segun Estado (Femenino)",fontsize=16)
grafico_pivot_educacion_fem.set_xlabel("Estado",fontsize=12)
grafico_pivot_educacion_fem.set_ylabel("Nivel",fontsize=12)
pivot_educacion_masc
plt.subplots(figsize=(8,8))
grafico_pivot_educacion_masc=sns.heatmap(pivot_educacion_masc,linewidths=.5,fmt="d",annot=True,cmap="nipy_spectral_r")
grafico_pivot_educacion_masc.set_title("Cantidad de Usuarios por Nivel de Educacion segun Estado (Masculino)",fontsize=16)
grafico_pivot_educacion_masc.set_xlabel("Estado",fontsize=12)
grafico_pivot_educacion_masc.set_ylabel("Nivel",fontsize=12)
educacion_edad=pd.merge(educacion,edad,on='idpostulante',how='inner')
pivot_educacion_edad=educacion_edad.pivot_table(index='Sexo',columns='Estado',values='Edad',aggfunc='mean')
pivot_educacion_edad.rename(index={'FEM':'Femenino','MASC':'Masculino'},inplace=True)
pivot_educacion_edad
grafico_pivot_educacion_edad=pivot_educacion_edad.plot(kind='bar',figsize=(8,8),fontsize=12,color=['red','green','blue'],rot=0)
grafico_pivot_educacion_edad.set_title("Edad Promedio segun Estado de Educacion",fontsize=20)
grafico_pivot_educacion_edad.set_xlabel("Genero",fontsize=12)
grafico_pivot_educacion_edad.set_ylabel("Edad",fontsize=12)
leyenda=plt.legend(["Abandonado","En Curso","Graduado"],fontsize=12,title='Nivel de Educacion',frameon=True,facecolor='white',edgecolor='black',bbox_to_anchor=(1.05, 1))
leyenda.get_frame().set_linewidth(1.0)
genero_filtrado=genero[(genero['Sexo'] != 'NO_DECLARA')]
educacion_genero=pd.merge(educacion,genero_filtrado,on='idpostulante',how='inner')
educacion_genero_graduado=educacion_genero[(educacion_genero['Estado']=='Graduado')]
educacion_genero_cursando=educacion_genero[(educacion_genero['Estado']=='En Curso')]
pivot_educacion_genero_graduado=educacion_genero_graduado.pivot_table(index='Nivel',columns='Sexo',values='idpostulante',aggfunc='count')
pivot_educacion_genero_cursando=educacion_genero_cursando.pivot_table(index='Nivel',columns='Sexo',values='idpostulante',aggfunc='count')
pivot_educacion_genero_graduado
grafico_pivot_educacion_genero_graduado=pivot_educacion_genero_graduado.plot(kind='bar',figsize=(8,8),fontsize=12,color=['salmon','royalblue'],rot=70)
grafico_pivot_educacion_genero_graduado.set_title("Cantidad de Usuarios Graduados por Nivel Educativo segun Genero",fontsize=16)
grafico_pivot_educacion_genero_graduado.set_xlabel("Nivel de Educacion",fontsize=12)
grafico_pivot_educacion_genero_graduado.set_ylabel("Cantidad de Usuarios",fontsize=12)
leyenda=plt.legend(['Femenino','Masculino'],fontsize=12,title='Genero',frameon=True,facecolor='white',edgecolor='black')
leyenda.get_frame().set_linewidth(1.0)
pivot_educacion_genero_cursando
grafico_pivot_educacion_genero_cursando=pivot_educacion_genero_cursando.plot(kind='bar',figsize=(8,8),fontsize=12,color=['salmon','royalblue'],rot=70)
grafico_pivot_educacion_genero_cursando.set_title("Cantidad de Usuarios Cursando por Nivel Educativo segun Genero",fontsize=16)
grafico_pivot_educacion_genero_cursando.set_xlabel("Nivel de Educacion",fontsize=12)
grafico_pivot_educacion_genero_cursando.set_ylabel("Cantidad de Usuarios",fontsize=12)
leyenda=plt.legend(['Femenino','Masculino'],fontsize=12,title='Genero',frameon=True,facecolor='white',edgecolor='black')
leyenda.get_frame().set_linewidth(1.0)
df_postulantes_educacion = pd.read_csv("../input/fiuba_1_postulantes_educacion.csv", low_memory=False)
df_postulantes_genero_y_edad = pd.read_csv("../input/fiuba_2_postulantes_genero_y_edad.csv", low_memory=False)
df_vistas = pd.read_csv("../input/fiuba_3_vistas.csv", low_memory=False)
df_postulaciones = pd.read_csv("../input/fiuba_4_postulaciones.csv", low_memory=False)
df_aviso_online = pd.read_csv("../input/fiuba_5_avisos_online.csv", low_memory=False)
df_avisos_detalle = pd.read_csv("../input/fiuba_6_avisos_detalle.csv", low_memory=False)
datos = df_avisos_detalle['tipo_de_trabajo'].value_counts()
g = sns.barplot(y=datos.index, x=datos.values, palette=sns.color_palette("Set2", 10))
g.set_title('Cantidad de avisos segun tipo de trabajo', fontsize=18);
g.set_xlabel('Cantidad de avisos');
g.set_ylabel('Tipo de trabajo');
g = sns.barplot(y=datos.index, x=datos.values, palette=sns.color_palette("Set2", 10))
g.set_title('Cantidad de avisos segun tipo de trabajo (Resto)', fontsize=18);
g.set_xlabel('Cantidad de avisos');
g.set_ylabel('Tipo de trabajo');
df_vistas.rename(columns={'idAviso': 'idaviso'}, inplace=True)
df_avisos_detalle_vistas = pd.merge(df_avisos_detalle, df_vistas, on='idaviso', how='inner')
df_tipos_vistas = pd.DataFrame(df_avisos_detalle_vistas['tipo_de_trabajo'].value_counts())
df_tipos_vistas.reset_index(inplace=True)
df_avisos_detalle_postulaciones = pd.merge(df_avisos_detalle, df_postulaciones, on='idaviso', how='inner')
df_tipos_postulaciones = pd.DataFrame(df_avisos_detalle_postulaciones['tipo_de_trabajo'].value_counts())
df_tipos_postulaciones.reset_index(inplace=True)
df_tipos_postulaciones.rename(columns={'index':'tipo', 'tipo_de_trabajo':'postulaciones'}, inplace=True)
df_tipos_vistas.rename(columns={'index':'tipo', 'tipo_de_trabajo':'vistas'}, inplace=True)
df_tipo_postulaciones_vistas = pd.merge(df_tipos_postulaciones, df_tipos_vistas, on='tipo', how='inner')
df_tipo_postulaciones_vistas.set_index('tipo',inplace=True)
g = df_tipo_postulaciones_vistas.loc[['Full-time', 'Part-time']].plot(kind='bar')
g.set_title('Cantidad de vistas y postulaciones segun tipo de trabajo', fontsize=18);
g.set_ylabel('Cantidad');
g.set_xlabel('Tipo de trabajo')
leyenda=plt.legend(['Full-Time','Part-Time'],fontsize=12,title='Tipo de Trabajo',frameon=True,facecolor='white',edgecolor='black')
leyenda.get_frame().set_linewidth(1.0)
g = df_tipo_postulaciones_vistas.loc['Pasantia':].plot(kind='bar')
g.set_title('Cantidad de vistas y postulaciones segun tipo de trabajo', fontsize=18);
g.set_ylabel('Cantidad');
g.set_xlabel('Tipo de trabajo');
leyenda=plt.legend(['Full-Time','Part-Time'],fontsize=12,title='Tipo de Trabajo',frameon=True,facecolor='white',edgecolor='black')
leyenda.get_frame().set_linewidth(1.0)
df_avisos_detalle_ft = df_avisos_detalle[df_avisos_detalle['tipo_de_trabajo'] == 'Full-time']
df_avisos_detalle_pt = df_avisos_detalle[df_avisos_detalle['tipo_de_trabajo'] == 'Part-time']
datos = df_avisos_detalle_ft['nombre_area'].value_counts().head(10)
g = sns.barplot(y=datos.index, x=datos.values, palette=sns.color_palette('Reds_r', 15))
g.set_title('Ranking de cantidad de avisos Full-time', fontsize=18);
g.set_xlabel('Cantidad de avisos');
g.set_ylabel('Area');
datos = df_avisos_detalle_pt['nombre_area'].value_counts().head(10)
g = sns.barplot(y=datos.index, x=datos.values, palette=sns.color_palette('Blues_r', 15))
g.set_title('Ranking de cantidad de avisos Part-time', fontsize=18);
g.set_xlabel('Cantidad de avisos');
g.set_ylabel('Area');
datos = df_avisos_detalle[df_avisos_detalle['tipo_de_trabajo'] == 'Teletrabajo']['nombre_area'].value_counts().head(10)
g = sns.barplot(y=datos.index, x=datos.values, palette=sns.color_palette('Oranges_r', 15))
g.set_title('Ranking de cantidad de avisos Teletrabajo', fontsize=18);
g.set_xlabel('Cantidad de avisos');
g.set_ylabel('Area');
datos = df_avisos_detalle[df_avisos_detalle['tipo_de_trabajo'] == 'Pasantia']['nombre_area'].value_counts().head(10)
g = sns.barplot(y=datos.index, x=datos.values, palette=sns.color_palette('Purples_r', 15))
g.set_title('Ranking de cantidad de avisos Pasantia', fontsize=18);
g.set_xlabel('Cantidad de avisos');
g.set_ylabel('Area');
area = 'Por Horas'
datos = df_avisos_detalle[df_avisos_detalle['tipo_de_trabajo'] == area]['nombre_area'].value_counts().head(10)
g = sns.barplot(y=datos.index, x=datos.values, palette=sns.color_palette('afmhot', 20))
g.set_title('Ranking de cantidad de avisos {}'.format(area), fontsize=18);
g.set_xlabel('Cantidad de avisos');
g.set_ylabel('Area');
area = 'Temporario'
datos = df_avisos_detalle[df_avisos_detalle['tipo_de_trabajo'] == area]['nombre_area'].value_counts().head(10)
g = sns.barplot(y=datos.index, x=datos.values, palette=sns.color_palette('Greens_r', 15))
g.set_title('Ranking de cantidad de avisos {}'.format(area), fontsize=18);
g.set_xlabel('Cantidad de avisos');
g.set_ylabel('Area');
area = 'Por Contrato'
datos = df_avisos_detalle[df_avisos_detalle['tipo_de_trabajo'] == area]['nombre_area'].value_counts().head(10)
g = sns.barplot(y=datos.index, x=datos.values, palette=sns.color_palette('pink', 15))
g.set_title('Ranking de cantidad de avisos {}'.format(area), fontsize=18);
g.set_xlabel('Cantidad de avisos');
g.set_ylabel('Area');
df_avisos_no_pt_ft = df_avisos_detalle[-df_avisos_detalle['tipo_de_trabajo'].isin(['Full-time', 'Part-time'])]
df_avisos_no_pt_ft['nivel'] = 1
df_areas_ppales = df_avisos_no_pt_ft[
    df_avisos_no_pt_ft['nombre_area'].isin(['Ventas', 'Tecnología', 'Comercial', 'Legal', 'Administración', 
                                            'Producción', 'Medicina', 'Contabilidad', 'Construcción', 'Sistemas',
                                            'Programación', 'Transporte', 'Salud'])
    | (df_avisos_no_pt_ft['nombre_area'].str.contains('Industrial'))
    | (df_avisos_no_pt_ft['nombre_area'].str.contains('Trainee'))
]
tabla = df_areas_ppales[['nombre_area', 'tipo_de_trabajo', 'nivel']].pivot_table(
    index='nombre_area', columns='tipo_de_trabajo', values='nivel', aggfunc='sum'
)
tabla
g = sns.heatmap(tabla, cmap='magma_r', linewidths=0.5)
g.set_title('Tipos de trabajo segun distintas areas');
g.set_ylabel('Area');
g.set_xlabel('Tipo de trabajo');
# Filtro las edades
df_postulantes_genero_y_edad['fechanacimiento_dt'] = pd.to_datetime(df_postulantes_genero_y_edad['fechanacimiento'],
                                                                   errors='coerce')
df_postulantes_genero_y_edad['edad'] = 2018 - df_postulantes_genero_y_edad['fechanacimiento_dt'].dt.year
df_postulantes_genero_y_edad = df_postulantes_genero_y_edad[
    (df_postulantes_genero_y_edad['edad'] >= 18)
    & (df_postulantes_genero_y_edad['edad'] <= 65)
]
df_postulantes_genero_y_edad =df_postulantes_genero_y_edad[df_postulantes_genero_y_edad['sexo'] != 'NO_DECLARA']
df_postulaciones_sexo = pd.merge(df_postulantes_genero_y_edad, df_postulaciones,
                                on='idpostulante', how='inner')
df_avisos_sexo = pd.merge(df_postulaciones_sexo, df_avisos_detalle, on='idaviso', how='inner')
datos_pt_ft = df_avisos_sexo[df_avisos_sexo['tipo_de_trabajo'].isin(['Full-time', 'Part-time'])]
datos_no_pt_ft = df_avisos_sexo[-df_avisos_sexo['tipo_de_trabajo'].isin(['Full-time', 'Part-time'])]
g = sns.boxplot(x='tipo_de_trabajo', y='edad', data=df_avisos_sexo)
g.set_title('Edad segun tipo de trabajo', fontsize=18);
g.set_ylabel('Edad');
g.set_xlabel('Tipo de trabajo');
g.set_xticklabels(g.get_xticklabels(), rotation=45);
g = sns.boxplot(x='tipo_de_trabajo', hue='sexo', y='edad', data=df_avisos_sexo, palette=['salmon','royalblue'])
g.set_title('Edad segun tipo de trabajo', fontsize=18);
g.set_ylabel('Edad');
g.set_xlabel('Tipo de trabajo');
g.set_xticklabels(g.get_xticklabels(), rotation=45);
datos = df_avisos_detalle_ft['denominacion_empresa'].value_counts().head(10)
g = sns.barplot(y=datos.index, x=datos.values, palette=sns.color_palette('Blues_r', 15))
g.set_title('Ranking de cantidad de avisos Full-time', fontsize=18);
g.set_xlabel('Cantidad de avisos Full-time');
g.set_ylabel('Empresa');
datos = df_avisos_detalle_pt['denominacion_empresa'].value_counts().head(10)
g = sns.barplot(y=datos.index, x=datos.values, palette=sns.color_palette('Reds_r',15))
g.set_title('Ranking de cantidad de avisos Part-time', fontsize=18);
g.set_xlabel('Cantidad de avisos Part-time');
g.set_ylabel('Empresa');
tipo = 'Teletrabajo'
datos = df_avisos_detalle[df_avisos_detalle['tipo_de_trabajo'] == tipo]['denominacion_empresa'].value_counts().head(10)
g = sns.barplot(y=datos.index, x=datos.values, palette=sns.color_palette('Purples_r', 15))
g.set_title('Ranking de cantidad de avisos de {}'.format(tipo), fontsize=18);
g.set_xlabel('Cantidad de avisos {}'.format(tipo));
g.set_ylabel('Empresa');
tipo = 'Por Contrato'
datos = df_avisos_detalle[df_avisos_detalle['tipo_de_trabajo'] == tipo]['denominacion_empresa'].value_counts().head(10)
g = sns.barplot(y=datos.index, x=datos.values, palette=sns.color_palette('Greens_r', 15))
g.set_title('Ranking de cantidad de avisos de {}'.format(tipo), fontsize=18);
g.set_xlabel('Cantidad de avisos {}'.format(tipo));
g.set_ylabel('Empresa');
tipo = 'Por Horas'
datos = df_avisos_detalle[df_avisos_detalle['tipo_de_trabajo'] == tipo]['denominacion_empresa'].value_counts().head(10)
g = sns.barplot(y=datos.index, x=datos.values, palette=sns.color_palette('Oranges_r', 15))
g.set_title('Ranking de cantidad de avisos de {}'.format(tipo), fontsize=18);
g.set_xlabel('Cantidad de avisos {}'.format(tipo));
g.set_ylabel('Empresa');
tipo = 'Temporario'
datos = df_avisos_detalle[df_avisos_detalle['tipo_de_trabajo'] == tipo]['denominacion_empresa'].value_counts().head(10)
g = sns.barplot(y=datos.index, x=datos.values, palette=sns.color_palette('Spectral_r', 15) )
g.set_title('Ranking de cantidad de avisos de {}'.format(tipo), fontsize=18);
g.set_xlabel('Cantidad de avisos {}'.format(tipo));
g.set_ylabel('Empresa');
tipo = 'Pasantia'
datos = df_avisos_detalle[df_avisos_detalle['tipo_de_trabajo'] == tipo]['denominacion_empresa'].value_counts().head(10)
g = sns.barplot(y=datos.index, x=datos.values, palette=sns.color_palette('cubehelix', 15))
g.set_title('Ranking de cantidad de avisos de {}'.format(tipo), fontsize=18);
g.set_xlabel('Cantidad de avisos {}'.format(tipo));
g.set_ylabel('Empresa');
tipo = 'Fines de Semana'
datos = df_avisos_detalle[df_avisos_detalle['tipo_de_trabajo'] == tipo]['denominacion_empresa'].value_counts().head(10)
g = sns.barplot(y=datos.index, x=datos.values, palette='bwr')
g.set_title('Ranking de cantidad de avisos de {}'.format(tipo), fontsize=18);
g.set_xlabel('Cantidad de avisos {}'.format(tipo));
g.set_ylabel('Empresa');
datos = df_avisos_detalle[(df_avisos_detalle['tipo_de_trabajo'] == 'Fines de Semana')]['titulo'].value_counts()
g = sns.barplot(y=datos.index, x=datos.values)
tipo = 'Primer empleo'
datos = df_avisos_detalle[df_avisos_detalle['tipo_de_trabajo'] == tipo]['denominacion_empresa'].value_counts().head(10)
g = sns.barplot(y=datos.index, x=datos.values)
g.set_title('Ranking de cantidad de avisos de {}'.format(tipo), fontsize=18);
g.set_xlabel('Cantidad de avisos {}'.format(tipo));
g.set_ylabel('Empresa');
df_avisos_detalle = df_avisos_detalle[-df_avisos_detalle['denominacion_empresa'].isnull()]
datos = df_avisos_detalle['denominacion_empresa'].value_counts().head(10)
g = sns.barplot(y=datos.index, x=datos.values, palette=sns.color_palette("Purples_r", 15));
g.set_title('Cantidad de avisos por empresa', fontsize=18);
g.set_xlabel('Cantidad de avisos');
g.set_ylabel('Empresa');
df_avisos_detalle['denominacion_empresa'].describe()
g = (np.log(df_avisos_detalle['denominacion_empresa'].value_counts() + 1)).plot.hist(bins=20)
g.set_title('Histograma de cantidad de avisos por empresa', fontsize=18);
g.set_ylabel('Frecuencia');
g.set_xlabel('Cantidad de avisos por empresa');
df_avisos_ventas = df_avisos_detalle[df_avisos_detalle['nombre_area'] == 'Ventas']
datos = df_avisos_ventas['denominacion_empresa'].value_counts().head(10)
g = sns.barplot(x=datos.values, y=datos.index, palette=sns.color_palette("Greens_r", 15))
g.set_title('Cantidad de avisos de venta por empresa', fontsize=18);
g.set_xlabel('Cantidad de avisos de venta');
g.set_ylabel('Empresa');
df_avisos_ventas = df_avisos_detalle[df_avisos_detalle['nombre_area'] == 'Comercial']
datos = df_avisos_ventas['denominacion_empresa'].value_counts().head(10)
g = sns.barplot(x=datos.values, y=datos.index, palette=sns.color_palette("Reds_r", 15))
g.set_title('Cantidad de avisos comerciales por empresa', fontsize=18);
g.set_xlabel('Cantidad de avisos comerciales');
g.set_ylabel('Empresa');
datos = df_avisos_detalle[df_avisos_detalle['denominacion_empresa'].str.contains('Swiss Medical')]['nombre_area'].value_counts().head(10)
g = sns.barplot(y=datos.index, x=datos.values)
g.set_title('Cantidad de avisos de Swiss Medical Group segun area', fontsize=18);
g.set_xlabel('Cantidad de avisos');
g.set_ylabel('Areas');
df_avisos_ventas = df_avisos_detalle[df_avisos_detalle['nombre_area'] == 'Administración']
datos = df_avisos_ventas['denominacion_empresa'].value_counts().head(10)
g = sns.barplot(x=datos.values, y=datos.index, palette=sns.color_palette("Blues_r", 15))
g.set_title('Cantidad de avisos de administracion por empresa', fontsize=18);
g.set_xlabel('Cantidad de avisos de administracion');
g.set_ylabel('Empresa');
df_avisos_ventas = df_avisos_detalle[df_avisos_detalle['nombre_area'] == 'Producción']
datos = df_avisos_ventas['denominacion_empresa'].value_counts().head(10)
g = sns.barplot(x=datos.values, y=datos.index, palette=sns.color_palette("magma", 15))
g.set_title('Cantidad de avisos de produccion por empresa', fontsize=18);
g.set_xlabel('Cantidad de avisos de produccion');
g.set_ylabel('Empresa');
df_avisos_ventas = df_avisos_detalle[df_avisos_detalle['nombre_area'] == 'Programación']
datos = df_avisos_ventas['denominacion_empresa'].value_counts().head(10)
g = sns.barplot(x=datos.values, y=datos.index, palette=sns.color_palette("YlOrBr_r", 15))
g.set_title('Cantidad de avisos de programación por empresa', fontsize=18);
g.set_xlabel('Cantidad de avisos de programación');
g.set_ylabel('Empresa');
df_avisos_ventas = df_avisos_detalle[df_avisos_detalle['nombre_area'] == 'Atención al Cliente']
datos = df_avisos_ventas['denominacion_empresa'].value_counts().head(10)
g = sns.barplot(x=datos.values, y=datos.index, palette=sns.color_palette("PiYG", 10))
g.set_title('Cantidad de avisos de atencion al cliente por empresa', fontsize=18);
g.set_xlabel('Cantidad de avisos de atencion al cliente');
g.set_ylabel('Empresa');
df_avisos_ventas = df_avisos_detalle[df_avisos_detalle['nombre_area'] == 'Call Center']
datos = df_avisos_ventas['denominacion_empresa'].value_counts().head(10)
g = sns.barplot(x=datos.values, y=datos.index, palette=sns.color_palette("RdYlBu", 10))
g.set_title('Cantidad de avisos de call center por empresa', fontsize=18);
g.set_xlabel('Cantidad de avisos de call center');
g.set_ylabel('Empresa');
df_avisos_detalle_empresas = df_avisos_detalle[
    df_avisos_detalle['denominacion_empresa'].str.contains('RANDSTAD') 
    | df_avisos_detalle['denominacion_empresa'].str.contains('Manpower') 
    | df_avisos_detalle['denominacion_empresa'].str.contains('Adecco')
    | df_avisos_detalle['denominacion_empresa'].str.contains('SOLUTIX')
    | df_avisos_detalle['denominacion_empresa'].str.contains('Pullmen')
    | df_avisos_detalle['denominacion_empresa'].str.contains('Grupo Gestión') 
    | df_avisos_detalle['denominacion_empresa'].str.contains('Excelencia Laboral') 
    | df_avisos_detalle['denominacion_empresa'].str.contains('Kaizen') 
]
df_avisos_detalle_empresas_areas = df_avisos_detalle_empresas[
    df_avisos_detalle_empresas['nombre_area'].isin([
      'Ventas', 'Comercial', 'Administración', 'Producción', 'Atención al Cliente', 'Call Center', 'Programación'  
    ])
]
datos = df_avisos_detalle_empresas_areas[['nombre_area', 'denominacion_empresa']]
datos['empresa'] = datos['denominacion_empresa'].map(lambda x: 'Addeco' if ('Adecco' in x) else ('Manpower' if ('Manpower' in x) else x))
datos = datos[['nombre_area', 'empresa']]
datos['nivel'] =1
tabla = datos.pivot_table(index='empresa', columns='nombre_area', values='nivel', aggfunc='sum')
g = sns.heatmap(tabla, cmap='Reds', linewidths=.5)
g.set_title('Cantidad de avisos de empresa segun area', fontsize=18);
g.set_xlabel('Area');
g.set_ylabel('Empresa');
df_avisos_detalle_senior = df_avisos_detalle[df_avisos_detalle['nivel_laboral'].str.contains('Senior')]
df_avisos_detalle_junior = df_avisos_detalle[df_avisos_detalle['nivel_laboral'].str.contains('Junior')]
df_avisos_detalle_jefe = df_avisos_detalle[df_avisos_detalle['nivel_laboral'].str.contains('Jefe')]
df_avisos_detalle_gerencia = df_avisos_detalle[df_avisos_detalle['nivel_laboral'].str.contains('Gerencia')]
datos = df_avisos_detalle_junior['denominacion_empresa'].value_counts().head(10)
g = sns.barplot(y=datos.index, x=datos.values, palette=sns.color_palette("BuGn_r", 15))
g.set_title('Cantidad de avisos nivel Junior segun empresa', fontsize=18);
g.set_xlabel('Cantidad de avisos');
g.set_ylabel('Empresa');
datos = df_avisos_detalle_senior['denominacion_empresa'].value_counts().head(10)
g = sns.barplot(y=datos.index, x=datos.values, palette=sns.color_palette("GnBu_d",15))
g.set_title('Cantidad de avisos nivel Senior segun empresa', fontsize=18);
g.set_xlabel('Cantidad de avisos');
g.set_ylabel('Empresa');
datos = df_avisos_detalle_jefe['denominacion_empresa'].value_counts().head(10)
g = sns.barplot(y=datos.index, x=datos.values, palette=sns.cubehelix_palette(10, reverse=True))
g.set_title('Cantidad de avisos nivel Jefe segun empresa', fontsize=18);
g.set_xlabel('Cantidad de avisos');
g.set_ylabel('Empresa');
datos = df_avisos_detalle_gerencia['denominacion_empresa'].value_counts().head(10)
g = sns.barplot(y=datos.index, x=datos.values, palette=sns.color_palette("OrRd_d",15))
g.set_title('Cantidad de avisos nivel Gerencia segun empresa', fontsize=18);
g.set_xlabel('Cantidad de avisos');
g.set_ylabel('Empresa');
df_postulaciones_avisos = pd.merge(df_postulaciones, df_avisos_detalle, on='idaviso', how='inner')
datos_postulaciones = df_postulaciones_avisos['denominacion_empresa'].value_counts().head(10)
g = sns.barplot(x=datos_postulaciones.values, y=datos_postulaciones.index, palette=sns.color_palette("PuBuGn_r",15))
g.set_title('Cantidad de postulaciones a empresas', fontsize=18);
g.set_xlabel('Cantidad de postulaciones');
g.set_ylabel('Empresa');
df_vistas.rename(columns={'idAviso':'idaviso'}, inplace=True)
df_vistas_avisos = pd.merge(df_vistas, df_avisos_detalle, on='idaviso', how='inner')
datos_vistas = df_vistas_avisos['denominacion_empresa'].value_counts().head(10)
g = sns.barplot(x=datos_vistas.values, y=datos_vistas.index, palette=sns.color_palette("RdYlBu_r",15))
g.set_title('Cantidad de vistas a empresas', fontsize=18);
g.set_xlabel('Cantidad de vistas');
g.set_ylabel('Empresa');
datos_vistas = df_vistas_avisos['denominacion_empresa'].value_counts()
datos_postulaciones = df_postulaciones_avisos['denominacion_empresa'].value_counts()
df_datos_postulaciones = pd.DataFrame(datos_postulaciones)
df_datos_vistas = pd.DataFrame(datos_vistas)
df_datos_postulaciones.reset_index(inplace=True)
df_datos_postulaciones.rename(columns={'index':'empresa', 'denominacion_empresa':'postulaciones'}, inplace=True)
df_datos_vistas.reset_index(inplace=True)
df_datos_vistas.rename(columns={'index':'empresa', 'denominacion_empresa':'vistas'}, inplace=True)
datos = pd.merge(df_datos_postulaciones, df_datos_vistas, on='empresa', how='inner')
datos['ratio'] = datos['postulaciones'] / datos['vistas']
g = sns.regplot('postulaciones', 'vistas', data=datos, scatter_kws={'alpha':0.5})
g.set_title('Postulaciones vs Vistas por empresa', fontsize=18);
g.set_xlabel('Postulaciones');
g.set_ylabel('Vistas');
datos[(datos['postulaciones']>1000) & (datos['vistas'] > 1000)].sort_values('ratio', ascending=True)
df_avisos_detalle['nombre_zona'].value_counts()
df_avisos_capital = df_avisos_detalle[df_avisos_detalle['nombre_zona'] == 'Capital Federal']
df_avisos_GBA = df_avisos_detalle[df_avisos_detalle['nombre_zona'] == 'Gran Buenos Aires']
datos_capital = df_avisos_capital['denominacion_empresa'].value_counts().head(10)
datos_GBA = df_avisos_GBA['denominacion_empresa'].value_counts().head(10)
g = sns.barplot(y=datos_capital.index, x=datos_capital.values)
g.set_title('Cantidad de avisos segun empresa en Capital Federal', fontsize=18);
g.set_xlabel('Cantidad de avisos');
g.set_ylabel('Empresa');
g = sns.barplot(y=datos_GBA.index, x=datos_GBA.values)
g.set_title('Cantidad de avisos segun empresa en Gran Buenos Aires', fontsize=18);
g.set_xlabel('Cantidad de avisos');
g.set_ylabel('Empresa');
df_mujeres = df_postulantes_genero_y_edad[df_postulantes_genero_y_edad['sexo'] == 'FEM']
df_hombres = df_postulantes_genero_y_edad[df_postulantes_genero_y_edad['sexo'] == 'MASC']
df_postulaciones_mujeres = pd.merge(df_mujeres, df_postulaciones, on='idpostulante', how='inner')
df_postulaciones_hombres = pd.merge(df_hombres, df_postulaciones, on='idpostulante', how='inner')
df_avisos_detalle_mujeres = pd.merge(df_avisos_detalle, df_postulaciones_mujeres, on='idaviso', how='inner')
df_avisos_detalle_hombres = pd.merge(df_avisos_detalle, df_postulaciones_hombres, on='idaviso', how='inner')
ranking_mujeres = df_avisos_detalle_mujeres['denominacion_empresa'].value_counts().head(10)
ranking_hombres = df_avisos_detalle_hombres['denominacion_empresa'].value_counts().head(10)
g = sns.barplot(y=ranking_mujeres.index, x=ranking_mujeres.values, palette=sns.color_palette("PuRd_r",15))
g.set_title('Cantidad de postulaciones de mujeres a empresas', fontsize=18);
g.set_xlabel('Cantidad de postulaciones');
g.set_ylabel('Empresas');
df_avisos_detalle_farmacity_BBVA = df_avisos_detalle[
    (df_avisos_detalle['denominacion_empresa'] == 'Farmacity')
    | (df_avisos_detalle['denominacion_empresa'] == 'BBVA Francés')                                           
]
df_postulaciones_sexo = pd.merge(df_postulantes_genero_y_edad, df_postulaciones, on='idpostulante', how='inner')
df_postulaciones_sexo = df_postulaciones_sexo[df_postulaciones_sexo['sexo'] != 'NO_DECLARA']
df_avisos_detalle_farmacity_BBVA_sexo = pd.merge(df_avisos_detalle_farmacity_BBVA, df_postulaciones_sexo, 
                                                 on='idaviso', how='inner')
df_avisos_detalle_farmacity_BBVA_sexo.head(1)
g = sns.countplot(hue='sexo', x='denominacion_empresa', data=df_avisos_detalle_farmacity_BBVA_sexo,
                  palette=['Blue', 'Red'])
g.set_title('Cantidad de postulaciones a Farmacity y BBVA Frances', fontsize=18);
g.set_xlabel('Empresa');
g.set_ylabel('Cantidad de postulaciones');
g = sns.barplot(y=ranking_hombres.index, x=ranking_hombres.values,palette=sns.color_palette("GnBu_r",15) )
g.set_title('Cantidad de postulaciones de hombres a empresas', fontsize=18);
g.set_xlabel('Cantidad de postulaciones');
g.set_ylabel('Empresas');
#Agrego la columna edad al DF de genero y edad
hoy = pd.Timestamp(DT.datetime.now())
genero_edad['Edad'] = (hoy - genero_edad['Fecha_Nacimiento']).astype('<m8[Y]')
genero_edad=genero_edad[(genero_edad['Edad']>=18) & (genero_edad['Edad']<=66)]
#realizamos un merge entre los df de postulaciones y de area, para tener en un solo df las fechas de vistas
#y la de postulacion. 

vistas_postulaciones_merged = pd.merge(postulaciones,vistas, on = ['idaviso','idpostulante'], how='inner')
vistas_postulaciones_merged.head()
#Aqui obtenemos un DF cuyas columnas son el id del postulante y el otro la cantidad de vistas promedio por cada aviso
#al que realizaron una postulacion. 

vistas_por_postulacion = vistas_postulaciones_merged.groupby(['idpostulante','idaviso'])\
                    .agg({'Fecha_Postulacion':'count'})\
                    .groupby('idpostulante').agg({'Fecha_Postulacion':'mean'})
        
vistas_por_postulacion.rename(columns = {'Fecha_Postulacion':'vistas/postulacion'}, inplace = True)
vistas_por_postulacion.reset_index(inplace = True)
vistas_por_postulacion.head()
#Aqui obtenemos un DF en el que se ve la Edad del postulante y su promedio de vistas por postulacion

vistas_por_postulacion_edad = pd.merge(genero_edad,vistas_por_postulacion,on='idpostulante',how = 'inner')[['Edad','vistas/postulacion']]
vistas_por_postulacion_edad.rename(columns={'Edad':'Edad_postulante'}, inplace = True)
vistas_por_postulacion_edad.head()
#Para poder graficar el promedio basado en la edad realizamos un groupby por Edad y calculamos el promedio de 
#la columna promedio vistas/postulacion. Luego, desechamos aquellas edades para las que no contemos con la cantidad
#suficiente de postulantes como para realizar un analisis significativo.

vistas_por_postulacion_promedio_edad = vistas_por_postulacion_edad.groupby('Edad_postulante')\
                                        .agg({'vistas/postulacion':['mean','count']})

vistas_por_postulacion_promedio_edad.columns=['promedio vistas/postulacion', 'cantidad de postulantes']
vistas_por_postulacion_promedio_edad = vistas_por_postulacion_promedio_edad.loc[vistas_por_postulacion_promedio_edad['cantidad de postulantes']>100]
vistas_por_postulacion_promedio_edad.reset_index(inplace = True)
vistas_por_postulacion_promedio_edad.head()
promedio_vistas_por_postulacion_por_edad_plot = vistas_por_postulacion_promedio_edad.plot(
                                                 kind = 'line',
                                                 x = 'Edad_postulante',
                                                 y = 'promedio vistas/postulacion',figsize=(8,8),
                                                 fontsize = 20,legend=False)

promedio_vistas_por_postulacion_por_edad_plot.set_title('Promedio de Vistas por Postulacion basado en Edad',
                                                        fontsize = 20)

promedio_vistas_por_postulacion_por_edad_plot.set_xlabel('Edad postulantes',fontsize = 12)

promedio_vistas_por_postulacion_por_edad_plot.set_ylabel('Promedio vistas/postulacion', fontsize = 12)

vistas_por_postulacion_edad['Edad_postulante'] = pd.cut(vistas_por_postulacion_edad['Edad_postulante'],[18,20,25,30,35,40,45,50,55,60,65,70,75,80])
vistas_por_postulacion_promedio_edad = vistas_por_postulacion_edad.groupby('Edad_postulante')\
                                        .agg({'vistas/postulacion':['mean','count']})

vistas_por_postulacion_promedio_edad.columns=['promedio vistas/postulacion', 'cantidad de postulantes']
vistas_por_postulacion_promedio_edad = vistas_por_postulacion_promedio_edad.loc[vistas_por_postulacion_promedio_edad['cantidad de postulantes']>500]
vistas_por_postulacion_promedio_edad.reset_index(inplace = True)
vistas_por_postulacion_promedio_edad.head()
vistas_por_postulacion_promedio_edad
promedio_vistas_por_postulacion_por_edad_plot = vistas_por_postulacion_promedio_edad.plot(
                                                 kind = 'bar',
                                                 x = 'Edad_postulante',
                                                 y = 'promedio vistas/postulacion',
                                                 color = ['coral','olive','teal','indigo'],legend=False,figsize=(8,8),
                                                 rot=0,fontsize = 12)

promedio_vistas_por_postulacion_por_edad_plot.set_title('Promedio de Vistas por Postulacion basado en Edad',
                                                        fontsize = 20)

promedio_vistas_por_postulacion_por_edad_plot.set_xlabel('Edad postulantes',fontsize = 12)

promedio_vistas_por_postulacion_por_edad_plot.set_ylabel('Promedio vistas/postulacion', fontsize = 12)
plt.ylim([0,3.5])
#Obtenemos un dataframe con el promedio de vistas/postulacion de cada postulante y su sexo

vistas_por_postulacion_sexo = pd.merge(genero_edad,vistas_por_postulacion,on='idpostulante',how = 'inner')[['Sexo','vistas/postulacion']]
vistas_por_postulacion_sexo.rename(columns={'Edad':'Edad_postulante'}, inplace = True)
vistas_por_postulacion_sexo['Sexo'].value_counts()
#Descartamos los postulantes que no declaran sexo, y calculamos el promedio de vistas/postulacion correspondiente a cada sexo

vistas_por_postulacion_sexo = vistas_por_postulacion_sexo.loc[vistas_por_postulacion_sexo['Sexo']!='NO_DECLARA']
vistas_promedio_por_postulacion_sexo = vistas_por_postulacion_sexo.groupby('Sexo')\
                                        .agg({'vistas/postulacion':'mean'})
vistas_promedio_por_postulacion_sexo.reset_index(inplace = True)
vistas_promedio_por_postulacion_sexo.replace({'FEM':'Femenino','MASC':'Masculino'},inplace=True)
vistas_promedio_por_postulacion_sexo
vistas_promedio_por_post_sexo_plot = vistas_promedio_por_postulacion_sexo.plot(
                                                 kind = 'bar',
                                                 x = 'Sexo',
                                                 y = 'vistas/postulacion',
                                                 color = ['salmon','royalblue'],
                                                 fontsize = 20,legend=False,rot=0,
                                                 figsize = (8,8))

vistas_promedio_por_post_sexo_plot.set_title('Promedio Vistas/Postulacion por Genero',fontsize = 20)
vistas_promedio_por_post_sexo_plot.set_xlabel('Genero',fontsize = 16)
vistas_promedio_por_post_sexo_plot.set_ylabel('Promedio Vistas/Postulacion', fontsize = 16)
#Agrego una columna nueva que indica si la visita realiza
vistas_postulaciones_merged['visita_anterior'] = vistas_postulaciones_merged['Fecha_Vista'] < vistas_postulaciones_merged['Fecha_Postulacion']
vistas_antes_despues = vistas_postulaciones_merged['visita_anterior'].value_counts()
vistas_antes_despues

vistas_antes_despues.sort_values(ascending = True, inplace = True)
vistas_antes_despues.index = ['Antes','Despues']
vistas_antes_despues_plot = vistas_antes_despues.plot.bar(figsize=(8,8),rot=0, fontsize = 20)
vistas_antes_despues_plot.set_title("Vistas a Avisos antes/despues de Postularse",fontsize=20)
vistas_antes_despues_plot.set_xlabel("Tiempo",fontsize=12)
vistas_antes_despues_plot.set_ylabel("Cantidad",fontsize=12)


#Tenemos un DF en el que tenemos los detalles de cada postulacion
postulaciones_detalle = pd.merge(postulaciones,avisos_detalle,on= 'idaviso')
postulaciones_detalle.head()
#Tenemos un DF con cada usuario que haya realizado alguna postulacion y el conjunto de 
#areas a las que ese usuario se postulo
areas_por_postulante = postulaciones_detalle.groupby('idpostulante').agg({'Nombre_Area': lambda x: set(x)
                                                                         ,'idaviso':'count'})
areas_por_postulante.reset_index(inplace = True)
areas_por_postulante.rename(columns = {'Nombre_Area':'Areas', 'idaviso':'cantidad_postulaciones'},inplace = True)
areas_por_postulante.head()

#Aqui juntamos el DF anterior con la edad de los postulantes

areas_edad = pd.merge(areas_por_postulante,genero_edad,on='idpostulante',how = 'inner')
hoy = pd.Timestamp(dt.datetime.now())
areas_edad['Edad'] = (hoy - areas_edad['Fecha_Nacimiento']).astype('<m8[Y]')
areas_edad = areas_edad.drop(labels = ['Fecha_Nacimiento','Sexo'],axis = 1)
areas_edad.head()
#Aqui obtenemos la cantidad de areas diferentes a la que cada postulante aplico
areas_edad['cantidad_areas'] = areas_edad['Areas'].apply(len)
areas_edad.head()
#Aqui calculamos promedio por edad de areas distintas a las que se postulan los usuarios

areas_edad_grouped = areas_edad.groupby('Edad').agg({'cantidad_areas':'mean','idpostulante':'count'})
areas_edad_grouped.reset_index(inplace = True)
areas_edad_grouped.rename(columns={'idpostulante':'cantidad_postulantes'}, inplace = True)
areas_edad_grouped = areas_edad_grouped.loc[areas_edad_grouped['cantidad_postulantes'] > 50]
areas_edad_grouped
areas_por_edad_plot = areas_edad_grouped.plot(kind = 'line',
                                             x='Edad',
                                             y='cantidad_areas',legend=False,
                                             fontsize = 20)
areas_por_edad_plot.set_title('Cantidad de areas por edad del postulante',fontsize = 20)
areas_por_edad_plot.set_xlabel('Edad del postulante',fontsize=16)
areas_por_edad_plot.set_ylabel('Cantidad de Areas',fontsize = 16)
areas_edad['Edad'] = pd.cut(areas_edad['Edad'], [15,20,25,35,40,45,50,55,60,65,70,100])
areas_edad.head()
areas_edad_grouped = areas_edad.groupby('Edad').agg({'cantidad_areas':'mean','idpostulante':'count'})
areas_edad_grouped.reset_index(inplace = True)
areas_edad_grouped.rename(columns={'idpostulante':'cantidad_postulantes'}, inplace = True)
areas_edad_grouped = areas_edad_grouped.loc[areas_edad_grouped['cantidad_postulantes'] > 150]
areas_edad_grouped
areas_por_edad_plot = areas_edad_grouped.plot(kind = 'bar',
                                             x='Edad',
                                             y='cantidad_areas',
                                             fontsize = 20,legend=False,
                                             color = ['coral','olive','teal','indigo'])
areas_por_edad_plot.set_title('Cantidad de areas por edad del postulante',fontsize = 20)
areas_por_edad_plot.set_xlabel('Edad del postulante',fontsize=16)
areas_por_edad_plot.set_ylabel('Cantidad de Areas',fontsize = 16)

# Se leen los dataframes.
postulantesEducacion = pd.read_csv('../input/fiuba_1_postulantes_educacion.csv')
postulantesGeneroYEdad = pd.read_csv('../input/fiuba_2_postulantes_genero_y_edad.csv')
oportunidadVistas = pd.read_csv('../input//fiuba_3_vistas.csv')
oportunidadPostulaciones = pd.read_csv('../input/fiuba_4_postulaciones.csv')
avisosOnline = pd.read_csv('../input/fiuba_5_avisos_online.csv')
avisosDetalle = pd.read_csv('../input/fiuba_6_avisos_detalle.csv')
# Elimino los postulantes cuyos ids o fechas de nacimiento son nulas.

# Elimino las filas que no tengan id del postulante.
postulantesEducacion.dropna(subset = ['idpostulante'], inplace = True)
postulantesGeneroYEdad.dropna(subset = ['idpostulante'], inplace = True)

# Creo un nuevo dataframe con los postulantes que tengan fecha de nacimiento.
postulantesConEdad = postulantesGeneroYEdad.dropna(subset=['fechanacimiento'])
# Creo una nueva columna con la edad del postulante. Descarto los postulantes cuyas edades no pueden ser averiguadas.

# Paso la columna fecha de nacimiento a un datetime.
postulantesConEdad['fechanacimiento'] = pd.to_datetime(postulantesConEdad['fechanacimiento'], errors='coerce')

# Creo la columna edad como diferencia entre la fecha actual y la fecha de nacimiento.
fechaActual = pd.Timestamp(datetime.datetime.now())
postulantesConEdad['edad'] = (fechaActual - postulantesConEdad['fechanacimiento']).astype('<m8[Y]')

# Elimino los postulantes sin edades.
postulantesConEdad.dropna(subset = ['edad'], inplace = True)

# Descarto la columna fecha de nacimiento.
postulantesConEdad.drop('fechanacimiento', 1, inplace = True)
# Creo un unico dataframe para los datos del postulante.

# Renombro las columnas del dataframe de educacion a algo mas descriptivo.
postulantesEducacion.rename(columns={'nombre': 'niveleducativo', 'estado': 'estadoniveleducation'}, inplace = True)

# Hago un left join tomando todos los registros del dataframe que contiene la edad.
postulantes = pd.merge(postulantesConEdad, postulantesEducacion, on = 'idpostulante', how = 'left')

# Tomo los campos relacionado a lo educativo, los uno en uno solo y elimino las dos columnas no necesarias.
postulantes['educacion'] = postulantes['niveleducativo'] + '-' + postulantes['estadoniveleducation']
postulantes.drop('niveleducativo', 1, inplace = True)
postulantes.drop('estadoniveleducation', 1, inplace = True)
# Agrego al dataframe de detalles de aviso una columna que diga especifique si sigue online o esta offline.

# Agrego una columna booleana al avisos online para que despues quede en el left merge.
avisosOnline['online'] = True

# Hago el merge entre los detalles y la tabla de online, luego completo los Nan de los offline con False.
avisos = pd.merge(avisosDetalle, avisosOnline, on = 'idaviso', how = 'left')
avisos['online'].fillna(False, inplace = True)
# TODO: Gastón - Aca va el análisis demográfico.
# Renombro la columna idAviso de las vistas a idaviso sin mayuscula así queda homogeneo.
oportunidadVistas.rename(columns={'idAviso': 'idaviso'}, inplace = True)

# Dataframe de vistas.
vistas = pd.merge(oportunidadVistas, avisos, on = 'idaviso', how = 'left')
vistas = pd.merge(vistas, postulantes, on = 'idpostulante', how = 'left')

# Dataframe de postulaciones.
postulaciones = pd.merge(oportunidadPostulaciones, avisos, on = 'idaviso', how = 'left')
postulaciones = pd.merge(vistas, postulantes, on = 'idpostulante', how = 'left')
# Para los siguientes análisis voy a tener en cuenta la cantidad de avisos por area.
cantidadAvisosArea = avisos[['nombre_area']]

# Elimino los registros que no tengan nombre de area.
cantidadAvisosArea.dropna(subset = ['nombre_area'], inplace = True)

# Creo una columna con todos 1 que va a servir de contador, luego realizo el groupby y por ultimo el ordenamiento.
cantidadAvisosArea['cantidadavisos'] = 1
cantidadAvisosArea = cantidadAvisosArea.groupby(['nombre_area'], as_index = False).count()
cantidadAvisosArea = cantidadAvisosArea.sort_values(by = 'cantidadavisos', ascending = False)
# Tomo del dataframe de vistas el nombre de area y la edad solamente.
areaVisitadas = vistas[['nombre_area', 'edad']]

# Agrego una columna cantidad para que luego haga las veces de contador en el group by.
areaVisitadas['cantidad'] = 1

# Divido el dataframe en los 3 grupos de edades.
areaVisitadas1 = areaVisitadas[areaVisitadas['edad'] < 25]
areaVisitadas1.drop('edad', 1, inplace = True)

areaVisitadas2 = areaVisitadas[(areaVisitadas['edad'] >= 25) & (areaVisitadas['edad'] < 35)]
areaVisitadas2.drop('edad', 1, inplace = True)

areaVisitadas3 = areaVisitadas[areaVisitadas['edad'] >= 35]
areaVisitadas3.drop('edad', 1, inplace = True)

# Agrupo segun el area.
areaVisitadas1 = areaVisitadas1.groupby(['nombre_area'], as_index = False).count()
areaVisitadas2 = areaVisitadas2.groupby(['nombre_area'], as_index = False).count()
areaVisitadas3 = areaVisitadas3.groupby(['nombre_area'], as_index = False).count()

# Ordeno de mayor a menor.
areaVisitadas1 = areaVisitadas1.sort_values(by = 'cantidad', ascending = False)
areaVisitadas2 = areaVisitadas2.sort_values(by = 'cantidad', ascending = False)
areaVisitadas3 = areaVisitadas3.sort_values(by = 'cantidad', ascending = False)

# Tomo las 10 areas mas visitadas y las menos visitadas de cada grupo.
areaMasVisitadas1 = areaVisitadas1.head(10)
areaMenosVisitadas1 = areaVisitadas1.tail(10)

areaMasVisitadas2 = areaVisitadas2.head(10)
areaMenosVisitadas2 = areaVisitadas2.tail(10)

areaMasVisitadas3 = areaVisitadas3.head(10)
areaMenosVisitadas3 = areaVisitadas3.tail(10)

# En este analisis se va a analizar la relacion entre la cantidad de avisos de un area y la cantidad de visitas 
# a los avisos de dicha area.

# Agrego a los dataframes de visitas de areas el contador de la cantidad de avisos del area en cuestion.
cantidadAvisosAreaMasVisitadas1 = pd.merge(areaMasVisitadas1, cantidadAvisosArea, on = 'nombre_area', how = 'left')
cantidadAvisosAreaMasVisitadas2 = pd.merge(areaMasVisitadas2, cantidadAvisosArea, on = 'nombre_area', how = 'left')
cantidadAvisosAreaMasVisitadas3 = pd.merge(areaMasVisitadas3, cantidadAvisosArea, on = 'nombre_area', how = 'left')

cantidadAvisosAreaMenosVisitadas1 = pd.merge(areaMenosVisitadas1, cantidadAvisosArea, on = 'nombre_area', how = 'left')
cantidadAvisosAreaMenosVisitadas2 = pd.merge(areaMenosVisitadas2, cantidadAvisosArea, on = 'nombre_area', how = 'left')
cantidadAvisosAreaMenosVisitadas3 = pd.merge(areaMenosVisitadas3, cantidadAvisosArea, on = 'nombre_area', how = 'left')
areaMasVisitadas1
plt.subplots(figsize = (16, 14))
areaMasVisitadas1Plot = sns.barplot(x = areaMasVisitadas1['nombre_area'], y = areaMasVisitadas1['cantidad'], orient = 'v')
areaMasVisitadas1Plot.set_title("Areas con mas visitas para el grupo de 18 a 24 años", fontsize = 20)
areaMasVisitadas1Plot.set_ylabel("Cantidad de visitas", fontsize = 14)
areaMasVisitadas1Plot.set_xlabel("Areas de Trabajo", fontsize = 14)
areaMasVisitadas1Plot.tick_params(labelsize = 12)
plt.show()
cantidadAvisosAreaMasVisitadas1
plt.subplots(figsize = (16, 14))
cantidadAvisosAreaMasVisitadaPlot1 = sns.barplot(x = cantidadAvisosAreaMasVisitadas1['nombre_area'], y = cantidadAvisosAreaMasVisitadas1['cantidadavisos'], orient = 'v')
cantidadAvisosAreaMasVisitadaPlot1.set_title("Cantidad de avisos de las areas mas visitas por el grupo de 18 a 24 años", fontsize = 20)
cantidadAvisosAreaMasVisitadaPlot1.set_ylabel("Cantidad de avisos", fontsize = 16)
cantidadAvisosAreaMasVisitadaPlot1.set_xlabel("Areas de Trabajo", fontsize = 16)
cantidadAvisosAreaMasVisitadaPlot1.tick_params(labelsize = 12)
plt.show()
areaMasVisitadasPlotAcumulado1 = cantidadAvisosAreaMasVisitadas1.set_index('nombre_area').plot.bar(stacked = True, figsize = (16, 16), fontsize = 12, rot = 0)
areaMasVisitadasPlotAcumulado1.set_xlabel("Areas con mas visitas", fontsize = 16)
areaMasVisitadasPlotAcumulado1.set_ylabel("Cantidad de visitas/anuncios", fontsize = 16)
areaMasVisitadasPlotAcumulado1.set_title("Relacion entre cantidad de visitas y cantidad de avisos de las areas mas visitas para el grupo de 18 a 24 años", fontsize = 16)
plt.show()
areaMenosVisitadas1
plt.subplots(figsize = (20, 14))
areaMenosVisitadas1Plot = sns.barplot(x = areaMenosVisitadas1['nombre_area'], y = areaMenosVisitadas1['cantidad'], orient = 'v')
areaMenosVisitadas1Plot.set_title("Areas con menos visitas para el grupo de 18 a 24 años", fontsize = 20)
areaMenosVisitadas1Plot.set_ylabel("Cantidad de visitas", fontsize = 16)
areaMenosVisitadas1Plot.set_xlabel("Areas de Trabajo", fontsize = 16)
areaMenosVisitadas1Plot.set_xticklabels(areaMenosVisitadas1Plot.get_xticklabels(), rotation = 45)
areaMenosVisitadas1Plot.tick_params(labelsize = 12)
plt.show()
cantidadAvisosAreaMenosVisitadas1
plt.subplots(figsize = (20, 14))
cantidadAvisosAreaMenosVisitadaPlot1 = sns.barplot(x = cantidadAvisosAreaMenosVisitadas1['nombre_area'], y = cantidadAvisosAreaMenosVisitadas1['cantidadavisos'], orient = 'v')
cantidadAvisosAreaMenosVisitadaPlot1.set_title("Cantidad de avisos de las areas con menos visitas por el grupo de 18 a 24 años", fontsize = 20)
cantidadAvisosAreaMenosVisitadaPlot1.set_ylabel("Cantidad de avisos", fontsize = 16)
cantidadAvisosAreaMenosVisitadaPlot1.set_xlabel("Areas de Trabajo", fontsize = 16)
cantidadAvisosAreaMenosVisitadaPlot1.set_xticklabels(cantidadAvisosAreaMenosVisitadaPlot1.get_xticklabels(), rotation = 45)
cantidadAvisosAreaMenosVisitadaPlot1.tick_params(labelsize = 12)
plt.show()
areaMenosVisitadasPlotAcumulado1 = cantidadAvisosAreaMenosVisitadas1.set_index('nombre_area').plot.bar(stacked = True, figsize = (16, 16), fontsize = 12, rot = 45)
areaMenosVisitadasPlotAcumulado1.set_xlabel("Areas menos visitadas", fontsize = 16)
areaMenosVisitadasPlotAcumulado1.set_ylabel("Cantidad de visitas/anuncios", fontsize = 16)
areaMenosVisitadasPlotAcumulado1.set_title("Relacion entre cantidad de avisos y cantidad de publicaciones de las areas con menos visitas para el grupo de 18 a 24 años", fontsize = 16)
plt.show()
# Tabla con la lista de las areas menos visitadas por el grupo de 25 a 34 años.
areaMenosVisitadas2
# Tabla con la lista de las areas mas visitadas por el grupo de 25 a 34 años.
areaMasVisitadas2
# Tabla con la lista de las areas mas visitadas por el grupo de mas de 35 años.
areaMasVisitadas3
# Tabla con la lista de las areas menos visitadas por el grupo de 18 a 24 años.
areaMenosVisitadas1
# Tabla con la lista de las areas menos visitadas por el grupo de mas de 35 años.
areaMenosVisitadas3
# Grafico que muestra las areas más vistadas para el grupo de 25 a 34 años.
plt.subplots(figsize = (16, 14))
areaMasVisitadas2Plot = sns.barplot(x = areaMasVisitadas2['nombre_area'], y = areaMasVisitadas2['cantidad'], orient = 'v')
areaMasVisitadas2Plot.set_title("Areas con mas vistas para el grupo de 25 a 34 años", fontsize = 20)
areaMasVisitadas2Plot.set_xlabel("Cantidad de Postulaciones", fontsize = 12)
areaMasVisitadas2Plot.set_ylabel("Areas de Trabajo", fontsize = 12)
plt.show()
# Grafico que muestra las areas más vistadas para el grupo de mas de 35.
plt.subplots(figsize = (16, 14))
areaMasVisitadas3Plot = sns.barplot(x = areaMasVisitadas3['nombre_area'], y = areaMasVisitadas3['cantidad'], orient = 'v')
areaMasVisitadas3Plot.set_title("Areas con mas vistas para el grupo de mas de 35 años", fontsize = 20)
areaMasVisitadas3Plot.set_xlabel("Cantidad de Postulaciones", fontsize = 12)
areaMasVisitadas3Plot.set_ylabel("Areas de Trabajo", fontsize = 12)
plt.show()
# Grafico que muestra las areas menos vistadas para el grupo de 25 a 34 años.
plt.subplots(figsize = (20, 14))
areaMenosVisitadas2Plot = sns.barplot(x = areaMenosVisitadas2['nombre_area'], y = areaMenosVisitadas2['cantidad'], orient = 'v')
areaMenosVisitadas2Plot.set_xticklabels(areaMenosVisitadas2Plot.get_xticklabels(), rotation = 45)
areaMenosVisitadas2Plot.set_title("Areas con menos vistas para el grupo de 25 a 34 años", fontsize = 20)
areaMenosVisitadas2Plot.set_xlabel("Cantidad de Postulaciones", fontsize = 12)
areaMenosVisitadas2Plot.set_ylabel("Areas de Trabajo", fontsize = 12)
plt.show()
# Grafico que muestra las areas menos vistadas para el grupo de mas de 35 años.
plt.subplots(figsize = (20, 14))
areaMenosVisitadas3Plot = sns.barplot(x = areaMenosVisitadas3['nombre_area'], y = areaMenosVisitadas3['cantidad'], orient = 'v')
areaMenosVisitadas3Plot.set_xticklabels(areaMenosVisitadas3Plot.get_xticklabels(), rotation = 45)
areaMenosVisitadas3Plot.set_title("Areas con menos vistas para el grupo de mas de 35 años", fontsize = 20)
areaMenosVisitadas3Plot.set_xlabel("Cantidad de Postulaciones", fontsize = 12)
areaMenosVisitadas3Plot.set_ylabel("Areas de Trabajo", fontsize = 12)
plt.show()