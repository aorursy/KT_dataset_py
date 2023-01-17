#importaciones
from IPython.display import HTML
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:90% !important; }</style>"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mpldates
import seaborn as sns
from pandas.plotting import _converter
plt.subplots(figsize=(15,4))
%matplotlib inline 
plt.style.use('default')
import datetime
import math
sns.set(style="whitegrid") # seteando tipo de grid en seaborn
# Leyendo csvs
df_edu = pd.read_csv("../input/fiuba_1_postulantes_educacion.csv")
df_gen_edad = pd.read_csv("../input/fiuba_2_postulantes_genero_y_edad.csv", parse_dates=['fechanacimiento'])
df_vistas = pd.read_csv("../input/fiuba_3_vistas.csv", parse_dates=['timestamp'])
df_postulaciones = pd.read_csv("../input/fiuba_4_postulaciones.csv", parse_dates=['fechapostulacion'])
df_avisos_detalle = pd.read_csv("../input/fiuba_6_avisos_detalle.csv")
df_edu.sample(3)
df_edu.isnull().sum()
print (df_edu.shape)
print (df_edu["idpostulante"].value_counts().count())
#estas son las variables categoricas de nombre de educacion y estado
df_edu.groupby(["nombre","estado"]).count()
df_gen_edad.head(3)
print (df_gen_edad.shape)
print (df_gen_edad["idpostulante"].value_counts().count())
# limpieza de fechas invalidas
df_gen_edad["fechanacimiento"] = pd.to_datetime(df_gen_edad["fechanacimiento"], errors="coerce")

def calc_edad(x):    
    if (pd.isnull(x)):
        return -1
    return math.floor((datetime.datetime.today()-x).days / 365)

# calculo edad para cada postulante
df_gen_edad["edad"] = df_gen_edad["fechanacimiento"].apply(calc_edad)
df_gen_edad.describe(include="all")
print(df_gen_edad.shape)
print(df_gen_edad.loc[df_gen_edad['edad']>0].shape)
sns.distplot(df_gen_edad.loc[df_gen_edad['edad']> 0]['edad'], color='red', rug=False)
df_gen_edad = df_gen_edad.loc[df_gen_edad['edad'] < 80]
sns.distplot(df_gen_edad.loc[(df_gen_edad['edad'] > 0)]['edad'], color='red', rug=False)
df_gen_edad["sexo"].value_counts().plot(kind="barh", title= "composicion del sexo de los postulantes")
print(df_vistas.shape)
print(df_vistas.isnull().sum())
df_vistas.sample(3)
df_vistas['date'] = df_vistas['timestamp'].dt.date
df_vistas.head()
print(df_postulaciones.shape)
print(df_postulaciones.isnull().sum())
df_postulaciones.sample(3)
df_postulaciones['date'] = df_postulaciones['fechapostulacion'].dt.date
df_postulaciones.head()
df_postulaciones.dtypes
df_postulaciones.describe(include="all")
idx1 = pd.Index(df_vistas['date']).drop_duplicates()
idx2 = pd.Index(df_postulaciones['date']).drop_duplicates()
idx=idx1.union(idx2)
idx.drop_duplicates()
df = pd.DataFrame(index = idx)
df.head()
df['vistas'] = df.index.to_series().apply(lambda x: 1 if x >= df_vistas['date'].min() and x <= df_vistas['date'].max() else np.NaN)
df.sample(5)
df['postulaciones'] = df.index.to_series().apply(lambda x: 2 if x >= df_postulaciones['date'].min() and x <= df_postulaciones['date'].max() else np.NaN)

p = df.plot(ylim=[0, 3], legend=False, title ='Rango de tiempo de Postulaciones y Vistas'
            , figsize =(12,7), lw=7, fontsize=12)
p.set_yticks([1., 2.])
p.set_yticklabels(['vistas', 'Postulaciones'])
#p.set_xlabel('Tiempo', size = 15)
p.title.set_size(18)
print ("max:",df_vistas['date'].min())
print ("min:",df_postulaciones['date'].max())
print (df_postulaciones['date'].max()-df_vistas['date'].min())
del df
df_avisos_detalle.head(10)
df_avisos_detalle.describe(include="all")
orden_nombre = {
    "Doctorado":0,
    "Master":1,
    "Posgrado":2,
    "Universitario":3,
    "Terciario/Técnico":4,
    "Secundario":5,
    "Otro":6
}

orden_estado = {
    "Graduado":0,
    "En Curso":1,
    "Abandonado":2,
}

df_edu["prioridad_nom"] = df_edu["nombre"].apply(lambda nombre: orden_nombre.get(nombre))
df_edu["prioridad_est"] = df_edu["estado"].apply(lambda estado: orden_estado.get(estado))

df_edu.sort_values(by= ["idpostulante","prioridad_est","prioridad_nom"],inplace= True)

df_edu_unicos_max = df_edu.drop_duplicates(subset = "idpostulante",keep= "first")

df_edu_max_postulaciones = df_edu_unicos_max.merge(df_postulaciones, on = "idpostulante")

df_aviso_corto = df_avisos_detalle[["idaviso","tipo_de_trabajo","nivel_laboral","nombre_area"]]

df_max_post_detalle = df_edu_max_postulaciones.merge(df_aviso_corto, on = "idaviso")
def top_n_areas_graduados(n, educacion):
    top_10_areas_graduados_secundario = df_max_post_detalle[(df_max_post_detalle["nombre"] == educacion) & (df_max_post_detalle["estado"] == "Graduado")] \
                                        .groupby("nombre_area")['idpostulante']\
                                        .count() \
                                        .sort_values(ascending=False) \
                                        .head(n)

    g = sns.barplot(x=top_10_areas_graduados_secundario.values, y=top_10_areas_graduados_secundario.index, orient='h', palette="hls")
    g.set_title("Postulantes graduados de nivel {0} por area".format(educacion.lower()), fontsize=18)
    g.set_xlabel("Cantidad de graduados", fontsize=12)
    g.set_ylabel("Nombre del Area", fontsize=12)
    plt.subplots_adjust(top=0.9)
top_n_areas_graduados(10, "Secundario")    
top_n_areas_graduados(10, "Universitario")
top_n_areas_graduados(10, "Doctorado")
df_edu_unicos_con_edad = df_edu_unicos_max.loc[df_edu_unicos_max['estado'] == 'Graduado']\
                                          .merge(df_gen_edad, on='idpostulante')
df_edu_unicos_con_edad = df_edu_unicos_con_edad.loc[df_edu_unicos_con_edad['edad'] > 0]
    
fig, axes = plt.subplots(4, 2, figsize=(15,15))

fig.suptitle("Distribución de edad para graduados", fontsize=18)

nombres = list(orden_nombre.keys())

# no parece coherente que haya personas de 20 años con doctorado, por lo 
# que vamos a fijar edades minimas esperadas
edad_minima_nivel_educativo = {
    "Doctorado": 25,
    "Master": 25,
    "Posgrado": 25,
    "Universitario": 23,
    "Terciario/Técnico": 0,
    "Secundario": 0,
    "Otro": 0
}

# no queremos que los casos aislados nos sesguen los gráficos
LIM_SUP_EDAD = 80

indice_nivel = 0

colores = sns.color_palette("hls", 8)

for i in range(0, 4):
    for j in range(0, 2):
        ax = axes[i][j]
        
        # tenemos 8 axes para 7 niveles de educacion, ocultamos el ultimo que quedaria vacio
        if i == 3 and j == 1:
            ax.set_visible(False)
            break
                
        nom = nombres[indice_nivel]
        
        serie = df_edu_unicos_con_edad.groupby(['nombre']) \
                                      .get_group(nom)['edad']
            
        serie = serie.loc[(edad_minima_nivel_educativo[nom] <= serie) & (serie <= LIM_SUP_EDAD)]
        
        ticks = np.arange(20, 80, 5)
        
        ax.hist(serie, 80 - 15, color=colores[indice_nivel]) # uno para cada edad
        ax.set_title(nom)
        ax.set_xticks(ticks)
        ax.set_xlabel("Edad", fontsize=12)
        ax.set_ylabel("Cantidad", fontsize=12)
        indice_nivel += 1
    
plt.tight_layout()
plt.subplots_adjust(top=0.9)
promedio_edad_por_educacion = df_edu_unicos_con_edad.groupby('nombre')['edad'].mean().reset_index()
g = sns.barplot(x=promedio_edad_por_educacion['edad'], y=promedio_edad_por_educacion['nombre'], orient='h', palette="hls")
g.set_title("Edad promedio por nivel educativo alcanzado", fontsize=18)
g.set_xlabel("Nivel educativo", fontsize=12)
g.set_ylabel("Edad promedio", fontsize=12);
del df_edu_unicos_con_edad
def top_n_tipo_trabajo_educacion(n, educacion):
    top_tipo_trabajo_universitario = df_max_post_detalle.loc[df_max_post_detalle["nombre"] == educacion]\
                                                         .groupby("tipo_de_trabajo")['idpostulante']\
                                                         .count()\
                                                         .sort_values(ascending=False)\
                                                         .head(n)

    g = sns.barplot(x=top_tipo_trabajo_universitario.values, y=top_tipo_trabajo_universitario.index, orient="h", palette="hls")    
    g.set_title("Distibucion de Tipo de trabajo en postulantes {0}".format(educacion), fontsize=18)
    g.set_xlabel("Cantidad", fontsize=12)
    g.set_ylabel("Tipo de Trabajo", fontsize=12);
    
    del top_tipo_trabajo_universitario
top_n_tipo_trabajo_educacion(3, "Universitario")
top_n_tipo_trabajo_educacion(3, "Secundario")
top_n_tipo_trabajo_educacion(3, "Doctorado")
df_gen_edad_valida = df_gen_edad.loc[(17 <= df_gen_edad['edad']) & (df_gen_edad['edad'] <= 80)]

df_edad_fem = df_gen_edad_valida.loc[df_gen_edad_valida['sexo'] == 'FEM']
df_edad_masc = df_gen_edad_valida.loc[df_gen_edad_valida['sexo'] == 'MASC']

g = sns.distplot(df_edad_fem['edad'], color='red', label='FEM', rug=False)
g = sns.distplot(df_edad_masc['edad'], color='blue', label='MASC', rug=False)

g.set_title("Edad por genero", fontsize=18)
g.set_xlabel("Edad", fontsize=12)
g.legend();
del df_edad_fem
del df_edad_masc
id_gen_corto = df_gen_edad_valida[["idpostulante","sexo"]]
df_sexo_tipo_trabajo =df_max_post_detalle.merge(id_gen_corto,on= "idpostulante")[["sexo","tipo_de_trabajo"]]
df_fem_tipo_trabajo = df_sexo_tipo_trabajo[df_sexo_tipo_trabajo["sexo"] == "FEM"]
top3_fem_tipo_trabajo= df_fem_tipo_trabajo.groupby(["tipo_de_trabajo"])\
                      .count()\
                      .sort_values(by= "sexo",ascending=False)\
                      .head(2)
total= float(top3_fem_tipo_trabajo.sum())
top3_fem_tipo_trabajo["porcentaje"] = top3_fem_tipo_trabajo["sexo"].apply(lambda x: round(x/total*100))
df_masc_tipo_trabajo = df_sexo_tipo_trabajo[df_sexo_tipo_trabajo["sexo"] == "MASC"]
top3_masc_tipo_trabajo= df_masc_tipo_trabajo.groupby(["tipo_de_trabajo"])\
                      .count()\
                      .sort_values(by= "sexo",ascending=False)\
                      .head(2)
total= float(top3_masc_tipo_trabajo.sum())
top3_masc_tipo_trabajo["porcentaje"] = top3_masc_tipo_trabajo["sexo"].apply(lambda x: round(x/total*100))
N = 2
ind = np.arange(N)  # la locacion x del grupo
width = 0.27      # el ancho de la barra

fig = plt.figure()
ax = fig.add_subplot(111)

#genero los valores del grafico
mascvals = list(top3_masc_tipo_trabajo["porcentaje"])
rectmasc = ax.bar(ind, mascvals, width, color='b',alpha= 0.6)
femvals = list(top3_fem_tipo_trabajo["porcentaje"])
rectfem = ax.bar(ind+width, femvals, width, color='r',alpha= 0.6)

ax.set_title("Distribucion tipo de trabajo Por Sexo", fontsize=15)
ax.set_ylabel('Porcentaje')
ax.set_xticks(ind+width)
ax.set_xticklabels( ('FULL TIME', 'PART TIME') )
ax.legend( (rectmasc[0], rectfem), ('masc', 'fem') )

#creo la etiquetade los grupos
def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., h, '%d'%int(h),
                ha='center', va='bottom')

autolabel(rectmasc)
autolabel(rectfem)
del df_fem_tipo_trabajo
del df_masc_tipo_trabajo
del top3_fem_tipo_trabajo
del top3_masc_tipo_trabajo
edu_max_genero= df_edu_unicos_max.merge(id_gen_corto, on= "idpostulante")
edu_masc= edu_max_genero[edu_max_genero["sexo"]== "MASC"].groupby("nombre")["idpostulante",].count()
total= float(edu_masc["idpostulante"].sum())
edu_masc["porcentaje"] = edu_masc["idpostulante"].apply(lambda x: round(x/total*100))
edu_masc=edu_masc.reset_index()
edu_masc["prioridad"]= edu_masc["nombre"].apply(lambda nom: orden_nombre.get(nom))
edu_masc = edu_masc.sort_values(by= "prioridad",ascending= False)
edu_masc
edu_fem= edu_max_genero[edu_max_genero["sexo"]== "FEM"].groupby("nombre")["idpostulante",].count()
total= float(edu_fem["idpostulante"].sum())
edu_fem["porcentaje"] = edu_fem["idpostulante"].apply(lambda x: round(x/total*100))
edu_fem=edu_fem.reset_index()
edu_fem["prioridad"]= edu_fem["nombre"].apply(lambda nom: orden_nombre.get(nom))
edu_fem = edu_fem.sort_values(by= "prioridad",ascending= False)
edu_fem
N = 7
ind = np.arange(N)  # la locacion x del grupo
width = 0.27      # el ancho de la barra

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)

#genero los valores del grafico
mascvals = list(edu_masc["porcentaje"])
rectmasc = ax.bar(ind, mascvals, width, color='b',alpha= 0.6)
femvals = list(edu_fem["porcentaje"])
rectfem = ax.bar(ind+width, femvals, width, color='r',alpha= 0.6)

ax.set_title("Distribución de nivel educativo alcanzado por sexo", fontsize=18)
ax.set_ylabel('Porcentaje', fontsize=12)
ax.set_xlabel('Nivel', fontsize=12)
ax.set_xticks(ind+width)
ax.set_xticklabels( list(edu_masc["nombre"]))
ax.legend( (rectmasc[0], rectfem), ('masc', 'fem') )

plt.xticks(rotation=30)

#creo la etiquetade los grupos
def autolabel(rects):
    for rect in rects:
        h = rect.get_height()
        ax.text(rect.get_x()+rect.get_width()/2., h, '%d'%int(h),
                ha='center', va='bottom')

autolabel(rectmasc)
autolabel(rectfem)
# elimino los dataframes que ya no se van a usar
del edu_fem
del edu_masc
del df_sexo_tipo_trabajo
del id_gen_corto
df_avisos_postulantes = df_gen_edad.merge(df_postulaciones, on='idpostulante').merge(df_avisos_detalle, on='idaviso')

# tabla cruzada con frecuencia de sexo FEM y MASC
ct = pd.crosstab(df_avisos_postulantes['nombre_area'], df_avisos_postulantes['sexo'])

# columnas para comparar y reordenar
ct['tot'] = ct['FEM'] + ct['MASC']
ct['diff'] = ct['FEM'] - ct['MASC']
ct = ct.sort_values(by='diff', ascending=False)

# tomo los top 10 y preparo para graficar
ct_top_fem_stacked = ct.head(10)[['FEM', 'MASC']].stack().reset_index().rename(columns={0:'count'})

# el iloc[::-1] es para invertirlo dado que estamos tomando los 10 ultimos en forma descendiente
ct_top_masc_stacked = ct.tail(10).iloc[::-1][['FEM', 'MASC']].stack().reset_index().rename(columns={0:'count'})
# caso fem
g = sns.factorplot(x='nombre_area', 
                   y='count',
                   data=ct_top_fem_stacked,
                   hue='sexo',
                   palette= {'FEM': 'r', 'MASC': 'b'},
                   errwidth=0.8,
                   kind="bar",
                   legend_out=False,
                   size=7,
                   alpha=0.6)

g.set_xticklabels(rotation=75)

g.set_xlabels("Areas", fontsize=12)
g.set_ylabels("Cantidad de aplicantes", fontsize=12)

yticks = np.arange(0, 250000, 25000)
g.set(yticks=yticks)

plt.subplots_adjust(top=0.93)
g.fig.suptitle('Areas con mayor diferencia en sexo. Predominancia femenina', fontsize=18);
# caso masc
g = sns.factorplot(x='nombre_area', 
                   y='count',
                   data=ct_top_masc_stacked,
                   hue='sexo',
                   palette= {'FEM': 'red', 'MASC': 'blue'},
                   errwidth=0.8,
                   kind="bar",
                   legend_out=False,
                   size=7,
                   alpha=0.6)

g.set_xticklabels(rotation=75)

g.set_xlabels("Areas", fontsize=12)
g.set_ylabels("Cantidad de aplicantes", fontsize=12)

yticks = np.arange(0, 250000, 25000)
g.set(yticks=yticks)

plt.subplots_adjust(top=0.93)
g.fig.suptitle('Areas con mayor diferencia en sexo. Predominancia masculina', fontsize=18);
del ct_top_fem_stacked
del ct_top_masc_stacked
del ct
del df_avisos_postulantes
df_vistas["hour"] = df_vistas["timestamp"].dt.hour
#df_vistas["date"] = df_vistas["timestamp"].dt.date
df_postulaciones["hour"] = df_postulaciones["fechapostulacion"].dt.hour
#df_postulaciones["date"] = df_postulaciones["fechapostulacion"].dt.date
df_postulaciones["weekday"] = df_postulaciones["fechapostulacion"].dt.weekday
df_postulaciones["corrected_hour"] = (df_postulaciones["fechapostulacion"] + pd.DateOffset(hours=5)).dt.hour
def postulaciones_dias_heatmap(col_agrupacion_horas):
    postulaciones_dia_hora = df_postulaciones.groupby(["weekday",col_agrupacion_horas])["idaviso",].count()
    postulaciones_dia_hora = postulaciones_dia_hora.reset_index()
    pivot_post = postulaciones_dia_hora.pivot_table(index=col_agrupacion_horas, columns="weekday", values= "idaviso")
    pivot_post = pivot_post.sort_index(ascending=False)
    plt.figure(figsize=(7,5.6))
    cmap = sns.cm.rocket_r
    g = sns.heatmap(pivot_post, xticklabels = ['Mon','Tue','Wen','Thu','Fri','Sut','Sun'] ,cmap= cmap)
    g.set_title("Distribución de Postulaciones", fontsize=18)
    g.set_xlabel("Día", fontsize=12)
    g.set_ylabel("Hora", fontsize=12)
postulaciones_dias_heatmap("hour")
postulaciones_dias_heatmap("corrected_hour")
date_ini = datetime.date(year=2018,month=2,day=23)
date_fin = datetime.date(year=2018,month=2,day=28)
df_periodo_postulaciones = df_postulaciones[df_postulaciones["fechapostulacion"].dt.date >= date_ini]
df_periodo_vistas = df_vistas[(df_vistas["timestamp"].dt.date >= date_ini) & (df_vistas["timestamp"].dt.date <= date_fin)]
def postulaciones_vistas_radial(extractor_clave_agrupamiento):
    # extractor_clave_agrupamiento puede ser un string o una funcion
    # https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.groupby.html
    
    categories = range(0, 24)
    N = len(categories)

    def avarage (list):
        for i in range(0,len(list)):
            list[i] = list[i]/((date_fin-date_ini).days)

    from math import pi
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # inicializar grafica
    ax = plt.subplot(111, polar=True)

    # buscamos que el primer valor comienze desde arriba:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # dibujar cada tick con su label
    plt.xticks(angles[:-1], categories)

    # dibujar labels para el eje y
    ax.set_rlabel_position(0)
    plt.yticks([2500,5000,7500,10000,12500], ["2.5","5","7.5","10","12.5","15"], color="grey", size=10)
    plt.ylim(0,15000)

    values2 = list(df_periodo_vistas.groupby("hour")["idAviso"].count())
    values2 += values2[:1]
    avarage(values2)
    ax.plot(angles, values2, linewidth=1, linestyle='solid', label="Cantidad Promedio de Vistas por Hora")
    ax.fill(angles, values2, 'b', alpha=0.1)

    values = list(df_periodo_postulaciones.set_index("fechapostulacion").groupby(extractor_clave_agrupamiento)["idaviso"].count())
    values += values[:1]
    avarage(values)
    ax.plot(angles, values, linewidth=1, linestyle='solid',color = "C2", label="Cantidad Promedio de Postulaciones por Hora")
    ax.fill(angles, values, 'r', alpha=0.1)

    plt.suptitle('Cantidad de vistas y postulaciones promedio por hora', fontsize=18)
    plt.subplots_adjust(top=.825)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1));
postulaciones_vistas_radial("hour")
def groupby_date_add_5_hours(x):
    return (x + pd.DateOffset(hours=5)).hour

postulaciones_vistas_radial("corrected_hour")
df_postulaciones.drop(labels="corrected_hour", axis=1, inplace=True)
# columna con fecha y hora
df_postulaciones['datehour'] = pd.to_datetime({'year': df_postulaciones['fechapostulacion'].dt.year,
                                               'month': df_postulaciones['fechapostulacion'].dt.month,
                                               'day': df_postulaciones['fechapostulacion'].dt.day,
                                               'hour': df_postulaciones['fechapostulacion'].dt.hour})
df_vistas['datehour'] = pd.to_datetime({'year': df_vistas['timestamp'].dt.year,
                                       'month': df_vistas['timestamp'].dt.month,
                                       'day': df_vistas['timestamp'].dt.day,
                                       'hour': df_vistas['timestamp'].dt.hour})
# Tomamos el rango comun entre los dos sets de datos
datehour_min = max(min(df_vistas['datehour']), min(df_postulaciones['datehour']))
datehour_max = min(max(df_vistas['datehour']), max(df_postulaciones['datehour']))

postulaciones_por_fecha_y_hora = df_postulaciones.loc[(datehour_min <= df_postulaciones['datehour']) & (df_postulaciones['datehour'] <= datehour_max)].groupby("datehour")["idaviso"].count().sort_index()
vistas_por_fecha_y_hora = df_vistas.loc[(datehour_min <= df_vistas['datehour']) & (df_vistas['datehour'] <= datehour_max)].groupby("datehour")["idAviso"].count().sort_index()
fig, ax = plt.subplots()

ax.plot(postulaciones_por_fecha_y_hora.index, postulaciones_por_fecha_y_hora)
ax.plot(vistas_por_fecha_y_hora.index, vistas_por_fecha_y_hora)

fig = ax.get_figure()
fig.suptitle("Postulaciones y vistas a lo largo del tiempo", fontsize=18)
fig.set_figwidth(12)
fig.set_figheight(5)

ax.set_xlabel("Fecha y hora", fontsize=12)
ax.set_ylabel("Cantidad", fontsize=12)
ax.set_xlim(datehour_min, datehour_max)

ax.xaxis.set_major_locator(mpldates.HourLocator(byhour=np.arange(0,24), interval=6))
ax.xaxis.set_major_formatter(mpldates.DateFormatter("%d %h %Hhs"))

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ['Postulaciones', 'Vistas'], fontsize=14)

fig.autofmt_xdate()
# Correjimos el offset de 5 horas que hay entre postulacioens y vistas
postulaciones_por_fecha_y_hora.index = postulaciones_por_fecha_y_hora.index.map(lambda x: x + pd.DateOffset(hours=5))
# Regeneramos el rango común entre ambos dado que correjimos la hora de postulaciones
datehour_min = max(min(df_vistas['datehour']), min(postulaciones_por_fecha_y_hora.index))
datehour_max = min(max(df_vistas['datehour']), max(postulaciones_por_fecha_y_hora.index))
vistas_por_fecha_y_hora = df_vistas.loc[(datehour_min <= df_vistas['datehour']) & (df_vistas['datehour'] <= datehour_max)].groupby("datehour")["idAviso"].count().sort_index()

fig, ax = plt.subplots()

ax.plot(postulaciones_por_fecha_y_hora.index, postulaciones_por_fecha_y_hora)
ax.plot(vistas_por_fecha_y_hora.index, vistas_por_fecha_y_hora)

fig = ax.get_figure()
fig.suptitle("Postulaciones y vistas a lo largo del tiempo", fontsize=18)
fig.set_figwidth(12)
fig.set_figheight(5)

ax.set_xlabel("Fecha y hora", fontsize=12)
ax.set_ylabel("Cantidad", fontsize=12)
ax.set_xlim(datehour_min, datehour_max)

ax.xaxis.set_major_locator(mpldates.HourLocator(byhour=np.arange(0,24), interval=6))
ax.xaxis.set_major_formatter(mpldates.DateFormatter("%d %h %Hhs"))

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, ['Postulaciones', 'Vistas'], fontsize=14)

fig.autofmt_xdate()
del postulaciones_por_fecha_y_hora
del vistas_por_fecha_y_hora
# Vamos a usar solo las fechas para determinar el rango comun
date_min = max(min(df_vistas['date']), min(df_postulaciones['date']))
date_max = min(max(df_vistas['date']), max(df_postulaciones['date']))
postulaciones_dia_hora = df_postulaciones.loc[(date_min <= df_postulaciones['date']) & (df_postulaciones['date'] <= date_max)]\
                                         .groupby(["date","hour"])["idaviso",].count()
postulaciones_dia_hora = postulaciones_dia_hora.reset_index()
pivot_post = postulaciones_dia_hora.pivot_table(index= "hour", columns="date", values= "idaviso")
pivot_post = pivot_post.sort_index(ascending=False)
plt.figure(figsize=(7,5.6))
cmap = sns.cm.rocket_r

g = sns.heatmap(pivot_post, cmap=cmap)
g.set_title("Distribución de Postulaciones en el tiempo", fontsize=18)
g.set_xlabel("Fecha", fontsize=12)
g.set_ylabel("Hora", fontsize=12)

fig = g.get_figure()
fig.set_figwidth(18)
fig.set_figheight(10)
vistas_dia_hora = df_vistas.loc[(date_min <= df_vistas['date']) & (df_vistas['date'] <= date_max)]\
                           .groupby(["date","hour"])["idAviso",].count()
vistas_dia_hora = vistas_dia_hora.reset_index()
pivot_post = vistas_dia_hora.pivot_table(index= "hour", columns="date", values= "idAviso")
pivot_post = pivot_post.sort_index(ascending=False)
plt.figure(figsize=(7,5.6))
cmap = sns.cm.rocket_r

g = sns.heatmap(pivot_post, cmap=cmap)
g.set_title("Distribución de Vistas en el tiempo", fontsize=18)
g.set_xlabel("Fecha", fontsize=12)
g.set_ylabel("Hora", fontsize=12)

fig = g.get_figure()
fig.set_figwidth(18)
fig.set_figheight(10)
df_postulaciones.drop(labels='datehour', axis=1, inplace=True)
df_vistas.drop(labels='datehour', axis=1, inplace=True)
del postulaciones_dia_hora
del vistas_dia_hora
del pivot_post
df_posts_avisos = df_avisos_detalle[['idaviso','nombre_area','denominacion_empresa', 'nivel_laboral']].merge(df_postulaciones[['idaviso','fechapostulacion','idpostulante']], on='idaviso')
df_posts_avisos['fecha'] = df_posts_avisos['fechapostulacion'].dt.date
def graficar_top_n_postulaciones(N, clave_agrupacion, clave_titulo, x_tick_step=1):
    df_posts_agrupado = df_posts_avisos.groupby(clave_agrupacion)\
                                        .agg({'idpostulante':'count'})\
                                        .rename(columns={'idpostulante': 'count'})

    tot = df_posts_agrupado['count'].sum()

    df_posts_agrupado['perc'] = df_posts_agrupado['count'] * 100.0 / tot
    df_posts_agrupado = df_posts_agrupado.sort_values(by='count', ascending=False)
    df_posts_agrupado = df_posts_agrupado.reset_index()

    df_top_posts = pd.DataFrame(df_posts_agrupado.head(N))

    max_perc = math.ceil(df_top_posts['perc'].max())
    xticks = np.arange(0, max_perc, x_tick_step)

    g = sns.barplot(x=df_top_posts['perc'], y=df_top_posts[clave_agrupacion], orient='h',
                    palette="hls")
    g.set_title("Top {0} de postulaciones por {1}".format(N, clave_titulo.lower()), fontsize=18)
    g.set_xlabel("Postulaciones (%)", fontsize=12)
    g.set_ylabel(clave_titulo, fontsize=12)
    g.set(xticks=xticks);
    
    return g
graficar_top_n_postulaciones(10, 'nombre_area', 'Area');
graficar_top_n_postulaciones(10, 'denominacion_empresa', 'Empresa', x_tick_step=0.5);
#obtenemos las 5 areas con mas avisos
top_areas = df_avisos_detalle['nombre_area'].value_counts()\
                                            .sort_values(ascending=False)\
                                            .head(5)\
                                            .keys()\
                                            .tolist()

df_top_area_laboral = df_avisos_detalle.loc[df_avisos_detalle['nombre_area'].isin(top_areas)]\
                                       .groupby(['nombre_area','nivel_laboral'])\
                                       .agg({'nombre_area':'count'})\
                                       .rename(columns = {'nombre_area':'cant_avisos'})

axs = df_top_area_laboral.reset_index()\
                         .pivot(index = 'nombre_area', columns = 'nivel_laboral', values='cant_avisos')\
                         .plot(kind = 'bar', figsize =(12,8), fontsize = 12, rot=75,
                                title ='Cantidad de avisos por nivel laboral y por area')

axs.set_ylabel('Avisos', size = 12)
axs.set_xlabel('Area', size = 12)
axs.legend(fontsize = 12)
axs.title.set_size(18)
#obtenemos las 5 areas con mas postulaciones
top_post_areas = df_posts_avisos['nombre_area'].value_counts()\
                                              .sort_values(ascending = False)\
                                              .head(5)\
                                              .keys()\
                                              .tolist()
                
df_top_posts_area =  df_posts_avisos.loc[df_posts_avisos['nombre_area'].isin(top_post_areas)]\
                                    .groupby(['nombre_area','nivel_laboral'])\
                                    .agg({'nombre_area':'count'})\
                                    .rename(columns = {'nombre_area':'count'})

axs = df_top_posts_area.reset_index()\
                       .pivot(index = 'nombre_area', columns = 'nivel_laboral', values='count')\
                       .plot(kind = 'bar', figsize =(12,8), fontsize = 12, rot=75,
                             title ='Cantidad de postulaciones por nivel laboral y por area')

axs.set_ylabel('Postulaciones', size = 15)
axs.set_xlabel('Area', size = 15)
axs.legend(fontsize = 12)
axs.title.set_size(18)
del df_top_posts_area
del df_top_area_laboral
del top_areas
del top_post_areas
del df_posts_avisos
postulaciones_por_edades = df_postulaciones[['idpostulante']].merge(df_gen_edad[['idpostulante', 'edad']], on='idpostulante')

postulaciones_por_edades = postulaciones_por_edades.loc[(0 < postulaciones_por_edades['edad']) & (postulaciones_por_edades['edad'] <= 80)]

min_edad = min(postulaciones_por_edades['edad'])
max_edad = max(postulaciones_por_edades['edad'])

ax = postulaciones_por_edades.hist(bins=(max_edad - min_edad))[0][0]

xticks = np.arange(15, 80, 5)

ax.set_title('')
ax.set_xlabel('Edad', fontsize=12)
ax.set_ylabel('Postulaciones', fontsize=12)
ax.set_xticks(xticks)

fig = ax.get_figure()
fig.suptitle('Postulaciones por edad', fontsize=18);
postulaciones_por_edades['aux'] = 1
prom_postulaciones_por_edades = postulaciones_por_edades.groupby(['edad', 'idpostulante'])\
                                                        .count()\
                                                        .reset_index()\
                                                        .groupby('edad')\
                                                        .agg({'aux':'mean'})\
                                                        .reset_index()\
                                                        .rename(columns={'aux':'prom'})
                        
ax = prom_postulaciones_por_edades.set_index('edad').plot.bar()

# oculto labels no multiplos de 5
for t in ax.get_xticklabels():
    if int(t.get_text()) % 5 != 0:
        t.set_visible(False)        

ax.set_title('')
ax.set_xlabel('Edad', fontsize=12)
ax.set_ylabel('Promedio de postulaciones', fontsize=12)

fig = ax.get_figure()
fig.suptitle('Promedio de postulaciones por edad', fontsize=18)

plt.xticks(rotation=0);
df_gen_edad_valida.groupby('edad')\
                  .count()\
                  .rename(columns={'idpostulante':'count'})\
                  .sort_values(by='count')\
                  .loc[lambda x: x['count'] < 100,:]
                
# Las edades más altas tienen en promedio mayor cantidad de postulaciones
# por la baja cantidad de personas en esos grupos
df_postulantes_por_aviso = df_gen_edad[['idpostulante', 'edad']].merge(df_postulaciones[['idpostulante', 'idaviso']], on='idpostulante')\
                                                                .merge(df_avisos_detalle[['idaviso','nivel_laboral']], on='idaviso')

g = sns.FacetGrid(df_postulantes_por_aviso.loc[(0 <= df_postulantes_por_aviso['edad']) & (df_postulantes_por_aviso['edad'] <= 80)], col="nivel_laboral", hue="nivel_laboral", col_wrap=2, size=6)
g.map(sns.distplot, "edad", rug=False)
g.set_titles("Nivel: {col_name}", fontsize=14)

for ax in g.axes.flat:
    plt.setp(ax.get_yticklabels(), visible=True)
    plt.setp(ax.get_xticklabels(), visible=True)
    ax.set_xlabel('Edad', fontsize=12)

plt.tight_layout()
del df_postulantes_por_aviso