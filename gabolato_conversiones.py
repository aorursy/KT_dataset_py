%matplotlib inline
import pandas as pd 
import datetime
import numpy as np
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from math import pi
troca_persons = pd.read_csv('../input/events.csv', low_memory = False)
#Diversidad de columnas del df
troca_persons.columns
#Previo chequeo de nulos en la columna person:

troca_persons.person.isnull().values.any()
personas = troca_persons.person.value_counts()
personas
# Chequeo de nulos en la columna event
troca_persons.event.isnull().values.any()
personasPorEvento = troca_persons[['event','person']]

PPE = personasPorEvento.groupby('person')['event'].count().sort_values(ascending = False).reset_index()

PPE.head(10).person.values

g = sns.catplot(x="person", y="event", kind = "bar", data = PPE.head(15), height = 15,aspect = 1, palette="hls")
g.set_xticklabels(rotation=30)
plt.title('Cantidad de eventos por persona (15 personas con mas eventos)', fontsize = 18)
plt.ylabel('total de eventos', fontsize = 16)
plt.xlabel('Quince personas con mas eventos', fontsize = 12)
diezPersonasConMasEvents = np.array(PPE.head(10).person.values)
events = troca_persons[troca_persons['person'].isin(diezPersonasConMasEvents)]
events = pd.crosstab(events.person , events.event)
events

# ------- PARTE 1: Defino una funcion que hace un plot recibiendo etiquetas, los stats de las columnas 
# y el titulo del plot

def hacerRadarChart( labels, stats, nombrePersona):
  
    #Establezco el ángulo de las coordenadadas polares. 
    #Y usamos el np.concatenate para dibujar un recinto cerrado para el radar chart."""
    angles = np.linspace(0,2*np.pi,len(labels), endpoint=False)
    # cerrando el plot
    stats=np.concatenate((stats,[stats[0]]))
    angles=np.concatenate((angles,[angles[0]]))

    #Creo una figura
    fig= plt.figure(figsize=(4,4))

    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8],projection='polar', facecolor='lightblue')

    fig.text(0.5, 1.20, nombrePersona,
             horizontalalignment='center', color='black', weight='bold', size='large')

    #Defino metricas de la figura y dibujo el plot
    ax.plot(angles, stats, 'o-', linewidth=2)
    ax.set_rmax(1)

    #Grafico el area de stats
    ax.fill(angles, stats, alpha=0.25)

    #Grafico los labels
    ax.set_thetagrids(angles * 180/np.pi, labels)

    #Defino una orientacion diagonal para labels en grados
    ticks= np.linspace(0,360,12)[:-1] 
    #Convierto a radianes
    ax.set_xticks(np.deg2rad(ticks))

    #Redibujo la figura
    plt.gcf().canvas.draw_idle()
    #sumo pi al angulo de cada label si el coseno da negativo(invierte el sentido de la palabra)
    angles[np.cos(angles) < 0] = angles[np.cos(angles) < 0] + np.pi
    #Paso radianes a grados
    angles = np.rad2deg(angles)

    #itero sobre dos listas en paralelo: angulos y labels
    for label, angle in zip(ax.get_xticklabels(), angles):
        #obtengo la posicion en x,y de cada label
        x,y = label.get_position()
        #obtengo el label y le aplico una transformacion, cambio la distancia hacia el chart
        lab = ax.text(x,y-0.29, label.get_text(), transform=label.get_transform(),
                      ha=label.get_ha(), va=label.get_va())
        #roto el label el angulo obtenido en angles
        lab.set_rotation(angle)

    #Elimino las etiquetas por defecto del chart
    ax.set_xticklabels([])

    ax.grid(True)


# ------- PARTE 2: Aplico la funcion para cada persona

for x in range(0, 10):
    stats = np.array(pd.to_numeric(events.reset_index().iloc[x,1:12].values))
    nombrePersona = events.reset_index().iloc[x,0]
    hacerRadarChart( labels= np.array(events.columns), stats=stats, nombrePersona = nombrePersona)

#Me quedo con el mes de Mayo
persons1 = troca_persons[(troca_persons['timestamp'].str.contains("2018-05"))]

#Me quedo con Brazil y visited site
persons = persons1[(persons1['event'] == 'visited site') & (persons1['country'] == 'Brazil')]
persons['timestamp'] = pd.to_datetime(persons['timestamp'])

#Saco la cantidad de personas
cantidadDePersonasEnMayoBrazil = persons.groupby('country')['person'].count().values.sum()

#Cantidad de personas total en Mayo
persons2 = persons1[(persons1['event'] == 'visited site')]
totalPersonsEnMayo = persons2.groupby('country')['person'].count().values.sum()

#Saco la proporcion
(cantidadDePersonasEnMayoBrazil / totalPersonsEnMayo)*100
#Miro la cantidad de usuarios que visitan el site dependiendo como llegaron al sitio
#Obtengo los visitantes
visitas = troca_persons[(troca_persons['event'] == 'visited site')]
v = visitas[['person']]
personasQueVisitaron = v['person'].values

#Obtengo de esos visitantes los que entran en 'campaign source'
origenDeCampania = troca_persons[troca_persons['person'].isin(personasQueVisitaron)]
oC = origenDeCampania[['person','campaign_source']].dropna()
oc = oC.groupby('campaign_source')['person'].count().sort_values(ascending = False)
oc = pd.DataFrame(data = oc)
oc['personlog'] = np.log(oc['person'])
oc = oc.reset_index()
oc
#Pequenio grafico

fig, ax = plt.subplots(figsize=(10,7))
ax.plot(oc['campaign_source'].values, oc['personlog'].values)
plt.xticks(rotation=90)

plt.title('Visitas al sitio de acuerdo a la plataforma de Ads', fontsize = 18)
plt.xlabel('Plataformas de campania',fontsize = 18)
plt.ylabel('Cantidad (log)',fontsize = 18)

plt.show()



dispUtilizadoVisitas = troca_persons[(troca_persons['event'] == 'visited site') & (troca_persons['device_type'])]
dispUtilizadoVisitas1 = dispUtilizadoVisitas.groupby('device_type')['event'].count()
dispUtilizadoVisitas1.sort_values(ascending = False)

dispUtilizadoVisitas2 = troca_persons[['screen_resolution', 'device_type']]

trocaResolution = troca_persons[(troca_persons['event'] == 'visited site') & (troca_persons['screen_resolution']) ]
res = trocaResolution [['device_type','screen_resolution','event']]
#Me quedo con los 30 con mas visitas
res1 = res.groupby('screen_resolution')['event'].count().sort_values(ascending = False).head(30)

res1
troca_persons['search_term'].dropna().value_counts()
navegador = troca_persons[(troca_persons['event'] == 'visited site') & (troca_persons['browser_version'])]
#Me quedo con los 30 mas importantes
navegador.groupby('browser_version')['event'].count().sort_values(ascending = False).head(30)
sistOp = troca_persons[(troca_persons['event'] == 'visited site') & (troca_persons['operating_system_version'])]
#Me quedo con los 30 mas importantes
reso.groupby('operating_system_version')['event'].count().sort_values(ascending = False).head(30)
# Analizo la cantidad de usuarios nuevos y los recurrentes
usersNew =  troca_persons[['person','new_vs_returning']]
#Elimino los nulos
usersNew = usersNew.dropna()
usersNew.groupby('new_vs_returning')['person'].count()
leads = troca_persons[(troca_persons['event'] == 'lead')]
#Me quedo con los 30 mas importantes
l = leads.groupby('person')['event'].count().sort_values(ascending = False).head(30)
l
#Cantidad de leads
l.sum()
buscados = troca_persons[troca_persons['event'] == 'searched products']
#Cantidad de busquedas desde el sitio
cant = buscados.groupby('person')['event'].count().sum()
cant
#Proporcion de cantidad de busquedas desde el sitio sobre la cantidad total de visitas del sitio
visitas = troca_persons[troca_persons['event'] == 'visited site']
total = visitas.groupby('person')['event'].count().sum()

prop = cant / total
#Proporcion en porcentaje
print(np.round(prop * 100) ,'%')

#Paginas visitadas en mayo
visitsPages = troca_persons[(troca_persons['event'] == 'staticpage') & (troca_persons['timestamp'].str.contains("2018-05"))]
VP = visitsPages.groupby('person')['event'].count().sum()
#Paginas visitadas en el total del semestre
totalPages = troca_persons[(troca_persons['event'] == 'staticpage')]
TP = totalPages.groupby('person')['event'].count().sum()
#Proporcion:
VP / TP
listados = troca_persons[(troca_persons['event'] == 'brand listing')]
cant1 = listados.groupby('person')['event'].count().sum()
cant1
#Cantidad de listados visitados por los users
#cantidadDeListados vistos en Mayo
listMayo = troca_persons[(troca_persons['event'] == 'brand listing') & (troca_persons['timestamp'].str.contains("2018-05"))]
cant2 = listMayo.groupby('person')['event'].count().sum()
cant2
#Cantidad de listados visitados por los users
# Proporcion
cant2 / cant1
checks = troca_persons[(troca_persons['event'] == 'checkout')]
checks.groupby('person')['event'].count().sum()
#Cantidad total de checkouts
convs = troca_persons[(troca_persons['event'] == 'conversion')]
convs.groupby('person')['event'].count().sum()
#Cantidad total de conversiones
check = persons1[persons1['event'] == 'checkout']
cnv = persons1[persons1['event'] == 'conversion']
cnv1 = cnv.groupby('person')['event'].count()
checks = check.groupby('person')['event'].count()

#Dataframe con las personas que convirtieron en Mayo de todos los paises
cnv2 = cnv1.reset_index()
cnv2.rename(columns={'event': 'cantidad_Conversiones'}, inplace=True)
cnv2 = cnv2.sort_values(by='cantidad_Conversiones',ascending = False)
#Dataframe con las personas que realizaron Checkouts en Mayo de todos los paises
checks1 = checks.reset_index()
checks1.rename(columns={'event': 'cantidad_Checkouts'}, inplace=True)
checks1 = checks1.sort_values(by='cantidad_Checkouts',ascending = False)

#hago el left join
checksVsVisits = pd.merge(checks1, cnv2,on = 'person', how='left').dropna()


g = sns.jointplot( "cantidad_Checkouts", "cantidad_Conversiones", data=checksVsVisits, kind="reg",
                  xlim=(0, 60), ylim=(0, 12), color="m", height=7)

plt.title('Conversiones por Checkouts en Mayo ', fontsize = 8)
plt.xlabel('Checkouts)',fontsize = 18)
plt.ylabel('Conversiones',fontsize = 18)

disp = persons.groupby('person')['device_type'].value_counts().sort_values(ascending = False)
disp = pd.DataFrame(data = disp)
disp.rename(columns={'device_type': 'cantidad_visitas'}, inplace=True)
disp = disp.reset_index()
dispMay = disp.sort_values(by = 'cantidad_visitas',ascending = False)
dispMay
dispMay.groupby('device_type')['cantidad_visitas'].sum().sort_values(ascending = False)
#Creo un df con las personas que convirtieron en mayo (Brazil)
convMayo =  persons1[(persons1['event'] == 'conversion') & (persons1['person'].isin(dispMay['person'].values))]
convM =  convMayo.groupby('person')['event'].count()
convM = convM.reset_index()
convM.rename(columns={'event': 'cantidad_Conversiones'}, inplace=True)
convM
#joineo para tener las conversiones por device_type 
nuevoDf = pd.merge(dispMay, convM,on = 'person', how='left').dropna()
nuevoDf
nuevoDf['visitas_log'] = np.log(nuevoDf['cantidad_visitas'])
plt.figure(figsize=(10,8))
ax = sns.scatterplot(x="visitas_log", y="cantidad_Conversiones", hue="device_type",style="device_type",data=nuevoDf,\
                     )
plt.title('Conversiones en Mayo en funcion de las visitas al sitio desde Brasil ', fontsize = 18)
plt.xlabel('visitas (log)',fontsize = 18)
plt.ylabel('conversiones',fontsize = 18)
#Creo un df con las personas que realizaron checkouts en mayo (Brazil)
checksMayo =  persons1[(persons1['event'] == 'checkout') & (persons1['person'].isin(dispMay['person'].values))]
check2 =  checksMayo.groupby('person')['event'].count()
check2 = check2.reset_index()
check2.rename(columns={'event': 'cantidad_checkouts'}, inplace=True)
check2
#joineo para tener todo junto por device_type y persona
join2 = pd.merge(nuevoDf, check2,on = 'person', how='outer').dropna()
join2
#Aqui un pequenio grafico de los checkouts en funcion del dispositivo utilizado
sns.catplot(x="device_type", y="cantidad_checkouts", kind="violin", data=join2);
plt.title('Checkouts en Mayo segun device_type ', fontsize = 18)
plt.xlabel('Dispositivo del cual se accede',fontsize = 18)
plt.ylabel('Cantidad de checkouts',fontsize = 18)

sns.catplot(x="device_type", y="cantidad_Conversiones", kind="violin", data=join2);
plt.title('Conversiones en Mayo segun device_type ', fontsize = 18)
plt.ylabel('conversiones',fontsize = 18)
plt.xlabel('Dispositivo del cual se accede',fontsize = 18)
origenCampania = origenCampania[['person','campaign_source']]
join3 = pd.merge(origenCampania,join2,on = 'person', how='left').dropna()
plt.figure(figsize=(10,8))
ax= sns.lineplot(x="cantidad_checkouts", y="cantidad_Conversiones",hue="campaign_source",data=join3)

plt.title('Checkouts vs conversiones en funcion del trafico al que pertenecen (Mayo)', fontsize = 18)
plt.xlabel('Cantidad de checkouts',fontsize = 18)
plt.ylabel('Cantidad de conversiones',fontsize = 18)