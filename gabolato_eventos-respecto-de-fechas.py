import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
fechaYHora = pd.read_csv('../input/events.csv', low_memory = False)
fechaYHora.head()
#Convierto 'timestamp' a formato fecha 
fechaYHora['timestamp'] = pd.to_datetime(fechaYHora['timestamp'])
#Cantidad de eventos por mes
fechaYHora['timestamp'].dt.month.value_counts().sort_index()
#Cantidad de eventos por dia
fechaYHora['timestamp'].dt.day.value_counts().sort_index()

#Cantidad de personas por mes
personasPorDia = fechaYHora[['person','timestamp']]

#Agrego una columna dias
personasPorDia['dias'] = personasPorDia['timestamp'].dt.day

ppd = personasPorDia.groupby('dias')['person'].count()
totalVisitas = ppd.sum()
ppdia = ppd.reset_index().rename(columns = {'person':'cantidad'})
#Aqui ya tengo el porcentaje en una columna
ppdia['porcentaje'] = (ppdia['cantidad']/totalVisitas)*100

plt.figure(figsize=(16,12))
ax = sns.regplot(x="dias", y="porcentaje", data=ppdia,\
               x_estimator=np.mean, logx=True, truncate=True)
plt.title('Porcentaje promedio de personas por dia en Trocafone', fontsize = 18)
plt.xlabel('Dias', fontsize = 18)
plt.ylabel('Porcentaje', fontsize = 18)

#Cantidad de conversiones por mes

mesesYConversiones = pd.DataFrame( data = ((fechaYHora[(fechaYHora['event'] == 'conversion')])['timestamp']\
                       .dt.month.value_counts().sort_index()))
mesesYConversiones = mesesYConversiones.reset_index()
mesesYConversiones.rename(columns={'index': 'mes', 'timestamp':'conversiones'}, inplace=True)
mesesYConversiones.plot(x='mes', y='conversiones', colormap="autumn")
# Area plot
x=mesesYConversiones.mes
y=mesesYConversiones.conversiones
plt.fill_between(x, y)

plt.title('Conversiones por mes 2018', fontsize = 18)
plt.xlabel('Meses', fontsize = 18)
plt.ylabel('Conversiones', fontsize = 18)

#Cantidad de conversiones por dia en el total de meses
diasYConversiones = pd.DataFrame( data = ((fechaYHora[(fechaYHora['event'] == 'conversion')])['timestamp']\
                      .dt.day.value_counts().sort_index()))
diasYConversiones = diasYConversiones.reset_index()
diasYConversiones.rename(columns={'index': 'dia', 'timestamp':'conversiones'}, inplace=True)

diasYConversiones.plot(x='dia', y='conversiones')
plt.title('Conversiones por dia en el total de Enero-2018 a Junio-2018', fontsize = 18)
plt.xlabel('Dias', fontsize = 18)
plt.ylabel('Conversiones', fontsize = 18)

#Agrego columnas por separado
fechaYHora['dia'] =  fechaYHora['timestamp'].dt.day
fechaYHora['mes'] =  fechaYHora['timestamp'].dt.month
fechaYHora['anio'] =  fechaYHora['timestamp'].dt.year
enero = fechaYHora[(fechaYHora['event'] == 'conversion') & (fechaYHora['mes'] == 1)]
enero.groupby('mes')['event'].value_counts()

#Se puede ver un total de 63 conversiones en el mes de Enero
junio = fechaYHora[(fechaYHora['event'] == 'conversion') & (fechaYHora['mes'] == 6)]
junio.groupby('mes')['event'].value_counts()
#Se puede ver un total de 103 conversiones en el mes de junio
enero = enero[['dia','event']]
january = enero.groupby('dia')['event'].count()
january.plot(x=january.index , y = january.values, cmap = "Dark2")

junio = junio[['dia','event']]
june = junio.groupby('dia')['event'].count()
june.plot(x=june.index , y = june.values, cmap = "autumn")

plt.title('Conversiones por dia enero("azul") y junio("rojo")', fontsize = 18)
plt.xlabel('Dias', fontsize = 18)
plt.ylabel('Conversiones', fontsize = 18)
fechaYHora.columns
#creo una columna hora
fechaYHora['hora'] = fechaYHora['timestamp'].dt.hour
#Filtro por pais: Brazil

brazilTimeVisits = fechaYHora[(fechaYHora['country'] == 'Brazil')]
brazilVisits = brazilTimeVisits[['country','hora']].groupby('hora')['country'].count()

brazilVisits
#Vista de eventos respecto a horas en Brasil

labels = brazilVisits.index
stats = np.log(brazilVisits.values)

def drawRadarChart(labels,stats,title):
  
    #Seteo los angulos de las etiquetas
    angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    #Creo un plot cerrado
    stats=np.concatenate((stats,[stats[0]]))
    angles=np.concatenate((angles,[angles[0]]))

    fig= plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, stats, 'o-', linewidth=2)
    ax.fill(angles, stats, alpha=0.25)
    ax.set_thetagrids(angles * 180/np.pi, labels)

    ax.grid(True)

    fig.text(0.5, 1.00, title,horizontalalignment='center', color='black', weight='bold', size='large')
drawRadarChart(labels = labels, stats =stats, title = "Distribucion de cantidad de  eventos respecto a las horas en Brasil (log)")
convArgentina = fechaYHora[(fechaYHora['event'] == 'visited site') & (fechaYHora['country'] == 'Argentina')]
cA = pd.DataFrame(data = convArgentina.groupby('hora')['event'].count())
cA.rename(columns = { 'event' : 'visitasPorHora'},inplace = True)
convArg = cA.reset_index()
ax = sns.lineplot(x="hora", y="visitasPorHora", data=convArg)
plt.title('Horas de visitas al sitio en Argentina', fontsize = 18)
brazilCities = fechaYHora[(fechaYHora['country'] == 'Brazil')  & (fechaYHora['event'])]
#Filtro por ciudades mas importantes
braC = brazilCities['city'].dropna().value_counts()
braC

argentinaCities = fechaYHora[(fechaYHora['country'] == 'Argentina') & (fechaYHora['event'])]
argC = argentinaCities['city'].dropna().value_counts()
argC
usaCities = fechaYHora[(fechaYHora['country'] == 'United States')  & (fechaYHora['event'])]
usaC = usaCities['city'].dropna().value_counts()
usaC
#Me quedo con las ciudades de mayor relevancia(mayor cantidad en eventos)
ciudades = ["São Paulo", "Rio de Janeiro", "Buenos Aires", "Mountain View", "The Bronx"]
#Filtro del dataframe por las ciudades
nuevoCities = fechaYHora[(fechaYHora['city'].isin(ciudades))]
#Me quedo ademas con las personas que acceden desde tales ciudades
nuevoCities = nuevoCities[['mes','person','city',]]

#Armo otro df y me quedo con las las conversiones y personas
nuevoConversiones = fechaYHora[(fechaYHora['person'].isin(nuevoCities['person'].values)) &\
                               (fechaYHora['event'] == 'conversion') ]
nuevoConversiones.sort_values(by = 'mes')

nuevoConversiones =  nuevoConversiones[['event','mes','person']]

nueC = nuevoConversiones.groupby('person')['event'].count().reset_index()
nueC.rename(columns = {'event':'cantidadConversiones'},inplace = True)

#join
join = pd.merge(nueC, nuevoCities, how='right', on='person')
join.dropna()['city'].value_counts()


#plot
g = sns.FacetGrid(join, col='city', hue='city', col_wrap=4, )
 
# Agrego el line plot 
g = g.map(plt.plot, 'mes', 'cantidadConversiones')
 
# Relleno el area entre lineas
g = g.map(plt.fill_between, 'mes', 'cantidadConversiones', alpha=0.2).set_titles("{col_name} city")
 
# Titulo de cada 'facet'
g = g.set_titles("{col_name}")

plt.subplots_adjust(top=.92)

g = g.fig.suptitle('Evolucion de las conversiones en el semestre en las ciudades mas importantes', fontsize = 15)
 

