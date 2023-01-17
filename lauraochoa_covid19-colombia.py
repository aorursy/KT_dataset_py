
#Data Analysis
import pandas as pd


#Visualization
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
%matplotlib inline

import plotly
import plotly.express as px
import folium   #This allows us to use maps


from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
#data from the source
df=pd.read_csv("../input/covid-colombia/Casos_positivos_de_COVID-19_en_Colombia (2).csv")  #Available in www.datos.gov.co

#Coordenadas de las capitales de los departamentos de Colombia
Col_coordinates=pd.read_csv("../input/covid-colombia/coordenadas_col.csv",header=None,names=['Departamento_distrito','Latitud','Longitud']) 
#Available in https://www.geodatos.net/coordenadas/colombia/


#Se visualizan las primeras 5 filas del dataframe
#First 5 rows of df
df.head()

Col_coordinates.head()
#Como los datos se ovtuvieron en formato json, todos son tipo objeto
#All the columns are object type

df.dtypes
#Define columns names 
df.columns=['ID_caso', 'Fecha_notificacion', 'Cod_DIVIPOLA', 'Ciudad_ubicacion',
       'Departamento_distrito', 'Atencion', 'Edad', 'Sexo', 'Tipo', 'Estado',
       'Pais_procedencia', 'FIS', 'Fecha_muerte', 'Fecha_diagnostico',
       'Fecha_recuperado', 'fecha reporte web']     
#States that have had cases
df['Departamento_distrito'].unique()
#Rename States 
df['Departamento_distrito']=df['Departamento_distrito'].replace({'Bogotá D.C.':'Bogota','Valle del Cauca':'Valle_del_Cauca','Cartagena D.T. y C.':'Cartagena',
                                    'Barranquilla D.E.':'Barranquilla','Santa Marta D.T. y C.':'Santa_Marta','Archipiélago de San Andrés Providencia y Santa Catalina':'San_Andres',
                                    'Atlántico':'Atlantico','Boyacá':'Boyaca', 'Córdoba':'Cordoba', 'Bolívar':'Bolivar','Buenaventura D.E.':'Buenaventura', 'Chocó':'Choco',
                                                               'Caquetá':'Caqueta','Vaupés':'Vaupes','Norte de Santander':'Norte_de_Santander','Quindío':'Quindio'
                                                              })

df.head()
#The dates were objects so for the analysis it's better to change them to datetime type

df['Fecha_muerte'] = pd.to_datetime(df['Fecha_muerte'], errors='coerce')
df['Fecha_diagnostico']=pd.to_datetime(df['Fecha_diagnostico'],errors='coerce')
df['Fecha_recuperado']=pd.to_datetime(df['Fecha_recuperado'],errors='coerce')

#Change Age to integer
df['Edad']=df['Edad'].astype(int)
df['Estado'].unique() 
df['Estado']=df['Estado'].replace({'leve':'Leve'})
df.isnull().sum() #There are some null values in 'Estado', this is because not all the cases has a State associated. 
#Like was expected, Recovery date and Death date also has null values.
df.head()
#Cases per day
Cases=df.groupby('Fecha_diagnostico')['ID_caso'].count().to_frame().reset_index().rename({'ID_caso':'Casos'},axis='columns')
Cases['Acum']=Cases['Casos'].cumsum()
figure1=px.bar(Cases,x='Fecha_diagnostico',y='Casos',title='Casos por dia',color_discrete_sequence =['#f38181'])
figure1.show()
#Accumulated cases per day
figure2=px.bar(Cases,x='Fecha_diagnostico',y='Acum',title='Acumulado de casos',color_discrete_sequence =['#f38181'])
figure2.show()

#Deaths per day
Deaths=df.groupby('Fecha_muerte')['ID_caso'].count().to_frame().reset_index().rename({'ID_caso':'Muertes'},axis='columns')
figure3=px.bar(Deaths, x='Fecha_muerte',y='Muertes',title='Muertes por día')
figure3.show()
#Accumulated deaths per day
Deaths['Acum']=Deaths['Muertes'].cumsum()
figure4=px.bar(Deaths, x='Fecha_muerte',y='Acum',title='Acumulado muertes')
figure4.show()
#Recovered people
Recovered=df.groupby('Fecha_recuperado')['ID_caso'].count().to_frame().reset_index().rename({'ID_caso':'Recuperados'},axis='columns')
Recovered['Acum']=Recovered['Recuperados'].cumsum()
figure5=px.bar(Recovered, x='Fecha_recuperado',y='Recuperados',title='Recuperados por dia',color_discrete_sequence = ['#a3de83'])
figure6=px.bar(Recovered, x='Fecha_recuperado',y='Acum',title='Acumulado recuperados',color_discrete_sequence = ['#a3de83'])
fig=make_subplots(rows=3, cols=2, shared_xaxes=False, horizontal_spacing=0.1,
                  subplot_titles=('No. casos por dia','No. casos acumulado',
                  'No. muertes por dia', 'No. muertes acumulado',
                  'No. recuperados por dia', 'No. recuperados acumulado'
                 ))

fig.add_trace(figure1['data'][0], row=1, col=1)
fig.add_trace(figure2['data'][0], row=1, col=2)
fig.add_trace(figure3['data'][0], row=2, col=1)
fig.add_trace(figure4['data'][0], row=2, col=2)
fig.add_trace(figure5['data'][0], row=3, col=1)
fig.add_trace(figure6['data'][0], row=3, col=2)

fig.show()
#Numero de casos  (Cases per City)
top_15=df.groupby('Ciudad_ubicacion')['ID_caso'].count().sort_values().tail(15).to_frame().reset_index().rename({'ID_caso':'Casos'},axis='columns')
figure7=px.bar(top_15,x='Casos',y='Ciudad_ubicacion',orientation='h',color_discrete_sequence =['#f38181'])
#Nuemro de muertes  (Deaths per City)
top15_m=df.loc[df['Fecha_muerte'].notnull()].groupby('Ciudad_ubicacion')['ID_caso'].count().sort_values().tail(15).to_frame().reset_index().rename({'ID_caso':'Muertes'}, axis='columns')
figure8=px.bar(top15_m,x='Muertes',y='Ciudad_ubicacion',orientation='h')
#Numero de recuperados  (Recovered per city)
top15_r=df.loc[df['Fecha_recuperado'].notnull()].groupby('Ciudad_ubicacion')['ID_caso'].count().sort_values().tail(15).to_frame().reset_index().rename({'ID_caso':'Recuperados'}, axis='columns')
figure9=px.bar(top15_r,x='Recuperados',y='Ciudad_ubicacion',orientation='h',color_discrete_sequence = ['#a3de83'])
fig2=make_subplots(rows=1, cols=3, shared_xaxes=False, horizontal_spacing=0.1,
                  subplot_titles=('Infectados','Muertes',
                  'Recuperados'))

fig2.add_trace(figure7['data'][0], row=1, col=1)
fig2.add_trace(figure8['data'][0], row=1, col=2)
fig2.add_trace(figure9['data'][0], row=1, col=3)
#Cities with more than 1000 cases
Cities=df.groupby(['Ciudad_ubicacion','Fecha_diagnostico'])['ID_caso'].count().to_frame().reset_index().rename({'ID_caso':'Casos'},axis='columns')
Cities['Acum']=Cities.groupby('Ciudad_ubicacion')['Casos'].cumsum()        
temp1=Cities.loc[Cities['Acum'] > 1000]
Top_cities_names=temp1['Ciudad_ubicacion'].unique()
Top_cities=Cities.loc[Cities['Ciudad_ubicacion'].isin(Top_cities_names)]
#Number of days before reaching 1000 cases 
temp1=Top_cities.groupby('Ciudad_ubicacion')
first_cases=temp1.head(1).reset_index()  #first
k_cases=Top_cities.loc[Top_cities['Acum'] >= 1000].groupby('Ciudad_ubicacion').head(1).reset_index()
Days_k_cases=(k_cases['Fecha_diagnostico'].sub(first_cases['Fecha_diagnostico'],axis=0)).dt.days
d={'Ciudad_ubicacion':Top_cities_names,'Days': Days_k_cases}
Days_k_cases=pd.DataFrame(data=d)
figure12=px.bar(Days_k_cases, x='Ciudad_ubicacion',y='Days',title='# Días en superar los 1000 casos ',color='Ciudad_ubicacion')
figure12.show()
#Breve descripción de la edad de los contagiados
df['Edad'].describe()
#Simetría- distribución de las edades de los contagiados
#Symmetry -distribution of the ages of the infected
Ages=df.groupby('Edad')['ID_caso'].count().to_frame().reset_index().rename({'ID_caso':'Casos'},axis='columns')
figure10=px.box(Ages,y='Edad')
figure10.show()

#Promedio de edad en los diferentes Departamentos y según el estado de cada contagiado
#Average age in the different Departments and according to the state of health
table_age=pd.pivot_table(df,index=['Departamento_distrito'],values=['Edad'],columns=['Estado'])
table_age
#Cases per Age
figure11=px.bar(Ages,x='Edad',y='Casos',color='Casos', title='Edades')
figure11.show()
#Sexo de los contagiados en el país
#Gender percentage
gender_d=df.groupby('Sexo')['Sexo'].value_counts()
fig=plt.figure(figsize=(4, 2), dpi=200)
axes=fig.add_axes([0.5,0.5,1,1])

axes.pie(gender_d,labels=['F','M'],autopct='%1.1f%%', textprops={'fontsize': 9})
axes.set_title("Sexo de los contagiados en todo el país", fontdict={'fontsize': 7})
plt.show()
gender_d=df.groupby(['Ciudad_ubicacion'])['Sexo'].value_counts()
gender_d

gender_d=pd.DataFrame(data=gender_d)
#Distribución del sexo de los contagiados en las ciudades con más casos
gender_d_top=gender_d.loc[gender_d.index.get_level_values(0).isin(Top_cities_names)]
list_names=gender_d_top.index.get_level_values(0).unique()
gender_d_top.rename(columns={'Sexo':'Cant'},inplace=True)
f_city=gender_d_top.loc[[list_names[0]]]
s_city=gender_d_top.loc[[list_names[1]]]
t_city=gender_d_top.loc[[list_names[2]]]
fth_city=gender_d_top.loc[[list_names[4]]]

#Gender distribution in the cities with more cases
f, a = plt.subplots(2,2)
f.set_size_inches(19.5, 12.5)
f_city['Cant'].plot(kind='pie',ax=a[0,0],autopct='%1.1f%%')
s_city.Cant.plot(kind='pie',ax=a[0,1],autopct='%1.1f%%')
t_city.Cant.plot(kind='pie',ax=a[1,0],autopct='%1.1f%%')
fth_city.Cant.plot(kind='pie',ax=a[1,1],autopct='%1.1f%%')
#State of health
df_estado=df.groupby('Estado')['ID_caso'].count().to_frame().rename({'ID_caso':'Cant'},axis='columns')
figure12=px.pie(df_estado,values='Cant',names=df_estado.index,title='Estado de los contagiados')
figure12.show()
mapa_info=df.groupby('Departamento_distrito')['ID_caso'].count().to_frame().reset_index().rename({'ID_caso':'Casos'}, axis='columns')
df_merge=pd.merge(Col_coordinates,df,on='Departamento_distrito',how='right')
#Merging between covid df and coordinates df
mapa_info=pd.merge(Col_coordinates,mapa_info,on='Departamento_distrito')
mapa_info.drop_duplicates();
#Covid-19 in Colombia states 
map=folium.Map(location=[4.5709,-74.2973],zoom_start=5,tiles='Stamenterrain',width='70%',height='70%')

for lat,long,value, name in zip(mapa_info['Latitud'],mapa_info['Longitud'],mapa_info['Casos'],mapa_info['Departamento_distrito']):
    folium.CircleMarker([lat,long],radius=value*0.01,popup=('<strong>Capital_Departamento</strong>: '+str(name).capitalize()+'<br>''<strong>Total_Casos</strong>: ' + str(value)+ '<br>'),color='red',fill_color='red',fill_opacity=0.3).add_to(map)
    
    

map
#Home,Recovered,Hospital,Hospital ICU
df_atencion=df.groupby(['Atencion'])['ID_caso'].count().to_frame().reset_index().rename({'ID_caso':'Casos'}, axis='columns')
figure13=px.pie(df_atencion,values='Casos',names='Atencion',hole=0.3)
figure13.show()
#Promedio de días de recuperación por Departamento y Distrito
#Average of recovery days per State
df_recovered=df[df['Fecha_recuperado'].notnull()]
df_recovered['Dias_recuperacion']=(df_recovered['Fecha_recuperado']-df_recovered['Fecha_diagnostico']).dt.days
df_deaths=df[df['Fecha_muerte'].notnull()]
df_recovered.groupby('Departamento_distrito').mean()
#Recovery rate
print("La tasa de recuperación es {0:.2%}".format(len(df_recovered)/len(df)))
#Death rate
print("La tasa de mortalidad es {0:.2%}".format(len(df_deaths)/len(df)))
#Recovery rate
r_rate=((df_recovered.groupby('Departamento_distrito')['ID_caso'].count())/(df.groupby('Departamento_distrito')['ID_caso'].count())).sort_values().to_frame().rename({'ID_caso':'Porcentaje'},axis='columns')*100
figure14=px.bar(r_rate,x=r_rate.index,y='Porcentaje',title='Tasa de recuperación')
figure14.show()