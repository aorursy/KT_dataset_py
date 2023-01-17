import numpy as np

import pandas as pd

import seaborn as sb

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV

from sklearn.kernel_ridge import KernelRidge

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_squared_error

import plotly.graph_objs as go

import datetime

import plotly.express as px

import folium

import warnings

import folium 

from folium import plugins

from math import sqrt

from sklearn.preprocessing import PolynomialFeatures



import matplotlib.pyplot as plt

import seaborn as sns



import plotly.express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

from plotly.graph_objs import *

from plotly.subplots import make_subplots



#Optimizacion bayesiana con hyperopt

from hyperopt import STATUS_OK

from timeit import default_timer as timer

from hyperopt import tpe

from hyperopt import Trials

from hyperopt import fmin

from hyperopt import hp, tpe

from hyperopt.fmin import fmin

from hyperopt import hp, tpe, Trials, STATUS_OK

from hyperopt.fmin import fmin

from hyperopt.pyll.stochastic import sample

import ast



import warnings



warnings.filterwarnings('ignore')



%matplotlib inline



#Funciones:



def rmsle_cv(model,x_test,y_test):

    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(x_test)

    rmse= np.sqrt(-cross_val_score(model, x_test, y_test, scoring="neg_mean_squared_error", cv = kf))

    return(rmse)
data_chile = pd.read_csv('https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto3/CasosTotalesCumulativo.csv')





ultima_fecha_cl = data_chile.columns

ultima_fecha_cl= ultima_fecha_cl[-1]

print("Ultima Actualización:",ultima_fecha_cl)
data_chile_r = pd.read_csv('https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto5/TotalesNacionales.csv')

grupo_fallecidos = pd.read_csv('https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto10/FallecidosEtario.csv')

data_crec_por_dia = pd.read_csv('https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto5/TotalesNacionales.csv')

casos_por_comuna = pd.read_csv('https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto25/CasosActualesPorComuna.csv')

sintomas = pd.read_csv('https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto21/SintomasHospitalizados.csv')







fechas_chile_crec = data_crec_por_dia.columns[-1]

fechas_chile = data_crec_por_dia.loc[:, '2020-03-03': fechas_chile_crec]

fechas_chile = fechas_chile.keys()



fechas_death = data_crec_por_dia.columns[-1]



death_cl = grupo_fallecidos.loc[:, '2020-04-09': ultima_fecha_cl]

dates_d = death_cl.keys()

c_d = []



for i in dates_d:

   

    c_d.append(grupo_fallecidos[i].sum())

    

confirmed_chile = data_chile.loc[:, '2020-03-03': ultima_fecha_cl]

dates_chile = confirmed_chile.keys()

days_chile = np.array([i for i in range(len(dates_chile))]).reshape(-1, 1)



casos_chile = []



for i in dates_chile:

   

    casos_chile.append(data_chile[i].iloc[16])

    



casos_por_dia_sintomas = []

casos_por_dia_asintomaticos = []

fallecidos_por_dia =[]

recuperados_por_dia=[]

casos_por_dia_totales =[]

activos_por_dia = []

casos_totales_acum_list = []

for i in fechas_chile:

    

    f = data_crec_por_dia[data_crec_por_dia['Fecha']=='Fallecidos'][i].sum()

    c_sintomas = data_crec_por_dia[data_crec_por_dia['Fecha']=='Casos nuevos con sintomas'][i].sum()

    c_asintomaticos = data_crec_por_dia[data_crec_por_dia['Fecha']=='Casos nuevos sin sintomas'][i].sum()



    c_t = data_crec_por_dia[data_crec_por_dia['Fecha']=='Casos nuevos totales'][i].sum()



    r = data_crec_por_dia[data_crec_por_dia['Fecha']=='Casos recuperados'][i].sum()



    activos = data_crec_por_dia[data_crec_por_dia['Fecha']=='Casos activos'][i].sum()

    casos_totales_acum = data_crec_por_dia[data_crec_por_dia['Fecha']=='Casos totales'][i].sum()





    casos_por_dia_sintomas.append(c_sintomas)

    casos_por_dia_asintomaticos.append(c_asintomaticos)



    casos_por_dia_totales.append(c_t)

    fallecidos_por_dia.append(f)

    recuperados_por_dia.append(r)

    activos_por_dia.append(activos)

    

    casos_totales_acum_list.append(casos_totales_acum)

    

data_death_date = pd.DataFrame({'Date':dates_d,'Death':c_d})



dates_cl_ = data_chile.drop(['Region'], axis=1)



datos_chile_cd_date = pd.DataFrame({'Date':dates_cl_.columns,'Cases':data_chile.iloc[16,1:].values})



data_cs_cl = pd.merge(datos_chile_cd_date, data_death_date, on='Date', how='outer')

data_cs_cl = data_cs_cl.replace(np.nan, 0)    
data_chile_map = data_chile.drop([16,9],axis=0)

data_chile_map.head()
print("ACTUALIZADO "+data_chile.columns[-1])
data_chile_map = data_chile_map.reset_index()

total =len(data_chile.columns)



# Adding Location data (Latitude,Longitude)

locations = {

    "Arica y Parinacota" : [-18.4745998,-70.2979202],

    "Tarapacá" : [-20.2132607,-70.1502686],

    "Antofagasta" : [-23.6523609,-70.395401],

    "Atacama" : [-27.3667908,-70.331398],

    "Coquimbo" : [-29.9533195,-71.3394699],

    "Valparaíso" : [-33.0359993,-71.629631],

    "Metropolitana" : [-33.4726900,-70.6472400],

    "O’Higgins" : [-48.4862300,-72.9105900],

    "Maule" : [-35.5000000,-71.5000000],

    #"Ñuble" : [1,1],

    "Biobío" : [-37.0000000,-72.5000000],

    "Araucanía" : [-38.7396507,-72.5984192],

    "Los Ríos" : [-40.293129,-73.0816727],

    "Los Lagos" : [-41.7500000,-73.0000000],

    "Aysén" : [-45.4030304,-72.6918411],

    "Magallanes" : [-53.1548309,-70.911293]

        

   

}



data_chile_map["Lat"] = ""

data_chile_map["Long"] = ""

for index in data_chile_map.Region :

    data_chile_map.loc[data_chile_map.Region == index,"Lat"] = locations[index][0]

    data_chile_map.loc[data_chile_map.Region == index,"Long"] = locations[index][1]

    #print(locations[index][0])

    





chile = folium.Map(location=[-30.0000000,-71.0000000], zoom_start=4,max_zoom=6,min_zoom=4,height=500,width="80%")





for i in range(0,len(data_chile_map[data_chile[ultima_fecha_cl]>0].Region)):

    folium.Circle(

        location=[data_chile_map.loc[i,"Lat"],data_chile_map.loc[i,"Long"]],

        

    

     tooltip = "<h5 style='text-align:center;font-weight: bold'>"+data_chile_map.iloc[i].Region+"</h5>"+

                    "<hr style='margin:10px;'>"+

                    "<ul style='color: #444;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+

        "<li>Confirmed: "+str(data_chile_map.iloc[i,total])+"</li>"+

        "</ul>",

    

        radius=(int(np.log2(data_chile_map.iloc[i,total]+1)))*7000,



         color='#ff6600',

        fill_color='#ff8533',

        fill=True).add_to(chile)

chile
data_chile_r
num_cases_cl = data_chile.drop([16],axis=0)

num_cases_cl = num_cases_cl[ultima_fecha_cl].sum()

num_death =  grupo_fallecidos[ultima_fecha_cl].sum()

num_rec = data_chile_r.iloc[2,-1].sum()



num_active = data_chile_r.iloc[4,-1].sum()



datos_chile_rdca = pd.DataFrame({'Fecha':[ultima_fecha_cl],'Fallecidos(Acumulados)':[num_death],'Cases Confirmados (Acumulados)': [num_cases_cl],'Recuperados(Acumulados)':[num_rec],

                                 'Activos': [num_active] })

temp = datos_chile_rdca

temp.style.background_gradient(cmap='Pastel1')
data_total_cl_2 = pd.DataFrame({'Fecha': pd.to_datetime(fechas_chile),'Activos Sintomaticos': 

                                casos_por_dia_sintomas ,'Activos Asintomaticos':casos_por_dia_asintomaticos ,'Totales Activos':activos_por_dia, 

                                'Fallecidos(Acumulados)': fallecidos_por_dia,'Recuperados(Acumulados)':recuperados_por_dia,'Casos Totales(Acumulados)':casos_por_dia_totales })



confirmed = '#393e46' 

death = '#ff2e63' 

recovered = '#21bf73' 

active = '#fe9801' 





tm = temp.melt(id_vars="Fecha", value_vars=['Activos', 'Fallecidos(Acumulados)','Recuperados(Acumulados)'])

fig = px.treemap(tm, path=["variable"], values="value", height=400, width=600,

                 color_discrete_sequence=[recovered, active, death])



fig5 = go.Figure()

fig5.add_trace(go.Scatter(x=data_total_cl_2['Fecha'], y=data_total_cl_2['Totales Activos'], name='Activos',line_color='#fe9801'))

fig5.add_trace(go.Scatter(x=data_total_cl_2['Fecha'], y=data_total_cl_2['Recuperados(Acumulados)'], name='Recuperados(Acumulados)',line_color='green'))

fig5.layout.update(title_text='Activo vs. Recuperado '+fechas_chile[-1],xaxis_showgrid=False, yaxis_showgrid=False, width=700,

            height=600,font=dict(

            size=15,

            color="Black"    

        ))

fig5.layout.plot_bgcolor = 'White'

fig5.layout.paper_bgcolor = 'White'





fig5.show()

fig.show()
trace1 = go.Scatter(

                x=dates_chile,

                y=casos_totales_acum_list,

                name="Casos Acumulados",

                mode='lines+markers',

                line_color='orange')

trace2 = go.Scatter(

                x=dates_chile,

                y=fallecidos_por_dia ,

                name="Fallecidos Acumulados",

                mode='lines+markers',

                line_color='red')



trace3 = go.Scatter(

                x=dates_chile,

                y=recuperados_por_dia,

                name="Recuperados Acumulados",

                mode='lines+markers',

                line_color='green')





layout = go.Layout(template="ggplot2", width=800, height=500, title_text = '<b>Casos vs Repurados vs Fallecidos</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig = go.Figure(data = [trace1,trace2,trace3], layout = layout)

fig.show()
trace = go.Scatter(

                x=fechas_chile,

                y=casos_por_dia_totales,

                name="growth",

                mode='lines+markers',

                line_color='red')



layout = go.Layout(template="ggplot2", width=850, height=800, title_text = '<b>Numero de Casos por día</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig = go.Figure(data = [trace], layout = layout)

fig.show()
confirmados = data_chile.loc[:, '2020-03-03': ultima_fecha_cl]

dates_chile = confirmados.keys()

datos = data_chile[['Region',ultima_fecha_cl]].drop([16],axis=0)





#Grafico 1

titulo ='COVID-19: Total de Casos acumulados de COVID19'



fig = px.bar(datos.sort_values(ultima_fecha_cl),

            x='Region', y=ultima_fecha_cl,

            title=titulo,

            text=ultima_fecha_cl 

            

)

fig.update_xaxes(title_text="Regiones")

fig.update_yaxes(title_text="Numero de casos")



#Grafico 2



fig2 = px.bar(datos.sort_values(ultima_fecha_cl), 

             x=ultima_fecha_cl, y="Region", 

             title=titulo,

              text=ultima_fecha_cl, 

             orientation='h', 

             width=800, height=700)

fig2.update_traces(marker_color='#008000', opacity=0.8, textposition='inside')



fig2.update_layout(template = 'plotly_white')





#Grafico 3



total_chile = []

for i in dates_chile:

    total_chile.append(data_chile[data_chile['Region']=='Total'][i].sum())



data_total_cl = pd.DataFrame({'Date': dates_chile,'Total Cases': total_chile})



fig3 = px.bar(data_total_cl,x='Date', y='Total Cases', color='Total Cases', orientation='v', height=600,

             title=titulo, color_discrete_sequence = px.colors.cyclical.mygbm)



fig3.update_layout(plot_bgcolor='rgb(250, 242, 242)')









fig.show()

fig2.show()

fig3.show()
#https://www.kaggle.com/gatunnopvp/covid-19-in-brazil-prediction-updated-04-20-20

by_date = data_cs_cl[['Date','Cases','Death']]



layout = Layout(

    paper_bgcolor='rgba(0,0,0,0)',

    plot_bgcolor='rgba(0,0,0,0)',

    title="Cases and Deaths by Day"

)



fig = go.Figure(data=[

    

    go.Bar(name='Cases'

           , x=by_date['Date']

           , y=by_date['Cases']),

    

    go.Bar(name='Death'

           , x=by_date['Date']

           , y=by_date['Death']

           , text=by_date['Death']

           , textposition='outside')

])



fig.update_layout(barmode='stack')

fig['layout'].update(layout)



fig.show()
grupo_fallecidos = pd.read_csv('https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto10/FallecidosEtario.csv')

grupo_casos_genero= pd.read_csv('https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto16/CasosGeneroEtario.csv')

grupo_fallecidos
grupo_casos_genero.head()
fecha_grupo_edad = grupo_casos_genero.columns[-1]



grupo_edad = grupo_casos_genero.iloc[0:17,0]

data_casos_grupo_edad_mf = pd.DataFrame({'Grupo de edad': grupo_edad, fecha_grupo_edad : 0})



fila = 0

for grupo in data_casos_grupo_edad_mf['Grupo de edad']:

    suma_casos_MF = grupo_casos_genero[grupo_casos_genero['Grupo de edad'] == grupo].iloc[:,-1].sum()

    data_casos_grupo_edad_mf.iloc[fila,1] = suma_casos_MF

    fila=fila+1

data_casos_grupo_edad_mf.head()
titulo ='Casos por grupo de edad Fecha: '+fecha_grupo_edad



fig = px.bar(data_casos_grupo_edad_mf.sort_values(fecha_grupo_edad),

            x='Grupo de edad', y=fecha_grupo_edad,

            title=titulo,

            text=fecha_grupo_edad 

            

)

fig.update_xaxes(title_text="Regiones")

fig.update_yaxes(title_text="Numero de casos")



#colors = ['gold', 'darkorange', 'crimson','mediumturquoise', 'sandybrown', 'grey',  'lightgreen','navy','deeppink','purple']

trace1 = go.Pie(

                labels=data_casos_grupo_edad_mf['Grupo de edad'],

                values=data_casos_grupo_edad_mf[fecha_grupo_edad],

                hoverinfo='label+percent', 

                textfont_size=12,

                marker=dict(#colors=colors, 

                            line=dict(color='#000000', width=2)))

layout = go.Layout(width=700, height=500,title_text = '<b>Porcentaje de Casos acumulados por Grupo de Edad '+fecha_grupo_edad+'</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig2 = go.Figure(data = [trace1], layout = layout)



fig.show()

fig2.show()

casso_m = grupo_casos_genero[grupo_casos_genero['Sexo'] == 'M']

casso_f = grupo_casos_genero[grupo_casos_genero['Sexo'] == 'F']



#https://stackoverrun.com/es/q/8510875

#https://www.it-swarm.dev/es/python/anadir-una-fila-pandas-dataframe/1066944305/

f = casso_f.columns[1:]

data_suma_casos_f = pd.DataFrame(index=np.arange(0, 1), columns=(f) )



for date in data_suma_casos_f:

    data_suma_casos_f[date].iloc[0] = casso_f[date].sum()

data_suma_casos_f['Sexo'].iloc[0] = 'F'



m = casso_m.columns[1:]

data_suma_casos_m = pd.DataFrame(index=np.arange(0, 1), columns=(f) )



for date in data_suma_casos_m:

    data_suma_casos_m[date].iloc[0] = casso_m[date].sum()

data_suma_casos_m['Sexo'].iloc[0] = 'M'





union = pd.concat([data_suma_casos_m, data_suma_casos_f])



fig1 = go.Figure()



fig1.add_trace(go.Scatter(x=data_suma_casos_f.columns, y=data_suma_casos_f.iloc[0], name='F'))

fig1.add_trace(go.Scatter(x=data_suma_casos_m.columns, y=data_suma_casos_m.iloc[0], name='M'))



fig1.layout.update(title_text='Total de casos por genero : '+fecha_grupo_edad,xaxis_showgrid=False, yaxis_showgrid=False, width=700,

            height=600,font=dict(

            size=15,

            color="Black"    

        ))

fig1.layout.plot_bgcolor = 'White'

fig1.layout.paper_bgcolor = 'White'



colors = ['#2356E7','#CD0ADD']

n_f = union[fecha_grupo_edad].iloc[1]

n_m = union[fecha_grupo_edad].iloc[0]



plt.figure(figsize=(7,7))

plt.title("Porcentaje de casos por Genero",fontsize=20)

labels='M','F'

sizes=[n_m,n_f]

explode=[0.1,0.1]

colors=['skyblue','lightcoral']

plt.axis('equal')

plt.pie(sizes,labels=labels,colors=colors,explode=explode,autopct='%1.1f%%',shadow=True,startangle=90)

plt.legend(labels, loc="best") 







fig1.show()

plt.show()
from plotly.subplots import make_subplots

fecha_frupo_fallecidos=grupo_fallecidos.columns[-1]

fig = make_subplots(rows=1, cols=2)

colors = ['gold', 'darkorange', 'crimson','mediumturquoise', 'sandybrown', 'grey',  'lightgreen','navy','deeppink','purple']

trace1 = go.Pie(

                labels=grupo_fallecidos['Grupo de edad'],

                values=grupo_fallecidos[ultima_fecha_cl],

                hoverinfo='label+percent', 

                textfont_size=12,

                marker=dict(colors=colors, 

                            line=dict(color='#000000', width=2)))

layout = go.Layout(width=700, height=500,title_text = '<b>Porcentaje de personas fallecidas por grupo de edad : '+fecha_frupo_fallecidos+'</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig = go.Figure(data = [trace1], layout = layout)

fig.show()



colors = ['lightslategray']*10 

colors[2] = 'crimson'

trace2 = go.Bar(

            x=grupo_fallecidos['Grupo de edad'], 

            y=grupo_fallecidos[ultima_fecha_cl],

            text=grupo_fallecidos[ultima_fecha_cl],

            textposition='auto',

            marker_color=colors)

layout = go.Layout(template="ggplot2",width=700, height=500, )



fig = go.Figure(data = [trace2], layout = layout)

fig.show()
jovenes_fallecidos_chile = []



for i in dates_d :

    f_j = grupo_fallecidos[grupo_fallecidos['Grupo de edad']=='<=39'][i].sum()

    jovenes_fallecidos_chile.append(f_j)



trace = go.Scatter(

                x=grupo_fallecidos.iloc[:,1:].columns,

                y=jovenes_fallecidos_chile,

                name="Pacientes Criticos",

                mode='lines+markers',

                line_color='red')



layout = go.Layout(template="ggplot2", width=800, height=600,title_text = '<b>Numero de Fallecidos Jovenes (<=39 años) fallecidos: '+ ultima_fecha_cl+'</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig = go.Figure(data = [trace], layout = layout)

fig.show()
grupo_fallecidos
num_casos_jovenes_acumulados = data_casos_grupo_edad_mf.iloc[0:8,1].sum()

fallecidos_jovenes = pd.DataFrame({'Numero Casos <=39 años': [num_casos_jovenes_acumulados], 'Fallecidos' : grupo_fallecidos[fecha_grupo_edad].iloc[0].sum()})

fallecidos_jovenes
colors = ['green', 'red']

trace1 = go.Pie(

                labels=fallecidos_jovenes.columns,

                values=fallecidos_jovenes.iloc[0,:],

                hoverinfo='label+percent', 

                textfont_size=12,

                marker=dict(colors=colors, 

                            line=dict(color='#000000', width=2)))

layout = go.Layout(width=800, height=600,title_text = '<b>Porcentaje de personas <=39 fallecidas del total de personas <=39 contagiadas: '+fecha_grupo_edad+'</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig = go.Figure(data = [trace1], layout = layout)

fig.show()
num_casos_ancianos_acumulados = data_casos_grupo_edad_mf.iloc[12:17,1].sum()

fallecidos_ancianos = pd.DataFrame({'Numero Casos >=60 años': [num_casos_ancianos_acumulados], 'Fallecidos' : grupo_fallecidos[fecha_grupo_edad].iloc[3:7].sum()})

fallecidos_ancianos
colors = ['green', 'red']

trace1 = go.Pie(

                labels=fallecidos_ancianos.columns,

                values=fallecidos_ancianos.iloc[0,:],

                hoverinfo='label+percent', 

                textfont_size=12,

                marker=dict(colors=colors, 

                            line=dict(color='#000000', width=2)))

layout = go.Layout(width=800, height=600,title_text = '<b>Porcentaje de personas >=60 fallecidas del total de personas >=60 contagiadas: '+fecha_grupo_edad+'</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig = go.Figure(data = [trace1], layout = layout)

fig.show()
grupo_uci = pd.read_csv('https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto9/HospitalizadosUCIEtario.csv')

pacientes_criticos = pd.read_csv('https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto23/PacientesCriticos.csv')

grupo_uci_reg = pd.read_csv('https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto8/UCI.csv')

tipo_cama = pd.read_csv('https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto24/CamasHospital_Diario.csv')

pacientes_ventilacion = pd.read_csv('https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto30/PacientesVMI.csv')
trace = go.Scatter(

                x=pacientes_criticos.iloc[:,1:].columns,

                y=pacientes_criticos.iloc[0,1:],

                name="Pacientes Criticos",

                mode='lines+markers',

                line_color='red')



layout = go.Layout(template="ggplot2", width=800, height=600,title_text = '<b>Numero de Pacientes Criticos Fecha: '+ ultima_fecha_cl+'</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig = go.Figure(data = [trace], layout = layout)

fig.show()
fig = make_subplots(rows=1, cols=2)

colors = ['gold', 'darkorange', 'crimson','mediumturquoise', 'sandybrown', 'grey',  'lightgreen','navy','deeppink','purple']

trace1 = go.Pie(

                labels=grupo_uci['Grupo de edad'],

                values=grupo_uci[ultima_fecha_cl],

                hoverinfo='label+percent', 

                textfont_size=12,

                marker=dict(colors=colors, 

                            line=dict(color='#000000', width=2)))

layout = go.Layout(width=700, height=500,title_text = '<b>Porcentaje de personas hospitalizadas por grupo de edad : '+ultima_fecha_cl+'</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig = go.Figure(data = [trace1], layout = layout)



#Grafico 1

titulo ='Numero de personas Hopitalizadas por grupo de edad Fecha: '+ultima_fecha_cl



fig2 = px.bar(x=grupo_uci['Grupo de edad'], y=grupo_uci[ultima_fecha_cl],

            title=titulo,

           text=grupo_uci[ultima_fecha_cl]

            

)

fig2.update_xaxes(title_text="Age Group")

fig2.update_yaxes(title_text="Number of cases")



fig.show()

fig2.show()
titulo ='Número de pacientes hospitalizados según el tipo de cama.: '+ultima_fecha_cl



fig = make_subplots(rows=1, cols=2)

colors = ['gold', 'darkorange', 'crimson','mediumturquoise', 'sandybrown', 'grey',  'lightgreen','navy','deeppink','purple']

trace1 = go.Pie(

                labels=tipo_cama['Tipo de cama'],

                values=tipo_cama[ultima_fecha_cl],

                hoverinfo='label+percent', 

                textfont_size=12,

                marker=dict(colors=colors, 

                            line=dict(color='#000000', width=2)))

layout = go.Layout(width=700, height=500,title_text = '<b>Porcentaje de pacientes hospitalizados según el tipo de cama.: '+ultima_fecha_cl+'</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig = go.Figure(data = [trace1], layout = layout)





#Grafico 1



fig2 = px.bar(x=tipo_cama['Tipo de cama'], y=tipo_cama[ultima_fecha_cl],

            title=titulo,

           text=tipo_cama[ultima_fecha_cl]

            

)

fig2.update_xaxes(title_text="type of bed")

fig2.update_yaxes(title_text="Number cases")

fig.show()

fig2.show()
pacientes_ventilacion

trace2 = go.Scatter(

                x=pacientes_ventilacion.iloc[:,1:].columns,

                y=pacientes_ventilacion.iloc[0,1:],

                name="Pacientes VMI",

                mode='lines+markers',

                line_color='red')







layout = go.Layout(template="ggplot2", width=800, height=600,title_text = '<b>Paciente VMI '+ ultima_fecha_cl+'</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig = go.Figure(data = [trace2], layout = layout)

fig.show()
trace = go.Scatter(

                x=data_chile_r.iloc[:,1:].columns,

                y=data_chile_r.iloc[6,1:],

                name="Casos Nuevos Totales por día",

                mode='lines+markers',

                line_color='blue')



trace2 = go.Scatter(

                x=pacientes_ventilacion.iloc[:,1:].columns,

                y=pacientes_ventilacion.iloc[0,1:],

                name="Pacientes VMI",

                mode='lines+markers',

                line_color='red')



layout = go.Layout(template="ggplot2", width=850, height=600,title_text = '<b>Casos Nuevos vs Pacientes VMI '+ ultima_fecha_cl+'</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig = go.Figure(data = [trace,trace2], layout = layout)

fig.show()
fig = px.bar(x=grupo_uci_reg[ultima_fecha_cl], y=grupo_uci_reg['Region'], 

             title='Numero de personas Hospitalizadas por Región: '+ultima_fecha_cl,

             orientation='h',

             width=800, height=700)

fig.update_traces(marker_color='#008000', opacity=0.8, textposition='inside')



fig.update_layout(template = 'plotly_white')

fig.update_yaxes(title_text="Age Group")

fig.update_xaxes(title_text='Number of Hospitalized')



trace1 = go.Pie(

                labels=grupo_uci_reg['Region'],

                values=grupo_uci_reg[ultima_fecha_cl],

                hoverinfo='label+percent', 

                textfont_size=12,

                marker=dict(colors=colors, 

                            line=dict(color='#000000', width=2)))

layout = go.Layout(width=800, height=650,title_text = '<b>Porcentaje de personas Hospitalizadas por Región </b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig2 = go.Figure(data = [trace1], layout = layout)



fig.show()

fig2.show()
fecha_uci_ge = fecha_grupo_edad

print("ACTUALIZADO FECHA: "+fecha_grupo_edad)



jovenes = data_casos_grupo_edad_mf.iloc[0:7,1].sum()

grupo_jovenes_uci = pd.DataFrame({'Numero Casos': [jovenes], 'UCI' : grupo_uci[fecha_grupo_edad].iloc[0]})

grupo_jovenes_uci
trace = go.Scatter(

                x=grupo_uci.iloc[:,1:].columns,

                y=grupo_uci.iloc[0,1:],

                name="Pacientes <39 en UCI",

                mode='lines+markers',

                line_color='red')



layout = go.Layout(template="ggplot2", width=800, height=600,title_text = '<b>Número de Pacientes <=39 en UCI '+ grupo_uci.columns[-1]+'</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig2 = go.Figure(data = [trace], layout = layout)







colors = ['green', 'red']

trace1 = go.Pie(

                labels=grupo_jovenes_uci.columns,

                values=grupo_jovenes_uci.iloc[0,:],

                hoverinfo='label+percent', 

                textfont_size=12,

                marker=dict(colors=colors, 

                            line=dict(color='#000000', width=2)))

layout = go.Layout(width=700, height=500,title_text = '<b>Porcentaje de personas <=39 en UCI: '+fecha_uci_ge+'</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig = go.Figure(data = [trace1], layout = layout)

fig.show()

fig2.show()
mayores = data_casos_grupo_edad_mf.iloc[13:17,1].sum()

grupo_mayores_uci = pd.DataFrame({'Numero Casos': [mayores], 'UCI' : grupo_uci['2020-05-08'].iloc[3:5].sum()})

grupo_mayores_uci
trace = go.Scatter(

                x=grupo_uci.iloc[:,1:].columns,

                y=grupo_uci.iloc[3:5,1:].sum(),

                name="Pacientes >=70 en UCI",

                mode='lines+markers',

                line_color='red')



layout = go.Layout(template="ggplot2", width=800, height=600,title_text = '<b>Número de Pacientes >=60 en UCI '+ grupo_uci.columns[-1]+'</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig2 = go.Figure(data = [trace], layout = layout)





colors = ['green', 'red']

trace1 = go.Pie(

                labels=grupo_mayores_uci.columns,

                values=grupo_mayores_uci.iloc[0,:],

                hoverinfo='label+percent', 

                textfont_size=12,

                marker=dict(colors=colors, 

                            line=dict(color='#000000', width=2)))

layout = go.Layout(width=700, height=500,title_text = '<b>Porcentaje de personas >=60 en UCI: '+fecha_uci_ge+'</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig = go.Figure(data = [trace1], layout = layout)

fig.show()

fig2.show()
sintomas.head()

fecha_sint = sintomas.columns[-1]

sintomas_ultima_fecha = sintomas[['Sintomas',fecha_sint]]



fig2 = px.bar(sintomas_ultima_fecha.sort_values(fecha_sint), 

             x=fecha_sint, y="Sintomas", 

             title='Sintomas de los casos Confirmados con Fecha: '+fecha_sint,

              text=fecha_sint, 

             orientation='h', 

             width=800, height=700)

fig2.update_traces(marker_color='#008000', opacity=0.8, textposition='inside')



fig2.update_layout(template = 'plotly_white')
num_vent = pd.read_csv('https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto20/NumeroVentiladores.csv')

dates_vent = num_vent.loc[:, '2020-04-14': ultima_fecha_cl]

dates_vent = dates_vent.keys()



ventiladores_oc =[]

ventiladores_dis = []

ventiladores_total = []

for i in dates_vent:

    oc = num_vent[num_vent['Ventiladores']=='ocupados'][i].sum()

    dis = num_vent[num_vent['Ventiladores']=='disponibles'][i].sum()

    total = num_vent[num_vent['Ventiladores']=='total'][i].sum()

    

    ventiladores_oc.append(oc)

    ventiladores_dis.append(dis)

    ventiladores_total.append(total)

    

num_vent
trace = go.Scatter(

                x=dates_vent,

                y=ventiladores_dis,

                name="Ventiladores Disponibles",

                mode='lines+markers',

                line_color='green')

trace2 = go.Scatter(

                x=dates_vent,

                y=ventiladores_oc,

                name="Ventiladores Ocupados",

                mode='lines+markers',

                line_color='red')



layout = go.Layout(template="ggplot2", width=800, height=600,title_text = '<b>Numero de Ventiladores Fecha: '+ ultima_fecha_cl+'</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig2 = go.Figure(data = [trace,trace2], layout = layout)

fig2.show()
ventiladiores = num_vent.drop([0],axis=0)



fig2 = make_subplots(rows=1, cols=2)

colors = ['green','red']

trace1 = go.Pie(

                labels=ventiladiores['Ventiladores'],

                values=ventiladiores[ultima_fecha_cl],

                hoverinfo='label+percent', 

                textfont_size=12,

                marker=dict(colors=colors, 

                            line=dict(color='#000000', width=2)))

layout = go.Layout(width=700, height=500,title_text = '<b>Porcentaje de ventiladores Fecha: '+ultima_fecha_cl+'</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig2 = go.Figure(data = [trace1], layout = layout)



fig2.show()
datos = data_chile[['Region',ultima_fecha_cl]].drop([16],axis=0)



fig = px.scatter(datos, y=datos.loc[:,ultima_fecha_cl],

                    x= datos.loc[:,"Region"],

                    color= "Region", hover_name="Region",

                    color_continuous_scale=px.colors.sequential.Plasma,

                    title='COVID-19: Numero Total de casos por Region',

                    size = np.power(datos[ultima_fecha_cl]+1,0.3)-0.5,

                    size_max = 30,

                    height =600,

                    )

fig.update_coloraxes(colorscale="hot")

fig.update(layout_coloraxis_showscale=False)

fig.update_yaxes(title_text="Numero casos")

fig.update_xaxes(title_text="Regiones")

fig.show()
data_por_comuna = pd.read_csv('https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto19/CasosActivosPorComuna.csv')

data_por_comuna.head()
data_casos_por_comuna = pd.read_csv('https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto2/2020-05-25-CasosConfirmados.csv')

fecha='2020-05-25'

data_casos_por_comuna_maule = data_casos_por_comuna[data_casos_por_comuna['Region']=='Maule']

data_casos_por_comuna.head()
data_casos_por_comuna_maule = data_casos_por_comuna_maule.sort_values('Casos Confirmados')



total_maule= data_casos_por_comuna_maule['Casos Confirmados'].sum()

total_maule = str(total_maule)



fig2 = px.bar(x=data_casos_por_comuna_maule['Comuna'], y=data_casos_por_comuna_maule['Casos Confirmados'],

            title='Numero de casos Totales Confirmados en el Maule por Comuna Total: '+total_maule+' Fecha: '+fecha,

           text=data_casos_por_comuna_maule['Casos Confirmados']

            

)

fig2.update_xaxes(title_text="Comunas")

fig2.update_yaxes(title_text="Numero de Casos")
fig2 = make_subplots(rows=1, cols=2)



trace1 = go.Pie(

                labels=data_casos_por_comuna_maule['Comuna'],

                values=data_casos_por_comuna_maule['Casos Confirmados'],

                hoverinfo='label+percent', 

                textfont_size=12,

                marker=dict(line=dict(color='#000000', width=2)))

layout = go.Layout(width=800, height=1000,title_text = '<b>Porcentaje de casos Totales Confirmados en el maule: '+fecha+'</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig2 = go.Figure(data = [trace1], layout = layout)



fig2.show()
data_casos_por_comuna_talca = data_casos_por_comuna[data_casos_por_comuna['Comuna']=='Talca']

data_talca = pd.DataFrame({'Tipo':['Enfermos','Sanos'],'Numero': [data_casos_por_comuna_talca.iloc[0,-1],data_casos_por_comuna_talca.iloc[0,4]]})

fig2 = make_subplots(rows=1, cols=2)



trace1 = go.Pie(

                labels=data_talca['Tipo'],

                values=data_talca['Numero'],

                hoverinfo='label+percent', 

                textfont_size=12,

                marker=dict(line=dict(color='#000000', width=2)))

layout = go.Layout(width=700, height=500,title_text = '<b>Porcentajes de enfermos en Talca: '+fecha+'</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig2 = go.Figure(data = [trace1], layout = layout)



fig2.show()
talca = data_por_comuna[data_por_comuna['Comuna']== 'Talca']

data_talca = pd.DataFrame({'Cases':['Sick','Healthy'],'Number Cases': [talca.iloc[0,-1],talca.iloc[0,4]]})



fecha_talca_act =talca.columns[-1]



trace = go.Scatter(

                x=talca.columns[5:],

                y=talca.iloc[0,5:],

                name="Pacientes Criticos",

                mode='lines+markers',

                line_color='red')



layout = go.Layout(template="ggplot2", width=800, height=500,title_text = '<b>Casos activos por fecha de inicio de síntomas en Talca '+ fecha_talca_act+'</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig = go.Figure(data = [trace], layout = layout)

fig.show()
data_casos_por_comuna_M = data_casos_por_comuna[data_casos_por_comuna['Region']=='Metropolitana']

total_santiago = str(data_casos_por_comuna_M['Casos Confirmados'].sum())



data_casos_por_comuna_M = data_casos_por_comuna_M.sort_values('Casos Confirmados',ascending=False)

fig2 = px.bar(x=data_casos_por_comuna_M['Comuna'], y=data_casos_por_comuna_M['Casos Confirmados'],

            title='Numero de casos Totales Confirmados en Santiago por Comuna Total: '+total_santiago+' Fecha: '+fecha,

           text=data_casos_por_comuna_M['Casos Confirmados']

            

)

fig2.update_xaxes(title_text="Comunas")

fig2.update_yaxes(title_text="Numero de Casos")
fig2 = make_subplots(rows=1, cols=2)



trace1 = go.Pie(

                labels=data_casos_por_comuna_M['Comuna'],

                values=data_casos_por_comuna_M['Casos Confirmados'],

                hoverinfo='label+percent', 

                textfont_size=12,

                marker=dict(line=dict(color='#000000', width=2)))

layout = go.Layout(width=800, height=1000,title_text = '<b>Numero de casos Totales Confirmados en Santiago: '+fecha+'</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig2 = go.Figure(data = [trace1], layout = layout)



fig2.show()
data_casos_por_comuna_r = data_casos_por_comuna[data_casos_por_comuna['Region']=='Los Ríos']



data_casos_por_comuna_r = data_casos_por_comuna_r.sort_values('Casos Confirmados')



total_rios= data_casos_por_comuna_r['Casos Confirmados'].sum()

total_rios = str(total_rios)



fig2 = px.bar(x=data_casos_por_comuna_r['Comuna'], y=data_casos_por_comuna_r['Casos Confirmados'],

            title='Numero de casos Totales Confirmados en el Los Ríos por Comuna'+' Fecha: '+fecha,

           text=data_casos_por_comuna_r['Casos Confirmados']

            

)

fig2.update_xaxes(title_text="Comunas")

fig2.update_yaxes(title_text="Numero de Casos")
fallecimientos_en_chile = pd.read_csv('https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto32/Defunciones_T.csv')

fallecimientos_en_chile.head()
data_fallecidos = fallecimientos_en_chile.drop(0,axis=0)

data_fallecidos = data_fallecidos.drop(1,axis=0)

data_fallecidos = data_fallecidos.drop(2,axis=0)

data_fallecidos
data_fallecidos
#https://www.it-swarm.dev/es/python/pandas-suma-las-filas-de-dataframe-para-columnas-dadas/1047832035/

data_fallecidos = fallecimientos_en_chile.drop(0,axis=0)

data_fallecidos = data_fallecidos.drop(1,axis=0)

data_fallecidos = data_fallecidos.drop(2,axis=0)



colum= data_fallecidos.iloc[:,1:].columns.tolist()

data_fallecidos[colum] = data_fallecidos[colum].astype(np.int64)

data_fallecidos['Total'] = data_fallecidos.sum(axis=1)

colum_el= data_fallecidos.iloc[:,1:338].columns.tolist()

data_fallecidos_sum = data_fallecidos.drop(colum_el,axis=1)

data_fallecidos_sum['Region'] = pd.to_datetime(data_fallecidos_sum.Region)

data_fallecidos_sum
años = data_fallecidos_sum['Region'].dt.strftime('%Y').unique()

total_fallecimientos_mes = pd.DataFrame({'Años': años,'January':0,'February':0,'March':0,'April':0,'May':0,'June':0,'July':0,'August':0,'August':0,'September':0,'October':0,'November':0,'December':0})

total_fallecimientos_mes
#https://stackoverrun.com/es/q/5246269

#filtro_registros_año = data_fallecidos_sum[data_fallecidos_sum['Region'].dt.strftime('%Y') == '2010']

#meses = filtro_registros_año.groupby(año['Region'].dt.strftime('%B'))['Total'].sum()

#meses

registros_meses = ['January','February','March','April','May','June','July','August','September','October','November','December']

for año_c in años:

    filtro_registros_año = data_fallecidos_sum[data_fallecidos_sum['Region'].dt.strftime('%Y') == año_c]

    #filtro_registros_mes = filtro_registros_año.groupby(año['Region'].dt.strftime('%B'))['Total'].sum()

    for i_meses in registros_meses:

        num_f = filtro_registros_año[filtro_registros_año['Region'].dt.strftime('%B') == i_meses].sum().values

        if(num_f[0] == 0):

            total_fallecimientos_mes.loc[total_fallecimientos_mes.Años == año_c, i_meses] = 0

        else:

            total_fallecimientos_mes.loc[total_fallecimientos_mes.Años == año_c, i_meses] = num_f



total_fallecimientos_mes    
año_2010 =[]

año_2011 =[]

año_2012 =[]

año_2013 =[]

año_2014 =[]

año_2015 =[]

año_2016 =[]

año_2017 =[]

año_2018 =[]

año_2019 =[]

año_2020 =[]



for i in registros_meses:

    reg_año_2010 = total_fallecimientos_mes[total_fallecimientos_mes['Años']==2010][i].sum()

    reg_año_2011 = total_fallecimientos_mes[total_fallecimientos_mes['Años']==2011][i].sum()

    reg_año_2012 = total_fallecimientos_mes[total_fallecimientos_mes['Años']==2012][i].sum()

    reg_año_2013 = total_fallecimientos_mes[total_fallecimientos_mes['Años']==2013][i].sum()

    reg_año_2014 = total_fallecimientos_mes[total_fallecimientos_mes['Años']==2014][i].sum()

    reg_año_2015 = total_fallecimientos_mes[total_fallecimientos_mes['Años']==2015][i].sum()

    reg_año_2016 = total_fallecimientos_mes[total_fallecimientos_mes['Años']==2016][i].sum()

    reg_año_2017 = total_fallecimientos_mes[total_fallecimientos_mes['Años']==2017][i].sum()

    reg_año_2018 = total_fallecimientos_mes[total_fallecimientos_mes['Años']==2018][i].sum()

    reg_año_2019 = total_fallecimientos_mes[total_fallecimientos_mes['Años']==2019][i].sum()

    reg_año_2020 = total_fallecimientos_mes[total_fallecimientos_mes['Años']==2020][i].sum()



    año_2010.append(reg_año_2010)

    año_2011.append(reg_año_2011)

    año_2012.append(reg_año_2012)

    año_2013.append(reg_año_2013)

    año_2014.append(reg_año_2014)

    año_2015.append(reg_año_2015)

    año_2016.append(reg_año_2016)

    año_2017.append(reg_año_2017)

    año_2018.append(reg_año_2018)

    año_2019.append(reg_año_2019)

    año_2020.append(reg_año_2020)
trace = go.Scatter(

                x=registros_meses,

                y=año_2010,

                name="2010",

                mode='lines+markers',

                line_color='#800080')

trace2 = go.Scatter(

                x=registros_meses,

                y=año_2011,

                name="2011",

                mode='lines+markers',

                line_color='green')

trace3 = go.Scatter(

                x=registros_meses,

                y=año_2012,

                name="2012",

                mode='lines+markers',

                line_color='#000080')

trace4 = go.Scatter(

                x=registros_meses,

                y=año_2013,

                name="2013",

                mode='lines+markers',

                line_color='#00FFFF')

trace5 = go.Scatter(

                x=registros_meses,

                y=año_2014,

                name="2014",

                mode='lines+markers',

                line_color='#FFFF00')

trace6 = go.Scatter(

                x=registros_meses,

                y=año_2015,

                name="2015",

                mode='lines+markers',

                line_color='#000000')

trace7 = go.Scatter(

                x=registros_meses,

                y=año_2016,

                name="2016",

                mode='lines+markers',

                line_color='#808080')

trace8 = go.Scatter(

                x=registros_meses,

                y=año_2017,

                name="2017",

                mode='lines+markers',

                line_color='#008080')

trace9 = go.Scatter(

                x=registros_meses,

                y=año_2018,

                name="2018",

                mode='lines+markers',

                line_color='#00FF00')

trace10 = go.Scatter(

                x=registros_meses,

                y=año_2019,

                name="2019",

                mode='lines+markers',

                line_color='#800000')

trace11 = go.Scatter(

                x=registros_meses,

                y=año_2020,

                name="2020",

                mode='lines+markers',

                line_color='red')





layout = go.Layout(template="ggplot2", width=1000, height=600,title_text = '<b>Numero de Fallecidos entre 2010-2020 hasta el momento en Chile(2020 no esta completo)</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig2 = go.Figure(data = [trace,trace2,trace3,trace4,trace5,trace6,trace7,trace8,trace9,trace10,trace11], layout = layout)

fig2.show()
suma_4meses = total_fallecimientos_mes

col_list= ['January','February','March','April']

suma_4meses['Total 4 Meses'] = suma_4meses[col_list].sum(axis=1)

suma_4meses
fig2 = px.bar(x=suma_4meses['Total 4 Meses'], y=total_fallecimientos_mes['Años'], 

             title='Total de Fallecidos en los meses de Enero a Abril',

              text=suma_4meses['Total 4 Meses'], 

             orientation='h', 

             width=800, height=500)

fig2.update_traces(marker_color='#008000', opacity=0.8, textposition='inside')



fig2.update_layout(template = 'plotly_white')

fig2.update_xaxes(title_text="Número de Fallecidos")

fig2.update_yaxes(title_text="Años")
def hyperopt_kr(X,y,max_iter):



    def objective(hyperparameters):

        

        global ITERATION

    

        ITERATION += 1

        start = timer()

        clf = KernelRidge(**hyperparameters)

        

        cv_results = cross_val_score(clf,X, y,cv=10,scoring='neg_mean_squared_error',n_jobs=-1).mean() 



        run_time = timer() - start

        

        best_score = cv_results

        loss = 1 - cv_results



        return {'loss': loss, 'hyperparameters': hyperparameters, 'iteration': ITERATION,

                'train_time': run_time, 'status': STATUS_OK}

    

    kernel_list = [ 'polynomial']



    

    space = {

            'alpha': hp.quniform('alpha', 0.000001, 20, 0.000001),

            'degree': hp.quniform('degree', 0.000001, 10, 0.00001),

            'coef0': hp.quniform('coef0', 0.000001, 10, 0.00001),

            'kernel': hp.choice('kernel', kernel_list),



    }









    tpe_algorithm = tpe.suggest

    trials = Trials()

    

    # Ejecutar optimización

    best = fmin(fn = objective, space = space, algo = tpe.suggest, trials = trials,

                max_evals = max_iter)

    

    best['kernel'] = kernel_list[best['kernel']]

    return best
X_cl = days_chile

y_cl = casos_chile



y_cl = np.array(casos_chile).reshape(-1, 1)



#from scipy.stats import boxcox



#y_cl, lam = boxcox(casos_chile)



X_train, X_test, y_train, y_test = train_test_split(X_cl

                                                    , y_cl

                                                    , test_size= 0.05

                                                    , shuffle=False

                                                    , random_state = 42)

#poly = PolynomialFeatures(degree=4)

#poly_X_train = poly.fit_transform(X_train)

#poly_X_test = poly.fit_transform(X_test)





#FUNCION OBTENIDA DE: https://www.kaggle.com/gatunnopvp/covid-19-in-brazil-prediction-updated-05-23-20

rmse = 10000

degree = 0

for i in range(71):

    # Transform our cases data for polynomial regression

    poly = PolynomialFeatures(degree=i)

    poly_X_train = poly.fit_transform(X_train)

    poly_X_test = poly.fit_transform(X_test)



    

    #b = hyperopt_svm(poly_X_train,y_train,500)

    # polynomial regression cases

    model_kr_cl = KernelRidge()

    model_kr_cl.fit(poly_X_train, y_train)

    test_linear_pred = model_kr_cl.predict(poly_X_test)



    # evaluating with RMSE

    rm = sqrt(mean_squared_error(y_test, test_linear_pred))

    if(rm<rmse):

        rmse = rm

        degree = i

    if(i==70):

        print('the best mae is:',round(rmse,2))

        print('the best degree for cases is:',degree)

        
global  ITERATION

ITERATION = 0

poly = PolynomialFeatures(degree=degree)

poly_X_train = poly.fit_transform(X_train)

poly_X_test = poly.fit_transform(X_test)

param = hyperopt_kr(poly_X_train,y_train,500)



#poly_X_train=X_train

#poly_X_test = X_test



model_kr_cl_1 = KernelRidge(**param)

model_kr_cl_1.fit(poly_X_train, y_train)

pred_rg_cl_1=model_kr_cl_1.predict(poly_X_test)



print('RMSE:', sqrt(mean_squared_error(y_test, pred_rg_cl_1)))

score = rmsle_cv(model_kr_cl_1,poly_X_test,y_test)

print("Kernel Ridge score CV: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))

dataframe=pd.DataFrame(X_test, columns=['Días'])



print("Root Mean Square Value:",np.sqrt(mean_squared_error(y_test,pred_rg_cl_1)))

print('MAE:', mean_absolute_error(pred_rg_cl_1, y_test))

print('MSE:',mean_squared_error(pred_rg_cl_1, y_test))



plt.figure(figsize=(11,6))

plt.plot(y_test,label="Actual Confirmed Cases")

plt.plot(dataframe.index,pred_rg_cl_1, linestyle='--',label="Predicted Confirmed Cases using Kernel Ridge",color='black')

plt.xlabel('Días')

plt.ylabel('Casos Confirmados')

plt.xticks(rotation=90)

plt.legend()
days_in_future_cl = 20

future_forcast_cl = np.array([i for i in range(len(dates_chile)+days_in_future_cl)]).reshape(-1, 1)

adjusted_dates_cl = future_forcast_cl[:-days_in_future_cl]

start_cl = '03/03/2020'

start_date_cl = datetime.datetime.strptime(start_cl, '%m/%d/%Y')

future_forcast_dates_cl = []

for i in range(len(future_forcast_cl)):

    future_forcast_dates_cl.append((start_date_cl + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))



future_forcast_cl = poly.fit_transform(future_forcast_cl)



kr_pred_cl = model_kr_cl_1.predict(future_forcast_cl)



Predict_df_cl_1= pd.DataFrame()

Predict_df_cl_1["Fecha"] = list(future_forcast_dates_cl[-days_in_future_cl-1:])

Predict_df_cl_1["N° Casos"] =np.round(kr_pred_cl[-days_in_future_cl-1:])

Predict_df_cl_1
trace1 = go.Scatter(

                x= np.array(future_forcast_dates_cl),

                y=casos_chile,

                name="Casos Confirmados",

                mode='lines+markers',

                line_color='green')



trace2 = go.Scatter(

                x=Predict_df_cl_1["Fecha"],

                y=Predict_df_cl_1["N° Casos"],

                name="Predicciones",

                mode='lines+markers',

                line_color='blue')



layout = go.Layout(template="ggplot2", width=900, height=600, title_text ='<b>Prediccion del Número de casos para los siguientes '+str(days_in_future_cl)+' días en Chile</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig = go.Figure(data = [trace1,trace2], layout = layout)

fig.show()
X_cl = days_chile

y_cl = casos_chile



#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10,shuffle=False) 

#Splitting data into train and test to evaluate our model



parameters = {

        'alpha':[0.000001,0.0001,0.001,0.1,0.0002,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.2,1.3,1.4,1.5,1.6,2,2.1,2.2,2.3,2.4,2.5,3,4,5],

        'kernel': ['polynomial'],

        'degree': [0.0001,0.001,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.5,1.6,2,2.1,2.2,2.3,2.4,3,4,4.2,4.3,4.5,5,6],

        'coef0': [0.0001,0.001, 0.1,0.0002,0.2,0.25,1,1.2,1.5,2,2.1,2.2,2.5,3,3.2,3.5,4,4.1,4.2,4.3,4.5,5]

    }

clf =KernelRidge()

clf1 = GridSearchCV(clf, parameters,scoring='neg_mean_squared_error', n_jobs=-1, cv=10)

#clf1.fit(X_cl, y_cl)

clf1.fit(X_cl, y_cl)



best_params = clf1.best_params_

beast_score =clf1.best_score_



print("Mejor puntuacion:",beast_score)

print("Mejores Parametros;",best_params)



model_kr_cl = KernelRidge(**best_params)

#model_kr_cl.fit(X_cl, y_cl)

model_kr_cl.fit(X_cl, y_cl)





#score = rmsle_cv(model_kr_cl,X_cl,y_cl)

score = rmsle_cv(model_kr_cl,X_cl,y_cl)



print("Kernel Ridge score: {:.4f} ({:.4f})\n".format(score.mean(), score.std()))



dataframe=pd.DataFrame(X_cl, columns=['Days'])

pred_rg_cl=model_kr_cl.predict(np.array(X_cl).reshape(-1,1))

print("Root Mean Square Value:",np.sqrt(mean_squared_error(y_cl,pred_rg_cl)))

print('MAE:', mean_absolute_error(pred_rg_cl, y_cl))

print('MSE:',mean_squared_error(pred_rg_cl, y_cl))



plt.figure(figsize=(11,6))

plt.plot(y_cl,label="Actual Confirmed Cases")

plt.plot(dataframe.index,pred_rg_cl, linestyle='--',label="Predicted Confirmed Cases using Kernel Ridge",color='black')

plt.xlabel('Days')

plt.ylabel('Confirmed Cases')

plt.xticks(rotation=90)

plt.legend()
days_in_future_cl = 20

future_forcast_cl = np.array([i for i in range(len(dates_chile)+days_in_future_cl)]).reshape(-1, 1)

adjusted_dates_cl = future_forcast_cl[:-days_in_future_cl]

start_cl = '03/03/2020'

start_date_cl = datetime.datetime.strptime(start_cl, '%m/%d/%Y')

future_forcast_dates_cl = []

for i in range(len(future_forcast_cl)):

    future_forcast_dates_cl.append((start_date_cl + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))

    

kr_pred_cl = model_kr_cl.predict(future_forcast_cl)



Predict_df_cl= pd.DataFrame()

Predict_df_cl["Date"] = list(future_forcast_dates_cl[-days_in_future_cl-1:])

Predict_df_cl["N° Cases"] =np.round(kr_pred_cl[-days_in_future_cl-1:])

Predict_df_cl
trace1 = go.Scatter(

                x= np.array(future_forcast_dates_cl),

                y=casos_chile,

                name="Casos Confirmados",

                mode='lines+markers',

                line_color='green')



trace2 = go.Scatter(

                x=Predict_df_cl["Date"],

                y=Predict_df_cl["N° Cases"],

                name="Predicciones",

                mode='lines+markers',

                line_color='blue')



layout = go.Layout(template="ggplot2", width=900, height=600, title_text ='<b>Prediccion del Número de casos para los siguientes '+str(days_in_future_cl)+' días en Chile</b>',

                  font=dict(family="Arial, Balto, Courier New, Droid Sans",color='black'))

fig = go.Figure(data = [trace1,trace2], layout = layout)

fig.show()
days_chile2 = np.array([i for i in range(len(dates_chile))])



data_ch = pd.DataFrame({'Días': list(days_chile2), 'Casos':casos_chile})

data_ch.head()
from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing



model_scores= []



x_train=data_ch.iloc[:int(data_ch.shape[0]*0.95)]

x_test=data_ch.iloc[int(data_ch.shape[0]*0.95):]

y_pred=x_test.copy()



es=ExponentialSmoothing(np.asarray(x_train['Casos']),seasonal_periods=3,trend='add', seasonal='mul').fit()

y_pred["prediccion"]=es.forecast(len(x_test))



model_scores.append(np.sqrt(mean_squared_error(y_pred["Casos"],y_pred["prediccion"])))

print("Root Mean Square Error: ",np.sqrt(mean_squared_error(y_pred["Casos"],y_pred["prediccion"])))
fig=go.Figure()

fig.add_trace(go.Scatter(x=x_train.index, y=x_train["Casos"],

                    mode='lines+markers',name="Casos Reales"))

fig.add_trace(go.Scatter(x=x_test.index, y=x_test["Casos"],

                    mode='lines+markers',name="Casos de Validacion",))

fig.add_trace(go.Scatter(x=x_test.index, y=y_pred["prediccion"],

                    mode='lines+markers',name="Casos Predichos",))

fig.update_layout(title="Prediccion de Casos",

                 xaxis_title="Date",yaxis_title="Casos Confirmados",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
es=ExponentialSmoothing(np.asarray(data_ch['Casos']),seasonal_periods=10,trend='add', seasonal='mul').fit()



days_in_future_cl = 20

future_forcast_cl = np.array([i for i in range(len(dates_chile)+days_in_future_cl)]).reshape(-1, 1)

adjusted_dates_cl = future_forcast_cl[:-days_in_future_cl]

start_cl = '03/03/2020'

start_date_cl = datetime.datetime.strptime(start_cl, '%m/%d/%Y')

future_forcast_dates_cl = []

for i in range(len(future_forcast_cl)):

    future_forcast_dates_cl.append((start_date_cl + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))

        

Predict_df_cl_1= pd.DataFrame()

Predict_df_cl_1["Fecha"] = list(future_forcast_dates_cl[-days_in_future_cl:])

Predict_df_cl_1["N° Casos"] =np.round(list(es.forecast(20)))

Predict_df_cl_1
fig=go.Figure()

fig.add_trace(go.Scatter(x=np.array(future_forcast_dates_cl), y=data_ch["Casos"],

                        mode='lines+markers',name="Casos Reales"))

fig.add_trace(go.Scatter(x=Predict_df_cl_1['Fecha'], y=Predict_df_cl_1["N° Casos"],

                        mode='lines+markers',name="Predicción de Casos",))



fig.update_layout(title="Proyección de casos en 20 días",

                    xaxis_title="Fecha",yaxis_title="Número de Casos",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
days_fallecidos_chile = np.array([i for i in range(len(fechas_chile ))])



data_ch_fallecidos = pd.DataFrame({'Días': list(days_fallecidos_chile), 'Fallecidos': [int(x) for x in fallecidos_por_dia]})

casos_f = data_ch_fallecidos['Fallecidos']+1

data_ch_fallecidos = pd.DataFrame({'Días': list(days_fallecidos_chile), 'Fallecidos':casos_f})

data_ch_fallecidos.head()
from statsmodels.tsa.api import Holt,SimpleExpSmoothing,ExponentialSmoothing



model_scores= []



x_train=data_ch_fallecidos.iloc[:int(data_ch_fallecidos.shape[0]*0.95)]

x_test=data_ch_fallecidos.iloc[int(data_ch_fallecidos.shape[0]*0.95):]

y_pred=x_test.copy()



es=ExponentialSmoothing(np.asarray(x_train['Fallecidos']),seasonal_periods=3,trend='add', seasonal='mul').fit()

y_pred["prediccion"]=es.forecast(len(x_test))



model_scores.append(np.sqrt(mean_squared_error(y_pred["Fallecidos"],y_pred["prediccion"])))

print("Root Mean Square Error: ",np.sqrt(mean_squared_error(y_pred["Fallecidos"],y_pred["prediccion"])))
fig=go.Figure()

fig.add_trace(go.Scatter(x=x_train.index, y=x_train["Fallecidos"],

                    mode='lines+markers',name="Casos Reales"))

fig.add_trace(go.Scatter(x=x_test.index, y=x_test["Fallecidos"],

                    mode='lines+markers',name="Fallecidos de Validacion",))

fig.add_trace(go.Scatter(x=x_test.index, y=y_pred["prediccion"],

                    mode='lines+markers',name="Fallecidos Predichos",))

fig.update_layout(title="Prediccion de Fallecidos",

                 xaxis_title="Date",yaxis_title="Fallecidos Confirmados",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()
es=ExponentialSmoothing(np.asarray(data_ch_fallecidos['Fallecidos']),seasonal_periods=5,trend='add', seasonal='mul').fit()



days_in_future_cl = 20

future_forcast_cl = np.array([i for i in range(len(dates_chile)+days_in_future_cl)]).reshape(-1, 1)

adjusted_dates_cl = future_forcast_cl[:-days_in_future_cl]

start_cl = '03/03/2020'

start_date_cl = datetime.datetime.strptime(start_cl, '%m/%d/%Y')

future_forcast_dates_cl = []

for i in range(len(future_forcast_cl)):

    future_forcast_dates_cl.append((start_date_cl + datetime.timedelta(days=i)).strftime('%m/%d/%Y'))

        

Predict_df_cl_1= pd.DataFrame()

Predict_df_cl_1["Fecha"] = list(future_forcast_dates_cl[-days_in_future_cl:])

Predict_df_cl_1["N° Fallecidos"] =np.round(list(es.forecast(20)))

Predict_df_cl_1
fig=go.Figure()

fig.add_trace(go.Scatter(x=np.array(future_forcast_dates_cl), y=data_ch_fallecidos["Fallecidos"],

                        mode='lines+markers',name="Fallecidos Reales"))

fig.add_trace(go.Scatter(x=Predict_df_cl_1['Fecha'], y=Predict_df_cl_1["N° Fallecidos"],

                        mode='lines+markers',name="Predicción de Fallecidos",))



fig.update_layout(title="Proyección de Fallecidos en 20 días",

                    xaxis_title="Fecha",yaxis_title="Número de Fallecidos",legend=dict(x=0,y=1,traceorder="normal"))

fig.show()