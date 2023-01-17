import pandas as pd

import numpy as np

import unicodedata



import matplotlib.pyplot as plt

import plotly.express as px

import seaborn as sns

from sklearn.neural_network import MLPRegressor

from datetime import date, timedelta

import plotly.graph_objects as go

 
data = pd.read_csv('../input/covid19perudataconubigeos/covid-19-peru-data-con-ubigeos.csv')

data.head()
data.isna().sum()
data.columns.to_list()
data = data[['region','date','confirmed','deaths','recovered','negative_cases','pob_2017']]

data.head()
data.date = pd.to_datetime(data["date"]) 

data.region = data.region.fillna('').astype(str)

data.deaths = data.deaths.fillna(0).astype(int)

data.deaths = data.deaths.fillna(0).astype(int)

data.recovered = data.recovered.fillna(0).astype(int)

data.negative_cases = data.negative_cases.fillna(0).astype(int)

data.pob_2017 = data.pob_2017.fillna(0).astype(int)
data = data[data.region!=''].reset_index()

data = data.rename(columns={'index':'id'})

data.head()
data.isna().sum()
data_acum = data



lista_regiones = data_acum['region'].value_counts().reset_index()['index'].tolist()

lista_regiones.sort()

lista_regiones
def generandoCasosConfirmados(data_acumulada, nueva_data):

    nueva_data = data_acumulada.sort_values(['region','date'],ascending=True)

    nueva_data = nueva_data.rename(columns={'confirmed':'confirmed_acum'})

    new_list=[]

    

    for ciudad in lista_regiones:

        

        lista_acum = nueva_data[nueva_data.region==ciudad].confirmed_acum.to_list()

        tam_lista = len(lista_acum)

        

        if tam_lista>0: 

            

            new_list.append(lista_acum[0])

            

            if(tam_lista>=2):

                for i in range(tam_lista-1):

                    new_list.append(lista_acum[i+1]-lista_acum[i])

    nueva_data['confirmed'] = new_list

    return nueva_data
data = generandoCasosConfirmados(data_acum, data)

data.head()
data_region = data.groupby("region")[["confirmed"]].sum().reset_index()

data_region = data_region.rename(columns={

    'region':'Region',

    'confirmed':'Nro de casos confirmados'

}) 
fig = px.bar(

    data_region[['Region','Nro de casos confirmados']],

    y = 'Nro de casos confirmados',

    x = 'Region',

    color = 'Region',

    log_y = True,

    template='ggplot2', 

    title='Cantidad de casos confirmados por region'

)

fig.show()
data_fecha_infectados = data.groupby("date")[['confirmed']].sum().reset_index()

data_fecha_infectados["confirmed_acum"] = data_fecha_infectados.confirmed.cumsum()
plt.figure(figsize=(10,5))

plt.title('Numero de nuevos infectados cada dia')

plt.plot(data_fecha_infectados.date,data_fecha_infectados.confirmed)
plt.figure(figsize=(10,5))

plt.title('Numero de infectados cada dia en Peru')

plt.plot(data_fecha_infectados.date,data_fecha_infectados.confirmed_acum)
data_muertes = pd.read_csv('../input/covid19perudataconubigeos/covid-19-peru-fallecimientos.csv')

data_muertes.head()
data_muertes.isna().sum()
data_muertes = data_muertes.rename(columns={'región':'region'})



data_muertes.fecha_anuncio = pd.to_datetime(data_muertes.fecha_anuncio)

data_muertes.fecha_fallecimiento = pd.to_datetime(data_muertes.fecha_fallecimiento)

data_muertes.fecha_ingreso = pd.to_datetime(data_muertes.fecha_ingreso)

data_muertes.fecha_retorno = pd.to_datetime(data_muertes.fecha_retorno)

data_muertes.fecha_contacto = pd.to_datetime(data_muertes.fecha_contacto)



data_muertes.neumonia = data_muertes.neumonia.fillna(0).astype(int)

data_muertes.insuf_resp = data_muertes.insuf_resp.fillna(0).astype(int)



data_muertes.contacto = data_muertes.contacto.fillna('desconocido').astype(str)

data_muertes.contacto_origen = data_muertes.contacto_origen.fillna('desconocido').astype(str)

data_muertes.otros_sintomas = data_muertes.otros_sintomas.fillna('ninguno').astype(str)

data_muertes.factores = data_muertes.factores.fillna('ninguno').astype(str)

data_muertes.viaje = data_muertes.viaje.fillna('no_aviajo').astype(str)

data_muertes.misc = data_muertes.misc.fillna('').astype(str)



data_muertes.head()
data_muertes_region = pd.DataFrame(data_muertes["region"])

data_muertes_region = data_muertes_region.region.value_counts().reset_index()

data_muertes_region = data_muertes_region.rename(columns={'index':'region','region':'cantidad'})
fig = px.bar(

    data_muertes_region[['region','cantidad']],

    y = 'cantidad',

    x = 'region',

    color = 'region',

    log_y = True,

    template='ggplot2', 

    title='Nro muertes en cada region',

    labels={'cantidad':'Nro de muertes','region':'Region'}

)

fig.show()
data_masculino_femenino = data_muertes[['sexo']]
plt.figure(figsize=(15, 5))

plt.title('Gender')

data_masculino_femenino.sexo.value_counts().plot.bar();
sintomas = pd.DataFrame((';'.join(data_muertes.otros_sintomas[data_muertes.otros_sintomas!='ninguno'].to_list())).split(';'))

sintomas = sintomas.rename(columns={0:'otros_sintomas'})

sintomas = sintomas.otros_sintomas.value_counts().reset_index()

sintomas = sintomas.rename(columns={'index':'otros_sintomas','otros_sintomas':'cantidad'})
fig = px.pie(sintomas,

             values="cantidad",

             names="otros_sintomas",

             title="Otros sintomas en los fallecidos",

             template="seaborn")



fig.update_layout(

    font=dict(

        size=15,

        color="#242323"

    )

    )  

fig.show()
data_muertes_factores = pd.DataFrame((';'.join(data_muertes.factores[data_muertes.factores!='ninguno'].to_list())).split(';'))

data_muertes_factores = data_muertes_factores.rename(columns={0:'factores'})

data_muertes_factores = data_muertes_factores.factores.value_counts().reset_index()

data_muertes_factores = data_muertes_factores.rename(columns={'index':'factores','factores':'cantidad'})
fig = px.pie(data_muertes_factores,

             values="cantidad",

             names="factores",

             title="Factores de los fallecidos",

             template="seaborn")



fig.update_layout(

    font=dict(

        size=15,

        color="#242323"

    )

    )  

fig.show()
data_muertes_edad_sexo = data_muertes[['edad','sexo']]
plt.figure(figsize=(10,6))

sns.set_style("darkgrid")

plt.title("Distribucion de edades de las muertes")

sns.kdeplot(data=data_muertes_edad_sexo.edad, shade=True)
data_muertes_femenino_edad = data_muertes_edad_sexo[data_muertes_edad_sexo.sexo=="femenino"]

data_muertes_masculino_edad = data_muertes_edad_sexo[data_muertes_edad_sexo.sexo=="masculino"]
sns.set_style("darkgrid")

plt.title("Age distribution of the confirmation by gender")

sns.kdeplot(data=data_muertes_masculino_edad.edad, label="Masculino", shade=True).set(xlim=(0))

sns.kdeplot(data=data_muertes_femenino_edad.edad,label="Femenino" ,shade=True).set(xlim=(0))
data_muertes_lugar_fallecimiento = data_muertes.lugar_fallecimiento.value_counts().reset_index()

data_muertes_lugar_fallecimiento = data_muertes_lugar_fallecimiento.rename(

    columns={

        'index':'lugar_fallecimiento',

        'lugar_fallecimiento':'cantidad'

    })
fig = px.pie(data_muertes_lugar_fallecimiento,

             values="cantidad",

             names="lugar_fallecimiento",

             title="Lugares de Fallecimiento",

             template="seaborn")



fig.update_traces(rotation=90, pull=0.05, textinfo="value+percent+label")

fig.show()
data_muertes_fallecidos_viaje = data_muertes.viaje.value_counts().reset_index()

data_muertes_fallecidos_viaje = data_muertes_fallecidos_viaje.rename(

    columns={

        'index':'viaje',

        'viaje':'cantidad'

    })
fig = px.pie(data_muertes_fallecidos_viaje,

             values="cantidad",

             names="viaje",

             title="Paises de donde retornaron al Peru",

             template="seaborn")



fig.update_traces(rotation=90, pull=0.05, textinfo="value+percent+label")

fig.show()
data_muertes_ingreso_fallecimiento = data_muertes[['fecha_fallecimiento','fecha_ingreso']]

data_muertes_ingreso_fallecimiento = data_muertes_ingreso_fallecimiento.dropna(subset=['fecha_fallecimiento'])

data_muertes_ingreso_fallecimiento = data_muertes_ingreso_fallecimiento.dropna(subset=['fecha_ingreso'])



data_muertes_ingreso_fallecimiento['dias'] = data_muertes_ingreso_fallecimiento.fecha_fallecimiento.dt.day - data_muertes_ingreso_fallecimiento.fecha_ingreso.dt.day



data_muertes_ingreso_fallecimiento = data_muertes_ingreso_fallecimiento.sort_values('dias')
plt.figure(figsize=(15, 5))

plt.title('Numero de dias desde el ingreso hasta la muerte')

data_muertes_ingreso_fallecimiento.dias.value_counts().plot.bar();
numero_fallecidos = data_muertes['insuf_resp'].count()



insuf_positivo = data_muertes['insuf_resp'].sum()

insuf_negativo = np.abs((numero_fallecidos-insuf_positivo))



neumonia_positivo = data_muertes['neumonia'].sum()

neumonia_negativo = numero_fallecidos-neumonia_positivo



data_muertes_insuf_neumo = pd.DataFrame(

                        {

                            'insuf_positivo':[insuf_positivo],

                            'insuf_negativo':[insuf_negativo],

                            'neumonia_positivo':[neumonia_positivo],

                            'neumonia_negativo':[neumonia_negativo]

                        }

                    ) 

fig = plt.figure(figsize=(15, 5))

ax = fig.add_axes([0,0,1,1])

plt.title('Fallecidos con : insuficiencia respiratoria y neumonia')

abscisas = data_muertes_insuf_neumo.columns.tolist()

ordenadas = data_muertes_insuf_neumo.values.tolist()[0]

ax.bar(abscisas,ordenadas)

plt.show()
data_fecha_infectados = data.groupby("date")[['confirmed']].sum().reset_index()

data_fecha_infectados["confirmed_acum"] = data_fecha_infectados.confirmed.cumsum()



x = np.arange(len(data_fecha_infectados)).reshape(-1, 1)

y = data_fecha_infectados.confirmed_acum.values
model = MLPRegressor(hidden_layer_sizes=[32, 32, 10], max_iter=50000, alpha=0.0005, random_state=51)

_=model.fit(x, y)



test = np.arange(len(data_fecha_infectados)+10).reshape(-1, 1)

pred = model.predict(test)



prediction = pred.round().astype(int)

week = [data_fecha_infectados.date[0] + timedelta(days=i) for i in range(len(prediction))]



dt_idx = pd.DatetimeIndex(week)



predicted_count = pd.Series(prediction, dt_idx)
fig = plt.figure(figsize=(15, 5))

plt.plot(data_fecha_infectados.date, data_fecha_infectados.confirmed_acum)

predicted_count.plot()

plt.title('Prediction of Accumulated Confirmed Count')

plt.legend(['current confirmd count', 'predicted confirmed count'])

plt.show()
fig = go.Figure([go.Bar(x=predicted_count.index, y=predicted_count.values)])

fig.show()