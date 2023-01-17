#Librerias



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import plotly.express as px

from matplotlib.pyplot import plot

import seaborn as sn

%matplotlib inline 
# Set your own project id here

PROJECT_ID = 'prueba-data-287014'

from google.cloud import bigquery

from bq_helper import BigQueryHelper



client = bigquery.Client(project=PROJECT_ID)



dataset_ref = client.dataset('Homicidios_2018', project=PROJECT_ID)



# Make an API request to fetch the dataset

dataset = client.get_dataset(dataset_ref)
tables = list(client.list_tables(dataset))



h_df = client.list_rows(tables[1], max_results=5).to_dataframe()

h_df.head()
#Query SQL

query = "SELECT * FROM `Homicidios_2018.Homicidios20181` WHERE NOT (Fecha = 'Fecha' OR Fecha = 'None')";

query_job1 = client.query(query)

df = query_job1.to_dataframe()

print(df.head())
#Query SQL

query = "SELECT count(*) FROM `Homicidios_2018.Homicidios20181`";

query_job1 = client.query(query)



print(query_job1.to_dataframe().head())
#Query SQL

query = "SELECT Fecha FROM `Homicidios_2018.Homicidios20181` WHERE NOT (Fecha = 'Fecha' OR Fecha = 'None')";

query_job1 = client.query(query)



print(query_job1.to_dataframe().head())
#Query SQL

query = "SELECT count(DISTINCT Fecha) FROM `Homicidios_2018.Homicidios20181` WHERE NOT (Fecha = 'Fecha' OR Fecha = 'None')";

query_job1 = client.query(query)



print(query_job1.to_dataframe().head())
#Query SQL

query = "SELECT count(*) FROM `Homicidios_2018.Homicidios20181` WHERE Profesion = '-' GROUP BY Profesion ";

query_job1 = client.query(query)

print(query_job1.to_dataframe())
#Query SQL

query = "SELECT DISTINCT Codigo FROM `Homicidios_2018.Homicidios20181` WHERE NOT (Fecha = 'Fecha' OR Fecha = 'None')";

query_job1 = client.query(query)



print(query_job1.to_dataframe().head())
df.head()
df['Cantidad'] = pd.to_numeric(df['Cantidad'])

df['Edad'] = pd.to_numeric(df['Edad'])

df['Fecha'] = pd.to_datetime(df['Fecha'])

df['Hora'] = pd.to_datetime(df['Hora']).dt.hour

df = df.sort_values(['Fecha'])

#df['Mes'] = df['Fecha'].dt.strftime('%B')

df['Mes'] = df['Fecha'].dt.month

df = df.drop(columns=['Codigo','Profesion'])





df
#Muni = df.groupby([df['Municipio'],df['Fecha'].dt.strftime('%B')])['Cantidad'].sum().sort_values('Mes').reset_index()

Muni = df.groupby([df['Mes'],df['Municipio']])['Cantidad'].sum().sort_values(ascending=False)



#df.groupby([df['Mes']])['Municipio','Cantidad'].max()

Muni.groupby(level=[0,1]).sum().reset_index(['Municipio','Mes']).sort_values(['Mes','Cantidad'],ascending=[1,0]).groupby('Mes').head(1)
df['Cantidad'] = pd.to_numeric(df['Cantidad'])



Municipios_df = df

Municipios_df = Municipios_df.groupby('Municipio').sum()[['Cantidad']].sort_values(by=['Cantidad'],ascending=False)

Municipios_df = Municipios_df.reset_index()
fig = px.bar(Municipios_df[:10], 

             x='Cantidad', y='Municipio', color_discrete_sequence=['#84DCC6'],

             title='Homicidios por Municipio', text='Cantidad', orientation='h')

fig.show()
Bog_df = df.loc[(df['Municipio'] == 'BOGOTÁ D.C. (CT)') & (df['Hora'] >= 18)]

Bog_df = Bog_df.groupby('Barrio').sum().sort_values(by='Cantidad', ascending=False).reset_index()
fig = px.bar(Bog_df[:10], 

             x='Cantidad', y='Barrio', color_discrete_sequence=['#84DCC6'],

             title='Homicidios por barrio en Bogotá', text='Cantidad', orientation='h')

fig.show()
Depto_df = df

Depto_df = Depto_df.groupby('Departamento').sum().sort_values(by=['Cantidad'],ascending=False)

Depto_df = Depto_df.reset_index()
fig = px.bar(Depto_df[:10], 

             x='Cantidad', y='Departamento', color_discrete_sequence=['#84DCC6'],

             title='Homicidios por Departamento', text='Cantidad', orientation='h')

fig.show()
Sexo_df = df

Sexo_df = Sexo_df.groupby('Sexo').sum().sort_values(by=['Cantidad'],ascending=False)

Sexo_df = Sexo_df.reset_index()
fig = px.pie(Sexo_df, values='Cantidad', names= 'Sexo',

             title='Sexo del Agresor')

fig.show()
Dia_df = df.groupby('Dia').sum()[['Cantidad']].reset_index()

Dia_df
fig = px.bar(Dia_df, 

             x='Dia', y='Cantidad', color_discrete_sequence=['#84DCC6'],

             title='Casos por día de la semana', text='Cantidad', orientation='v')

fig.show()
Hom_df = df.groupby('Fecha').sum()[['Cantidad']].sort_values(by='Fecha').reset_index()
fig = px.line(Hom_df, x="Fecha", y="Cantidad", 

              title="Homicidios a través del tiempo")

fig.show()
Hora_df = df.groupby('Hora').sum()[['Cantidad']].sort_values(by='Hora').reset_index()
fig = px.line(Hora_df, x="Hora", y="Cantidad", 

              title="Homicidios Acumulados por Hora")

fig.show()
for i in df.columns:

    print(i,len(df[i].unique()))

object_cols = [col for col in df.columns if df[col].dtype == "object"]
from sklearn.preprocessing import LabelEncoder



aa = df

# Apply label encoder 

encoder = LabelEncoder()



for col in object_cols:

    aa[col] = encoder.fit_transform(aa[col])

    

aa
corr = df.corr()

corr.style.background_gradient(cmap='coolwarm')
K_df = aa.drop(columns=['Cantidad','Fecha','Municipio','Barrio'])
from sklearn import preprocessing

import seaborn as sns



scaler = preprocessing.MinMaxScaler()

caract_normal = scaler.fit_transform(K_df)



pd.DataFrame(caract_normal).describe()
from sklearn.cluster import KMeans

wcss = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=500, n_init=10, random_state=0)

    kmeans.fit(caract_normal)

    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)

plt.title('Elbow Method')

plt.xlabel('Número de Clusters')

plt.ylabel('Inercia')

plt.show()
kmeans = KMeans(n_clusters=2, init='k-means++', max_iter=300, n_init=10, random_state=0)

pred_y = kmeans.fit_predict(caract_normal)
clases = pd.DataFrame(kmeans.labels_)

clases_df = pd.concat((K_df,clases),axis=1)

clases_df = clases_df.rename({0:'clases'},axis=1)

clases_df
sn.lmplot(x='Edad',y='Arma',data=clases_df,hue='clases',fit_reg=False)
sn.lmplot(x='Movil_agresor',y='Edad',data=clases_df,hue='clases',fit_reg=False)