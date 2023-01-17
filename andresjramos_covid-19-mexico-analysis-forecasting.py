#!pip install plotly -U 
#EDA section

import pandas as pd

import numpy as np

import urllib

import zipfile

import os 

import shutil

from datetime import datetime, timedelta

pd.set_option('display.max_columns', None)

from sklearn.cluster import KMeans

from sklearn import preprocessing

from datetime import timedelta

import datetime as dt

import itertools



#Graphic libraries

import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots



#ML libraries

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import mean_squared_error,r2_score



import warnings

warnings.filterwarnings('ignore')
#Download the actual version of data

url = 'http://datosabiertos.salud.gob.mx/gobmx/salud/datos_abiertos/datos_abiertos_covid19.zip'

#Create a new folder to download the data

folder_path = 'DataCovidMx'

#to avoid duplicates when updating, it is necessary to remove the csv

if os.path.exists(folder_path):

    shutil.rmtree(folder_path)

    os.makedirs(folder_path)   

else:

    os.makedirs(folder_path)

#Download the file from URL

urllib.request.urlretrieve(url,"DataCovidMx/DataCovidCSV.zip")

#getting the file name

file_name = zipfile.ZipFile("DataCovidMx/DataCovidCSV.zip").namelist()

#extracting file

with zipfile.ZipFile("DataCovidMx/DataCovidCSV.zip", 'r') as zip_ref:

    zip_ref.extractall("DataCovidMx/")

#Load Datasets

df = pd.read_csv('DataCovidMx/'+ file_name[0], encoding= 'unicode_escape',low_memory=False)

clean_mx = pd.read_csv('../input/covid19-mexico-clean-order-by-states/Covid_19_Mexico_Clean_Complete.csv')

df.head()
state_table = pd.read_excel('../input/dictionary/Catalogo.xlsx', sheet_name='Catálogo de ENTIDADES')

state_table.drop(labels= 'ABREVIATURA', axis = 1, inplace = True)

state_table.rename(columns= {'CLAVE_ENTIDAD':'ID', 'ENTIDAD_FEDERATIVA':'State'},inplace= True)

state_table.head()
#Import Data Dictionary - Municipality ID by State ID

Id_state = pd.read_excel("../input/dictionary/Catalogo.xlsx", sheet_name='Catálogo MUNICIPIOS')



#Creating a new column that contains the Municipality ID by state ID

Id_state['Full_ID']= Id_state['CLAVE_ENTIDAD'].astype(str) + '-' + Id_state['CLAVE_MUNICIPIO'].astype(str)



#Dictionary creation to replace the ID of the "ENTIDAD_RES" and "MUNICIPIO_RES" by name

dict_municipality = pd.Series(Id_state.MUNICIPIO.values, index = Id_state.Full_ID).to_dict()

dict_state = pd.Series(state_table.State.values, index= state_table.ID).to_dict()



#For replace the ID by name it's neccesary create a new column in the original DataSet that contain the State ID and Municipality ID. 

#Defie the new column

df['MUNICIPIO_RES'] = df['ENTIDAD_RES'].astype(str) + '-' + df['MUNICIPIO_RES'].astype(str)



#replace the code with the name

df['ENTIDAD_RES'].replace(dict_state,inplace = True)

df['MUNICIPIO_RES'].replace(dict_municipality,inplace = True)



#replace 1 & 2 as female & male

df['SEXO'].replace(1,'Female',inplace = True)

df['SEXO'].replace(2,'Male',inplace = True)



#Replace column names

actualNames = df.columns.to_list()

newNames = ['Update Date', 'ID', 'USMER','Health_institute', 'Location_institute','Sex','Birth_location','Residence_Entity', 'Residence_Municipality','Patien_type', 'Diagnosis_Date','Symptoms_Date','Death_Date','Tracheal_intubation','Pneumonia','Age','Nationality','Pregnancy','Indigenous_language','Diabetes','EPOC','Asthma','Immunosuppression','Hypertension','Other_comorbidity','Cardiovascular_disease','Obesity', 'Chronic_kidney_disease','smoking','Another_case','Result','Migrant','Nationality Country','Home_country','UCI']

replace_dict = dict(zip(actualNames,newNames))

df.rename(columns=replace_dict,inplace=True)



# Created a new column "Days until decease"

Days_decease = []

df['Symptoms_Date'] = pd.to_datetime(df['Symptoms_Date'])



for income, death in zip(df['Symptoms_Date'], df['Death_Date']):

  if death != '9999-99-99':

    deceaseDay = pd.to_datetime(death)

    days_until_decease = (deceaseDay-income).days

  else:

    days_until_decease = 0

  

  Days_decease.append(days_until_decease)



df['Days_decease'] = Days_decease



#clasify the death cases and non-death caes

df.loc[df.Death_Date != '9999-99-99', 'Death_Date'] = 1  #Pople death

df.loc[df.Death_Date == '9999-99-99', 'Death_Date'] = 0  #People alive

df['Death_Date'] = df['Death_Date'].astype(int)

df.rename(columns = {'Death_Date':'Dead'}, inplace = True)





df.head()
#order data by date

mexico = clean_mx.groupby('Date', as_index=False).sum()

mexico['Date'] = pd.to_datetime(mexico['Date'])



#plot data by confirmed and deaths

fig = px.bar(mexico, x='Date', y='Confirmed',

             hover_data=['Deaths'], color='Deaths',

             labels={}, height=400)

fig.show()



print('\n')

print("Recovered Cases: " ,clean_mx['Recovered'].sum())

print("Death Cases: ",clean_mx['Deaths'].sum())

print("Confirmed Cases: ",clean_mx['Confirmed'].sum())

print("Active Cases: ",clean_mx['Active'].sum())
#plot data

fig = px.histogram(df, x = 'Result', color='Sex')

fig.show()



#Print results

print("\n")

print("Total of infected people(1):", df.loc[df.Result == 1, 'Result'].count())

print("Total of negative people(2):", df.loc[df.Result == 2, 'Result'].count())

print("Total of pending result(3):", df.loc[df.Result == 3, 'Result'].count())
#Positive cases

pc = df.loc[(df['Result'] == 1)]

#Percentage

fem_per = (pc.loc[(pc['Sex'] == 'Female','Sex')].count()  / pc['Sex'].count())*100

male_per = (pc.loc[(pc['Sex'] == 'Male','Sex')].count()  / pc['Sex'].count())*100

#print results

print("Percentage Female cases:", fem_per)

print("Percentage Male cases", male_per)
label=['Death Cases','Surviving Cases']

values = [pc.loc[pc.Dead == 1 , 'Dead'].count(), pc.loc[pc.Dead == 0, 'Dead'].count()]



fig = go.Figure(data=[go.Pie(labels=label, values=values)])

fig.show()
fig = px.histogram(pc, x = 'Patien_type', color = 'Patien_type',title='Patient type')

fig.show()



print('\n')

print('Total outpatient:',pc.loc[pc.Patien_type == 1, 'Patien_type'].count())

print('Total inpatient',pc.loc[pc.Patien_type == 2, 'Patien_type'].count())
fig = px.box(pc, y= 'Age', color= 'Dead')

fig.show()
#Creating a new dataset with only death cases by COVID

death = df.loc[(df['Dead'] == 1) & (df['Result'] == 1)]

death.reset_index(inplace=True)

death = death.iloc[:,[15,16,20,21,22,23,24,26,27,28,29,36]]

print(death.shape)
#replace values 2, 97, 98 and 99 by 0

col = death.columns.to_list()

col.remove('Age')



for col in col: #Replace all columns different from 'EDAD' (Age)

  death[col].replace([2,97,98,99], 0, inplace = True)
X = death

X = preprocessing.StandardScaler().fit(X).transform(X)

X.shape
kclusters = 7

kmeans = KMeans(n_clusters = kclusters, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)

kmeans.fit(X)
death.insert(0, 'Cluster', kmeans.labels_)
Clusters = pd.DataFrame(columns=[], index=[])

Clusters['Group 1'] = death.loc[death['Cluster'] == 0].describe().iloc[1]



Clusters['Group 2'] = death.loc[death['Cluster'] == 1].describe().iloc[1]



Clusters['Group 3'] = death.loc[death['Cluster'] == 2].describe().iloc[1]



Clusters['Group 4'] = death.loc[death['Cluster'] == 3].describe().iloc[1]



Clusters['Group 5'] = death.loc[death['Cluster'] == 4].describe().iloc[1]



Clusters['Group 6'] = death.loc[death['Cluster'] == 5].describe().iloc[1]



Clusters['Group 7'] = death.loc[death['Cluster'] == 6].describe().iloc[1]



Clusters.drop(index='Cluster',inplace=True)

#Clusters.loc[['Age','Days_decease','Days_decease STD']] = Clusters.loc[['Age','Days_decease','Days_decease STD']].apply(round)

Clusters.loc[['Pneumonia','Diabetes','EPOC','Asthma','Immunosuppression','Hypertension','Cardiovascular_disease', 'Obesity','Chronic_kidney_disease','smoking']] = Clusters.loc[['Pneumonia','Diabetes','EPOC','Asthma','Immunosuppression','Hypertension','Cardiovascular_disease', 'Obesity','Chronic_kidney_disease','smoking']] * 100

Clusters
fig = make_subplots(

    rows=3, cols=3,

    subplot_titles=("Group 1", "Group 2", "Group 3", "Group 4", "Group 5", "Group 6", "Group 7"))



fig.add_trace(go.Bar(y=Clusters['Group 1']),1,1)

fig.add_trace(go.Bar(y=Clusters['Group 2']),1,2)

fig.add_trace(go.Bar(y=Clusters['Group 3']),1,3)

fig.add_trace(go.Bar(y=Clusters['Group 4']),2,1)

fig.add_trace(go.Bar(y=Clusters['Group 5']),2,2)

fig.add_trace(go.Bar(y=Clusters['Group 6']),2,3)

fig.add_trace(go.Bar(y=Clusters['Group 7']),3,1)



fig.show()
state = clean_mx.groupby('State', as_index = False).sum()



fig = px.bar(state, y='Confirmed', x='State', text='Confirmed')

fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

fig.show()



print(state.sort_values(by='Confirmed', ascending=False, ignore_index=True).head())
#Transforming the data into a cumulative data to forecasting 

mexico['Deaths'] = mexico['Deaths'].cumsum()

mexico['Confirmed'] = mexico['Confirmed'].cumsum()

mexico['Recovered'] = mexico['Recovered'].cumsum()



#Generate a "day" column since the first case

mexico['Day'] = mexico['Date'].dt.dayofyear



#Plot cumulative data

fig = go.Figure()



fig.add_trace(go.Scatter(x=mexico['Date'], y=mexico['Deaths'],

                    mode='lines+markers',

                    name='Deaths'))



fig.add_trace(go.Scatter(x=mexico['Date'], y=mexico['Confirmed'],

                    mode='lines+markers',

                    name='Confirmed'))



fig.add_trace(go.Scatter(x=mexico['Date'], y=mexico['Recovered'],

                    mode='lines+markers',

                    name='Recovered'))



fig.show()
#defining dependent and independent variables

X = mexico[['Day']]

y = mexico[['Confirmed']]



#split data into test and train

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 1)



#training linear model

lm = LinearRegression()

lm.fit(X_train,y_train)



#evaluating the model

y_pred = lm.predict(X)



#r2 score

print(lm.score(X_test,y_test))
#predicted data

Mexico_lm = mexico.iloc[:,[0,2]]

Mexico_lm['lm_pred'] = y_pred



#plot data comparison

fig = go.Figure()



fig.add_trace(go.Scatter(x=Mexico_lm['Date'], y=Mexico_lm['Confirmed'],

                    mode='lines+markers',

                    name='Confirmed'))



fig.add_trace(go.Scatter(x=Mexico_lm['Date'], y=Mexico_lm['lm_pred'],

                    mode='lines',

                    name='predicted'))



fig.show()
#defining dependent and independent variables

X = mexico[['Day']]

y = mexico[['Confirmed']]



#Transforming data

degree = 8

Poly_reg = PolynomialFeatures(degree)

Xpoly = Poly_reg.fit_transform(X)



#split data into test and train

X_train, X_test, y_train, y_test = train_test_split(Xpoly, y, test_size=0.20, random_state = 2)



#Training model

pm=LinearRegression()

pm.fit(X_train,y_train)



#evaluating the model

y_pred_poly = pm.predict(Xpoly)



#r2 score

print(r2_score(y,y_pred_poly))



#MSE

print(mean_squared_error(y,y_pred_poly,squared=False))
Mexico_pm = mexico.iloc[:,[0,2]]

Mexico_pm['pm_pred'] = y_pred_poly



#plot data comparison

fig = go.Figure()



fig.add_trace(go.Scatter(x=Mexico_pm['Date'], y=Mexico_pm['Confirmed'],

                    mode='lines+markers',

                    name='Confirmed'))



fig.add_trace(go.Scatter(x=Mexico_pm['Date'], y=Mexico_pm['pm_pred'],

                    mode='lines',

                    name='predicted'))



fig.show()
#creating the next 15 days df

pred = pd.DataFrame(columns=[], index=[])

pred["Dates"] = pd.date_range(start="09-10-2020", end="9-30-2020")
#Prediction for Linear model

X = pred["Dates"].dt.dayofyear

X = X.values.reshape(-1,1)

prediction_linear = lm.predict(X)

pred['Prediction_linear'] = prediction_linear.round()
#Prediction for polynomial model

X = pred["Dates"].dt.dayofyear

X = X.values.reshape(-1,1)

X = Poly_reg.fit_transform(X)

prediction_poly = pm.predict(X)

pred['Prediction_poly'] = prediction_poly.round()
#Plot model comparison



fig = go.Figure()



fig.add_trace(go.Scatter(x=pred['Dates'], y=pred['Prediction_linear'],

                    mode='lines+markers',

                    name='Linear model prediction'))



fig.add_trace(go.Scatter(x=pred['Dates'], y=pred['Prediction_poly'],

                    mode='lines+markers',

                    name='Polynomial model prediction'))



fig.add_trace(go.Scatter(x=Mexico_pm['Date'], y=Mexico_pm['Confirmed'],

                    mode='lines+markers',

                    name='Confirmed'))





fig.show()



print(pred)