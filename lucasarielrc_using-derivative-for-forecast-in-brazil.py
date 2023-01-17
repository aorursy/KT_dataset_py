# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 09:08:53 2020

@author: lucas
"""

### Load the package

import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
%matplotlib inline 


# Load the dataset

data_corona=pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv", index_col=0)


# Data cleaning

df = data_corona.drop('Last Update',axis=1)
today = date.today() # load the data now

# Change data format
for i in df.index:
    data = df['ObservationDate'][i]
    new_data= date(int(data.split('/')[2]),int(data.split('/')[0]),int(data.split('/')[1]))
    df.loc[i,'ObservationDate']=new_data
    
    
# Sorting the data
df=df.sort_values(['Country/Region','ObservationDate'],ascending =  [True ,False])
df.index = range(len(df))# redefinindo os índices após colocar em ordem alfabética

# Bringing together different provinces from the same country
for i in df['Country/Region'].unique():
    aux = df.loc[df['Country/Region']==i]
    if aux['Province/State'].isna().sum()==0:
        for j in aux['ObservationDate'].unique():
            aux2 = aux.loc[aux['ObservationDate']==j]
            df  = df.append({'ObservationDate':j,'Country/Region':i,'Confirmed':\
                             aux2['Confirmed'].sum(),'Deaths': aux2['Deaths'].sum(), \
                                 'Recovered': aux2['Recovered'].sum()}, ignore_index= True)
                
df = df.loc[df['Province/State'].isnull()]
df = df.drop('Province/State',axis=1)
    
df=df.sort_values(['Country/Region','ObservationDate'],ascending =  [True ,False])
df.index = range(len(df))# set the index

# Changing the date, instead of entering the date, the day since the first infection will be placed
for i in df['Country/Region'].unique():
    aux = df.loc[df['Country/Region']==i]
    data_primeiro_caso = aux.loc[aux['Confirmed']!=0]['ObservationDate'].min()
    for j in aux.index:
        df.loc[j,'ObservationDate'] =(df['ObservationDate'][j] - data_primeiro_caso).days 

# Remove lines before the first case
df = df.drop(df[df['ObservationDate']<0].index)

df = df.set_index('Country/Region')

# Set the name of United States and China
df= df.rename({'US':'United States'})
df= df.rename({'Mainland China':'China'})
df ['Active cases']= df['Confirmed']- df['Deaths']-df['Recovered']
df['Country/Region'] = df.index



# Dinamics of some countries and Brazil


paises_analisados = ['China','Brazil','Italy','France',\
                     'Germany']

fig, ax = plt.subplots()
for i in paises_analisados:
    aux = df.loc[df['Country/Region']==i]
    plt.plot(aux['ObservationDate'],aux['Active cases'], label = i)
    

plt.legend(loc='best')
plt.xlabel('Days')
plt.grid('True')

plt.ylabel('Number of active cases')
plt.title('Active cases of corona virus')
plt.show()

df2=df
df2.index = range(len(df2))# redefinindo os índices após colocar em ordem alfabética

df2=df2.sort_values(['Country/Region','ObservationDate'],ascending =  [True ,False])
df2.index = range(len(df2))# redefinindo os índices após colocar em ordem alfabética

# Remove countries with little data
numero_dados_minimo = 10
for i in df2['Country/Region'].unique():
    aux= df2.loc[df2['Country/Region']==i]
    if len(aux)<numero_dados_minimo:
        for j in aux.index:
            df2 = df2.drop(j)
df2=df2.sort_values(['Country/Region','ObservationDate'],ascending =  [True ,False])
df2.index = range(len(df2))
# Create feature first derivative
df2['Primeira derivada']=0
for i in df2['Country/Region'].unique():
    aux= df2.loc[df2['Country/Region']==i]
    for j in aux.index[0:len(aux.index)-3]:
        df2.loc[j,'Primeira derivada']= -aux['Active cases'].diff()[j+2]
        
#Create feature second derivative       
df2['Segunda derivada']=0
for i in df2['Country/Region'].unique():
    aux= df2.loc[df2['Country/Region']==i]
    for j in aux.index[0:len(aux.index)-3]:
        df2.loc[j,'Segunda derivada']= -aux['Primeira derivada'].diff()[j+2]
        
    

# Criando  media segunda derivada   
window_size = 4
df2['Media primeira derivada'] = 0
for i in df2['Country/Region'].unique():
    aux= df2.loc[df2['Country/Region']==i]
    for j in aux.index:
        if j+window_size-1<(aux.index[len(aux.index)-1]):
            df2.loc[j,'Media primeira derivada']=aux['Primeira derivada'].rolling(window_size).mean()[j+3]
        else:
            df2.loc[j,'Media primeira derivada']=0
            
df2['Media segunda derivada'] = 0
for i in df2['Country/Region'].unique():
    aux= df2.loc[df2['Country/Region']==i]
    for j in aux.index:
        if j+window_size-1<(aux.index[len(aux.index)-1]):
            df2[j,'Media segunda derivada']=aux['Segunda derivada'].rolling(window_size).mean()[j+3]
        else:
            df2[j,'Media segunda derivada']=0

df2['Valor anterior'] = 0
for i in df2['Country/Region'].unique():
    aux= df2.loc[df2['Country/Region']==i]
    for j in aux.index:
        if j+1<(aux.index[len(aux.index)-1]):
            df2.loc[j,'Valor anterior']=aux['Active cases'][j+1]
        else:
            df2.loc[j,'Valor anterior']=0           

## Machine learning

y  = df2['Active cases']

x= df2[['ObservationDate','Media primeira derivada','Media segunda derivada', 'Valor anterior','Primeira derivada', 'Segunda derivada']]


# Feature scaling
from sklearn import preprocessing

scaler = preprocessing.MinMaxScaler()
scaled_df = scaler.fit_transform(x)
scaled_df = pd.DataFrame(scaled_df, columns=['ObservationDate', 'Media primeira derivada'\
                                             , 'Media segunda derivada','Valor anterior'\
                                                 ,'Primeira derivada', 'Segunda derivada'])


x=scaled_df



from sklearn.preprocessing import PolynomialFeatures 
from sklearn.linear_model import LinearRegression 
lin = LinearRegression() 
  
lin.fit(x, y)  



 
poly = PolynomialFeatures(degree = 2)
X_poly = poly.fit_transform(x) 
  
poly.fit(X_poly, y) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y) 






## Forecast to brazil

x_brasil = x.loc[df2['Country/Region']=='Brazil']
y_brasil = y.loc[df2['Country/Region']=='Brazil']
df2_brasil = df2.loc[df2['Country/Region']=='Brazil']

dayx = df2_brasil['ObservationDate'].max()
numero_dias_previsto = 60


for i in range(1,numero_dias_previsto):
    # Retroalimentação
    df2_brasil = df2.loc[df2['Country/Region']=='Brazil']
    df2_brasil=df2_brasil.sort_values(['ObservationDate'],ascending =  [False])
    df2_brasil.index = range(len(df2_brasil))# redefinindo os índices após colocar em ordem alfabética 
    data_ultimo_dia = df2_brasil[df2_brasil['ObservationDate']==df2_brasil['ObservationDate'].max()].copy()

    data_ultimo_dia['ObservationDate']+=1
    data_ultimo_dia['Primeira derivada']=df2_brasil['Active cases'][df2_brasil.index[0]]-df2_brasil['Active cases'][df2_brasil.index[1]]
    data_ultimo_dia['Segunda derivada'] = df2_brasil['Primeira derivada'][df2_brasil.index[0]]-df2_brasil['Primeira derivada'][df2_brasil.index[1]]
    
    df2_brasil= df2_brasil.append(data_ultimo_dia)
    df2_brasil=df2_brasil.sort_values(['ObservationDate'],ascending =  [False])
    df2_brasil.index = range(len(df2_brasil))# redefinindo os índices após colocar em ordem alfabética 
    
    df2_brasil.loc[0,'Media primeira derivada'] = df2_brasil['Media primeira derivada'].rolling(window_size).mean()[3] 
    df2_brasil.loc[0,'Media segunda derivada'] = df2_brasil['Media segunda derivada'].rolling(window_size).mean()[3] 
    df2_brasil.loc[0,'Valor anterior']= df2_brasil['Active cases'][1]
    df2= df2.append(df2_brasil.iloc[0])
    # valores_maximos = df2.max(axis=0)
    df2=df2.sort_values(['Country/Region','ObservationDate'],ascending =  [True ,False])
    df2.index = range(len(df2))# redefinindo os índices após colocar em ordem alfabética 
    

    
    
    
    
    #Normalize the data again
    x_teste= df2[['ObservationDate','Media primeira derivada','Media segunda derivada', 'Valor anterior','Primeira derivada', 'Segunda derivada']]
    
    scaler = preprocessing.MinMaxScaler()
    scaled_df = scaler.fit_transform(x_teste)
    scaled_df = pd.DataFrame(scaled_df, columns=['ObservationDate', 'Media primeira derivada'\
                                             , 'Media segunda derivada','Valor anterior'\
                                                 ,'Primeira derivada', 'Segunda derivada'])
    x_teste=scaled_df
    x_teste_brasil = x_teste[df2['Country/Region']=='Brazil']

    # Forecast results
    prev=lin2.predict(poly.fit_transform(pd.DataFrame( x_teste_brasil.head(1)) ))
    index_change = df2[df2['Country/Region']=='Brazil'].index[0]
    df2.loc[index_change,'Active cases'] = prev[0]
# df2[df2['Active cases']<0]=0


# Plot the results
    
inteirar = lambda t: int(t)
y_pred = np.array([inteirar(xi) for xi in df2_brasil['Active cases']])
y_pred = y_pred[::-1]


def date_linspace(start, end, steps):
  delta = (end - start) / steps
  increments = range(0, steps) * np.array([delta]*steps)
  return start + increments



data_first_case_brasil = date(2020,2,26)
label_days = date_linspace(data_first_case_brasil ,date(today.year,today.month+2,today.day),len(y_pred))
label_days = [str(str(i).split('-')[2]+'/'+str(i).split('-')[1]) for i in label_days]


aaa=[label_days[i] for i in np.arange(0, len(label_days), 16)]

y_pred = pd.Series(y_pred)
y_pred.index = label_days



fig, ax = plt.subplots()
ax.plot(y_pred)
ax = plt.gca()
locs, labels=plt.xticks()
locs = [locs[i] for i in np.arange(0, len(locs), 16)]
new_xticks=aaa
plt.xticks(locs,new_xticks, rotation=45)
plt.xlabel('Date')
plt.ylabel('Number of active cases')
plt.title('Forecast of corona virus in Brazil')
plt.grid('True')
plt.show()

y_pred[dayx+1:].head(numero_dias_previsto)
