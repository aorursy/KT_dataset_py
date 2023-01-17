### Load the package

import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
from datetime import datetime as dt
import matplotlib.dates as mdates
import datetime



# Load the dataset
data_corona=pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv", index_col=0)
data_brasil=pd.read_csv("../input/corona-virus-brazil/brazil_covid19.csv")
data_brasil_states=pd.read_csv("../input/brazilianstates/states.csv")
data_idh_populacao=pd.read_csv("../input/population-hdi-countries/idh_populacao.csv",index_col=0)





today = datetime.datetime.now().strftime('%d/%m/%Y')
print(datetime.datetime.now())


# Data cleaning


df = data_corona.drop('Last Update',axis=1)

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
df.index = range(len(df))# redefinindo os índices após colocar em ordem alfabética


# Removendo países com poucos dados
numero_dados_minimo = 10
for i in df['Country/Region'].unique():
    aux= df.loc[df['Country/Region']==i]
    if len(aux)<numero_dados_minimo:
        for j in aux.index:
            df = df.drop(j)
df=df.sort_values(['Country/Region','ObservationDate'],ascending =  [True ,False])
df.index = range(len(df))# redefinindo os índices após colocar em ordem alfabética 


## Arrumando o nome dos eua e china
df = df.set_index('Country/Region')

df= df.rename({'US':'United States'})
df= df.rename({'Mainland China':'China'})
df ['Active cases']= df['Confirmed']- df['Deaths']-df['Recovered']
df['Country/Region'] = df.index
df.index = range(len(df))# redefinindo os índices após colocar em ordem alfabética

df=df.sort_values(['Country/Region','ObservationDate'],ascending =  [True ,False])
df.index = range(len(df))# redefinindo os índices após colocar em ordem alfabética

df3=df[df.columns]

# Dinamics of some countries and Brazil
import warnings  
warnings.filterwarnings('ignore')

#Contry more afected
n_cr = 4
aux2 = pd.DataFrame(columns=['Country','Number'])
for i in df['Country/Region'].unique():
    aux= df.loc[df['Country/Region']==i]
    aux2=aux2.append({'Country':i,'Number':aux['Active cases'].max()},ignore_index=True)
aux2 = aux2.set_index('Country')
paises_analisados = list(aux2['Number'].nlargest(n_cr).index)   



if  not 'Brazil' in paises_analisados:
    paises_analisados.append('Brazil')

if not 'Italy' in paises_analisados:
        paises_analisados.append('Italy')

if not 'China' in paises_analisados:
        paises_analisados.append('China')

fig,ax = plt.subplots(1,2 ,figsize=(20, 8))


ax[0].grid('True')
ax[0].set_title("Número de casos ativos",fontsize=20)
for i in paises_analisados:
    aux = df.loc[df['Country/Region']==i]
    ax[0].plot(aux['ObservationDate'],aux['Active cases'], label = i, linewidth=4)
    

ax[0].legend(loc='best',fontsize=20)
ax[0].set_xlabel('Data',fontsize=20)
    
import matplotlib.ticker as ticker # pacote para colocar as datas em formato cientifico

#ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.2f}'.format(x/1000) + 'K'))
ax[0].yaxis.set_major_formatter(ticker.EngFormatter())# isto põe os numeros no formato de engenheiro
ax[0].xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
ax[0].xaxis.set_tick_params(labelsize=20)
ax[0].yaxis.set_tick_params(labelsize=20)

# Changing the date, instead of entering the date, the day since the first infection will be placed
today = date.today() # load the data now
df2 = df
for i in df['Country/Region'].unique():
    aux = df.loc[df['Country/Region']==i]
    data_primeiro_caso = aux.loc[aux['Confirmed']!=0]['ObservationDate'].min()
    for j in aux.index:
        df2.loc[j,'ObservationDate'] =(df['ObservationDate'][j] - data_primeiro_caso).days 
        
# Removnedo linhas antes do primeiro caso
df2 = df2.drop(df[df['ObservationDate']<0].index)



  


ax[1].grid('True')
ax[1].set_title("Número de casos ativos",fontsize=20)
for i in paises_analisados:
    aux = df2.loc[df['Country/Region']==i]
    ax[1].plot(aux['ObservationDate'],aux['Active cases'], label = i, linewidth=4)
    

ax[1].legend(loc='best',fontsize=20)
ax[1].set_xlabel('Dias após primeiro caso',fontsize=20)

#ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:,.2f}'.format(x/1000) + 'K'))
ax[1].yaxis.set_major_formatter(ticker.EngFormatter())# isto põe os numeros no formato de engenheiro
ax[1].xaxis.set_tick_params(labelsize=20)
ax[1].yaxis.set_tick_params(labelsize=20)

plt.show()
fig,ax = plt.subplots(figsize=(20, 8))

cases_state = data_brasil.loc[data_brasil['date']== data_brasil['date'].max()]

cases_state=cases_state.sort_values(['state'])




p1 = plt.bar(cases_state['state'], cases_state['cases'],label = 'Insiria uma legenda aqui')


plt.title('Incidência por estado',fontsize=30)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
plt.xticks(rotation=90)
ax.yaxis.set_major_formatter(ticker.EngFormatter())# isto põe os numeros no formato de engenheiro


xlocs, xlabs = plt.xticks()

y = list(cases_state['cases'])

# create a list to collect the plt.patches data
totals = []



# set individual bar lables using above list
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x()-.03, i.get_height()+40,str(i.get_height()), fontsize=15)


plt.show()
cases_state = cases_state.set_index('state')
populacao_brasil = data_brasil_states[['State','Population']].set_index('State')
df_br = cases_state.merge(populacao_brasil, left_index=True, right_index=True)

df_br=df_br.sort_index()


fig,ax = plt.subplots(figsize=(20, 8))
p1 = plt.bar(df_br.index, df_br['cases']/df_br['Population']*100000)
plt.title('Número de infectados a cada 100 mil habitantes',fontsize=30)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
plt.xticks(rotation=90)
ax.yaxis.set_major_formatter(ticker.EngFormatter())# isto põe os numeros no formato de engenheiro
xlocs, xlabs = plt.xticks()

y = list(cases_state['cases'])

# set individual bar lables using above list
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x()-.03, i.get_height()+0.1,str(round(i.get_height(),2)), fontsize=15)

plt.show()
data_today = pd.DataFrame(columns= ['ObservationDate', 'Confirmed', 'Deaths', 'Recovered', 'Active cases',\
       'Country/Region'])
for i in df['Country/Region'].unique():
        aux= df.loc[df['Country/Region']==i]
        data_today= data_today.append(aux[aux['ObservationDate']== aux['ObservationDate'].max()])
        
data_today =  data_today.set_index('Country/Region')

data_today =  data_today.merge(data_idh_populacao, left_index=True, right_index=True)

corr_hdi_deaths = data_today['Deaths'].corr(data_today['humanDevelopmentIndex'])
print(corr_hdi_deaths)  

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

# Criando feacture primeira derivada
df2['Primeira derivada']=0
for i in df2['Country/Region'].unique():
    aux= df2.loc[df2['Country/Region']==i]
    for j in aux.index[0:len(aux.index)-3]:
        df2.loc[j,'Primeira derivada']= -aux['Active cases'].diff()[j+2]
        
        
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
            df2.loc[j,'Media segunda derivada']=aux['Segunda derivada'].rolling(window_size).mean()[j+3]
        else:
            df2.loc[j,'Media segunda derivada']=0
        
        
df2['Valor anterior'] = 0
for i in df2['Country/Region'].unique():
    aux= df2.loc[df2['Country/Region']==i]
    for j in aux.index:
        if j+1<(aux.index[len(aux.index)-1]):
            df2.loc[j,'Valor anterior']=aux['Active cases'][j+1]
        else:
            df2.loc[j,'Valor anterior']=0 
df2=df2.sort_values(['Country/Region','ObservationDate'],ascending =  [True ,False])
df2.index = range(len(df2))# redefinindo os índices após colocar em ordem alfabética


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



 
poly = PolynomialFeatures(degree = 3)
X_poly = poly.fit_transform(x) 
  
poly.fit(X_poly, y) 
lin2 = LinearRegression() 
lin2.fit(X_poly, y) 






## Previsão para o brasil

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
    

    
    
    
    
    #Normalizando os dados novamente
    x_teste= df2[['ObservationDate','Media primeira derivada','Media segunda derivada', 'Valor anterior','Primeira derivada', 'Segunda derivada']]
    
    scaler = preprocessing.MinMaxScaler()
    scaled_df = scaler.fit_transform(x_teste)
    scaled_df = pd.DataFrame(scaled_df, columns=['ObservationDate', 'Media primeira derivada'\
                                             , 'Media segunda derivada','Valor anterior'\
                                                 ,'Primeira derivada', 'Segunda derivada'])
    x_teste=scaled_df
    x_teste_brasil = x_teste[df2['Country/Region']=='Brazil']

    # Prevendo resultados
    prev=lin2.predict(poly.fit_transform(pd.DataFrame( x_teste_brasil.head(1)) ))
    index_change = df2[df2['Country/Region']=='Brazil'].index[0]
    df2.loc[index_change,'Active cases'] = prev[0]
# df2[df2['Active cases']<0]=0

# Plotando os resultados
    
inteirar = lambda t: int(t)
y_pred = np.array([inteirar(xi) for xi in df2_brasil['Active cases']])
y_pred = y_pred[::-1]


def date_linspace(start, end, steps):
  delta = (end - start) / steps
  increments = range(0, steps) * np.array([delta]*steps)
  return start + increments



data_first_case_brasil = date(2020,2,26)
label_days = date_linspace(data_first_case_brasil ,date(today.year,today.month+int(numero_dias_previsto/30),today.day),len(y_pred))



y_pred = pd.Series(y_pred)
y_pred.index = label_days



# fig, ax = plt.subplots()
# ax.plot(y_pred)
# ax = plt.gca()
# locs, labels=plt.xticks()
# locs = [locs[i] for i in np.arange(0, len(locs), 16)]
# new_xticks=aaa
# plt.xticks(locs,new_xticks, rotation=45)
# plt.xlabel('Date')
# plt.ylabel('Number of active cases')
# plt.title('Forecast of corona virus in Brazil')
# plt.grid('True')
# plt.show()

# y_pred[dayx+1:].head(numero_dias_previsto)








# ax.set_xticks(label_days)
# ax.set_xticklabels([label_days[5*i] for i in range(1,int(len(label_days)/5)) ])
# plt.xticks( aaa,np.arange(0, len(label_days), 5),rotation=90) 








fig, ax = plt.subplots(figsize = (14,8))
ax.plot(y_pred.index,y_pred, linewidth=8,color= 'red')
ax.set_xlabel('Data',fontsize=20)    
import matplotlib.ticker as ticker # pacote para colocar as datas em formato cientifico

ax.yaxis.set_major_formatter(ticker.EngFormatter())# isto põe os numeros no formato de engenheiro
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
ax.grid('True')
ax.set_title("Número de casos ativos",fontsize=40)
plt.show()
plt.show()


y_pred.to_csv(str(today)+'_ativos.csv')


previsoes_confirmados = pd.Series.to_frame(y_pred)
previsoes_confirmados=previsoes_confirmados.rename(columns={0: 'Confirmados'})
previsoes_confirmados =previsoes_confirmados.rename_axis( 'Data')

previsoes_confirmados.iloc[dayx+1:].head(numero_dias_previsto)

# Plot hte relationbetween recovered and confirmed


# Selecting Contry more afected
n_cr = 10
aux2 = pd.DataFrame(columns=['Country','Number'])
for i in df['Country/Region'].unique():
    aux= df.loc[df['Country/Region']==i]
    aux2=aux2.append({'Country':i,'Number':aux['Active cases'].max()},ignore_index=True)
aux2 = aux2.set_index('Country')
paises_analisados = list(aux2['Number'].nlargest(n_cr).index)   



if 'Brazil' not in paises_analisados:# selecting Brazil an d china because they are important
    paises_analisados.append('Brazil')
    
if 'China' not in paises_analisados:
    paises_analisados.append('China')


data_ultimo_dia_paises_analisados=pd.DataFrame(columns= df3.columns)
for i in paises_analisados:
    aux = df3.loc[df3['Country/Region']==i]
    data_ultimo_dia= aux.loc[aux['ObservationDate']==aux['ObservationDate'].max()]
    data_ultimo_dia_paises_analisados=data_ultimo_dia_paises_analisados.append(data_ultimo_dia)

data_ultimo_dia_paises_analisados=data_ultimo_dia_paises_analisados.set_index('Country/Region')
razao_recuperado_confirmado= pd.Series(index = paises_analisados, dtype=float)
razao_morto_confirmado= pd.Series(index = paises_analisados, dtype=float)

for i in paises_analisados:
    razao_recuperado_confirmado[i] = 100*data_ultimo_dia_paises_analisados['Recovered'][i]/data_ultimo_dia_paises_analisados['Confirmed'][i]
    razao_morto_confirmado[i] = 100*data_ultimo_dia_paises_analisados['Deaths'][i]/data_ultimo_dia_paises_analisados['Confirmed'][i]

razao_morto_confirmado=razao_morto_confirmado.sort_index()
razao_recuperado_confirmado=razao_recuperado_confirmado.sort_index()

fig,ax = plt.subplots(figsize=(20, 8))



p1 = plt.bar(razao_recuperado_confirmado.index, round(razao_recuperado_confirmado,2))


plt.title('Taxa de recuperação (%)',fontsize=30)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
plt.xticks(rotation=45)
ax.yaxis.set_major_formatter(ticker.EngFormatter())# isto põe os numeros no formato de engenheiro


xlocs, xlabs = plt.xticks()

y = list(cases_state['cases'])

# create a list to collect the plt.patches data
totals = []

# set individual bar lables using above list
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x()+.11, i.get_height()+0.1,str(i.get_height()), fontsize=15)

fig,ax = plt.subplots(figsize=(20, 8))



p1 = plt.bar(razao_morto_confirmado.index, round(razao_morto_confirmado,2))


plt.title('Taxa de mortalidade (%)',fontsize=30)
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)
plt.xticks(rotation=45)
ax.yaxis.set_major_formatter(ticker.EngFormatter())# isto põe os numeros no formato de engenheiro


xlocs, xlabs = plt.xticks()

y = list(cases_state['cases'])

# create a list to collect the plt.patches data
totals = []

# set individual bar lables using above list
for i in ax.patches:
    # get_x pulls left or right; get_height pushes up or down
    ax.text(i.get_x()+.11, i.get_height()+0.1,str(i.get_height()), fontsize=15)

plt.show()


# This file make the forecast of 'Confirmed' cases of  corona virus in april


# Criando feacture primeira derivada
df3['Primeira derivada']=0
for i in df3['Country/Region'].unique():
    aux= df3.loc[df3['Country/Region']==i]
    for j in aux.index[0:len(aux.index)-3]:
        df3.loc[j,'Primeira derivada']= -aux['Confirmed'].diff()[j+2]
        
        
df3['Segunda derivada']=0
for i in df3['Country/Region'].unique():
    aux= df3.loc[df3['Country/Region']==i]
    for j in aux.index[0:len(aux.index)-3]:
        df3.loc[j,'Segunda derivada']= -aux['Primeira derivada'].diff()[j+2]

           


# Criando  media segunda derivada   
window_size = 4
df3['Media primeira derivada'] = 0
for i in df3['Country/Region'].unique():
    aux= df3.loc[df3['Country/Region']==i]
    for j in aux.index:
        if j+window_size-1<(aux.index[len(aux.index)-1]):
            df3.loc[j,'Media primeira derivada']=aux['Primeira derivada'].rolling(window_size).mean()[j+3]
        else:
            df3.loc[j,'Media primeira derivada']=0
            
df3['Media segunda derivada'] = 0
for i in df3['Country/Region'].unique():
    aux= df3.loc[df3['Country/Region']==i]
    for j in aux.index:
        if j+window_size-1<(aux.index[len(aux.index)-1]):
            df3.loc[j,'Media segunda derivada']=aux['Segunda derivada'].rolling(window_size).mean()[j+3]
        else:
            df3.loc[j,'Media segunda derivada']=0
        
        
df3['Valor anterior'] = 0
for i in df3['Country/Region'].unique():
    aux= df3.loc[df3['Country/Region']==i]
    for j in aux.index:
        if j+1<(aux.index[len(aux.index)-1]):
            df3.loc[j,'Valor anterior']=aux['Confirmed'][j+1]
        else:
            df3.loc[j,'Valor anterior']=0           













df3=df3.sort_values(['Country/Region','ObservationDate'],ascending =  [True ,False])
df3.index = range(len(df3))# redefinindo os índices após colocar em ordem alfabética





# Changing the date, instead of entering the date, the day since the first infection will be placed
today = date.today() # load the data now
for i in df3['Country/Region'].unique():
    aux = df3.loc[df3['Country/Region']==i]
    data_primeiro_caso = aux.loc[aux['Confirmed']!=0]['ObservationDate'].min()
    for j in aux.index:
        df3.loc[j,'ObservationDate'] =(df3['ObservationDate'][j] - data_primeiro_caso).days 
        
# Removnedo linhas antes do primeiro caso
df3 = df3.drop(df3[df3['ObservationDate']<0].index)
















df3=df3.sort_values(['Country/Region','ObservationDate'],ascending =  [True ,False])
df3.index = range(len(df3))# redefinindo os índices após colocar em ordem alfabética


## Machine learning

y  = df3['Confirmed']

x= df3[['ObservationDate','Media primeira derivada','Media segunda derivada', 'Valor anterior','Primeira derivada', 'Segunda derivada']]


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






## Previsão para o brasil

x_brasil = x.loc[df3['Country/Region']=='Brazil']
y_brasil = y.loc[df3['Country/Region']=='Brazil']
df3_brasil = df3.loc[df3['Country/Region']=='Brazil']

dayx = df3_brasil['ObservationDate'].max()
numero_dias_previsto = 60


for i in range(1,numero_dias_previsto):
    # Retroalimentação
    df3_brasil = df3.loc[df3['Country/Region']=='Brazil']
    df3_brasil=df3_brasil.sort_values(['ObservationDate'],ascending =  [False])
    df3_brasil.index = range(len(df3_brasil))# redefinindo os índices após colocar em ordem alfabética 
    data_ultimo_dia = df3_brasil[df3_brasil['ObservationDate']==df3_brasil['ObservationDate'].max()].copy()

    data_ultimo_dia['ObservationDate']+=1
    data_ultimo_dia['Primeira derivada']=df3_brasil['Confirmed'][df3_brasil.index[0]]-df3_brasil['Confirmed'][df3_brasil.index[1]]
    data_ultimo_dia['Segunda derivada'] = df3_brasil['Primeira derivada'][df3_brasil.index[0]]-df3_brasil['Primeira derivada'][df3_brasil.index[1]]
    
    df3_brasil= df3_brasil.append(data_ultimo_dia)
    df3_brasil=df3_brasil.sort_values(['ObservationDate'],ascending =  [False])
    df3_brasil.index = range(len(df3_brasil))# redefinindo os índices após colocar em ordem alfabética 
    
    df3_brasil.loc[0,'Media primeira derivada'] = df3_brasil['Media primeira derivada'].rolling(window_size).mean()[3] 
    df3_brasil.loc[0,'Media segunda derivada'] = df3_brasil['Media segunda derivada'].rolling(window_size).mean()[3] 
    df3_brasil.loc[0,'Valor anterior']= df3_brasil['Confirmed'][1]
    df3= df3.append(df3_brasil.iloc[0])
    # valores_maximos = df3.max(axis=0)
    df3=df3.sort_values(['Country/Region','ObservationDate'],ascending =  [True ,False])
    df3.index = range(len(df3))# redefinindo os índices após colocar em ordem alfabética 
    

    
    
    
    
    #Normalizando os dados novamente
    x_teste= df3[['ObservationDate','Media primeira derivada','Media segunda derivada', 'Valor anterior','Primeira derivada', 'Segunda derivada']]
    
    scaler = preprocessing.MinMaxScaler()
    scaled_df = scaler.fit_transform(x_teste)
    scaled_df = pd.DataFrame(scaled_df, columns=['ObservationDate', 'Media primeira derivada'\
                                             , 'Media segunda derivada','Valor anterior'\
                                                 ,'Primeira derivada', 'Segunda derivada'])
    x_teste=scaled_df
    x_teste_brasil = x_teste[df3['Country/Region']=='Brazil']

    # Prevendo resultados
    prev=lin2.predict(poly.fit_transform(pd.DataFrame( x_teste_brasil.head(1)) ))
    index_change = df3[df3['Country/Region']=='Brazil'].index[0]
    df3.loc[index_change,'Confirmed'] = prev[0]
# df3[df3['Active cases']<0]=0

# Plotando os resultados
    
y_pred = np.array([inteirar(xi) for xi in df3_brasil['Confirmed']])
y_pred = y_pred[::-1]

data_first_case_brasil = date(2020,2,26)

label_days = date_linspace(data_first_case_brasil ,date(today.year,today.month+int(numero_dias_previsto/30),today.day),len(y_pred))



y_pred = pd.Series(y_pred)
y_pred.index = label_days


warnings.filterwarnings('ignore')

fig, ax = plt.subplots(figsize = (21,12))
ax.plot(y_pred,linewidth=8)
ax.set_xlabel('Data',fontsize=20)    
import matplotlib.ticker as ticker # pacote para colocar as datas em formato cientifico

ax.yaxis.set_major_formatter(ticker.EngFormatter())# isto põe os numeros no formato de engenheiro
ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
ax.xaxis.set_tick_params(labelsize=20)
ax.yaxis.set_tick_params(labelsize=20)

ax.grid('True')
ax.set_title("Número de casos confirmados",fontsize=40)
plt.show()

previsoes_confirmados = pd.Series.to_frame(y_pred)
previsoes_confirmados=previsoes_confirmados.rename(columns={0: 'Confirmados'})
previsoes_confirmados =previsoes_confirmados.rename_axis( 'Data')
previsoes_confirmados.iloc[dayx+1:].head(numero_dias_previsto)

previsao_mortos_brasil = int(razao_morto_confirmado.Brazil*previsoes_confirmados.max()/100)
print(previsao_mortos_brasil)

y_pred.to_csv(str(today)+'_confirmados.csv')
