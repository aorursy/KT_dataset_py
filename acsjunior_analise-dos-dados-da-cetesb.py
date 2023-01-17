import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from fbprophet import Prophet
from sklearn.metrics import mean_absolute_error
from matplotlib.cbook import boxplot_stats
parser = (lambda x:datetime.datetime.strptime(x, '%Y.%m.%d')) 
df = pd.read_csv('../input/sp-beaches-update/sp_beaches_update.csv', parse_dates=['Date'])
df.head()
print(f'Numero de praias: {len(df.Beach.unique())}') 
print(f'Numero de cidades: {len(df.City.unique())}')
print('Dados de: {} até {}'.format(df.Date.min().year, df.Date.max().year ))
print(df.isnull().sum(axis=0)) 
df.info() # Nao tem dados faltando
df2=df.loc[~df['Enterococcus'].isnull()]
df2.info()

plt.figure(figsize=(30,10))
sns.boxplot(y=df['Enterococcus'], x=df['Beach'])
for city in df.City.unique():
   # print(df.loc[df['City']==city])
    plt.figure(figsize=(30,5))
    sns.boxplot(y=df['Enterococcus'], x=df.loc[df['City']==city]['Beach']).set_title(city)
#df.loc[df['Beach']=="PULSO"].loc[df['City']=="BERTIOGA"]
df_pivot = df.sort_values(by=['Date'])
for city in df_pivot.City.unique():
    df_city = df_pivot.loc[df_pivot['City']==city].pivot( index='Date', columns='Beach', values='Enterococcus').fillna(0)
    df_city.plot(figsize=(30,5), title=city)
    x_coordinates= [df_pivot.loc[df_pivot['City']==city]['Date'].min(), df_pivot.loc[df_pivot['City']==city]['Date'].max()]
    y_coordinates100= [100,100]
    plt.plot(x_coordinates, y_coordinates100,'r--')
    y_coordinates400= [400,400]
    plt.plot(x_coordinates, y_coordinates400,'k--')
    
#df_pivot = df_pivot.dropna(subset=['Date'])
#df_pivot.loc[df_pivot['Date'].isnull()]
#df_pivot = df_pivot.loc[df_pivot['City']=="SANTOS"].pivot( index='Date', columns='Beach', values='Enterococcus').fillna(0)
#df_pivot
#df_pivot.plot(figsize=(30,10))
df2_limpo = df2.sort_values(by=['Date'])
#remover a praia do Leste, da cidade de iguape, pois esta praia sumiu por erosão em 2012
#remover a Lagoa Prumirim, da cidade de Ubatuba, pois esta praia possui somente 3 medições
df2_limpo = df2_limpo.loc[df2_limpo['Beach']!='DO LESTE'].loc[df2_limpo['Beach']!='LAGOA PRUMIRIM']
# prepare expected column names
#df2_pereque = df2_limpo.loc[df2_limpo['City']=="GUARUJÁ"].loc[df2_limpo['Beach']=="PEREQUÊ"][['Date','Enterococcus']]
cidade="UBATUBA"
praia="GRANDE"
test_size=5

df2_beach = df2_limpo.loc[df2_limpo['City']==cidade].loc[df2_limpo['Beach']==praia][['Date','Enterococcus']]
df2_beach.columns = ['ds', 'y']
df2_beach



pre_beach_plot =df2_limpo.loc[df2_limpo['City']==cidade].loc[df2_limpo['Beach']==praia][['Date','Beach','Enterococcus']]
beach_plot=pre_beach_plot.pivot( index='Date', columns='Beach', values='Enterococcus')
beach_plot
beach_plot.plot(figsize=(30,5), title=praia)
x_coordinates= [pre_beach_plot['Date'].min(), pre_beach_plot['Date'].max()]
y_coordinates100= [100,100]
plt.plot(x_coordinates, y_coordinates100,'r--')
y_coordinates400= [400,400]
plt.plot(x_coordinates, y_coordinates400,'k--')
whisker_upper = boxplot_stats(df2_beach['y']).pop(0)['whishi']
print(whisker_upper)
y_coordinates_whisker_upper = [whisker_upper,whisker_upper]
plt.plot(x_coordinates, y_coordinates_whisker_upper,'b--')
# define the model
model = Prophet()
# fit the model
model.fit(df2_beach)

# define the period for which we want a prediction
future = list()
for i in range(1, 13):
	date = '2020-%02d' % i
	future.append([date])
future = pd.DataFrame(future)
future.columns = ['ds']
future['ds']= pd.to_datetime(future['ds'])
future
# use the model to make a forecast
forecast = model.predict(future)
# summarize the forecast
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
# plot forecast
model.plot(forecast)
x_coordinates= [df2_beach['ds'].min(), forecast['ds'].max()]
y_coordinates100= [100,100]
plt.plot(x_coordinates, y_coordinates100,'r--')
y_coordinates400= [400,400]
plt.plot(x_coordinates, y_coordinates400,'k--')
whisker_upper = boxplot_stats(df2_beach['y']).pop(0)['whishi']
print(whisker_upper)
y_coordinates_whisker_upper = [whisker_upper,whisker_upper]
plt.plot(x_coordinates, y_coordinates_whisker_upper,'b--')
plt.show()

# create test dataset, remove last test_siz measurements

train = df2_beach.loc[df2_beach.index[1:len(df2_beach.index)-test_size]]
#print(train.tail())
train
# define the model
model2 = Prophet()
# fit the model
model2.fit(train)
future2=df2_beach.loc[df2_beach.index[-test_size:]]['ds']
future2 = pd.DataFrame(future2)
future2.columns = ['ds']
future2['ds'] = pd.to_datetime(future2['ds'])
future2
forecast2 = model.predict(future2)
forecast2
# calculate MAE between expected and predicted values 
y_true = df2_beach['y'][-test_size:].values
y_pred = forecast2['yhat'].values
mae = mean_absolute_error(y_true, y_pred)
print('MAE: %.3f' % mae)
# plot expected vs actual
plt.plot(y_true, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()
# summarize the forecast
print(forecast2[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
# plot forecast
model.plot(forecast2)
x_coordinates= [df2_beach['ds'].min(), forecast['ds'].max()]
y_coordinates100= [100,100]
plt.plot(x_coordinates, y_coordinates100,'r--')
y_coordinates400= [400,400]
plt.plot(x_coordinates, y_coordinates400,'k--')
whisker_upper = boxplot_stats(train['y']).pop(0)['whishi']
print(whisker_upper)
y_coordinates_whisker_upper = [whisker_upper,whisker_upper]
plt.plot(x_coordinates, y_coordinates_whisker_upper,'b--')
plt.show()