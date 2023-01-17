import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # organize plots
from functools import reduce
plt.rcParams['figure.figsize']=(15,8)
#A: tamaño comodo para las figuras

#FROM: https://pythonprogramminglanguage.com/web-scraping-with-pandas-and-beautifulsoup/
import requests
from bs4 import BeautifulSoup

def get_data_bcra(fname,url,params):
    "funcion para queries a la pagina del bcra Y pasar el resultado a un dataframe"
    #res = requests.post(url, data= qparams)
    with open('../input/'+fname, "r", encoding='ISO8859-1') as file_html:
        html = file_html.read()  

    soup = BeautifulSoup(html,'lxml')
    table = soup.find_all('table')[0] 
    df = pd.read_html(str(table),skiprows=1,decimal=',',thousands='.')[0] #.iloc[1:]
    #A: lei la tabla html, tire la primera fila (los titulos), lo guarde en el data frame
    #SEE: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_html.html
    df['FECHA']= pd.to_datetime(df['FECHA'], format='%d/%m/%Y')
    #A: converti la fecha en un datetime
    df.set_index('FECHA',inplace=True)
    #A: la converti en indice para plotear
    df.sort_index(inplace=True)
    #A: la ordene por fecha
    return df

bcra_url = 'http://www.bcra.gob.ar/PublicacionesEstadisticas/Principales_variables_datos.asp'


bcra_cer_qparams = {'desde': '17/12/2015', 'hasta': '27/03/2018','fecha':'Fecha_Ref','descri':19,'campo':'Lebac','primeravez':1,'alerta':5}
bcra_cer= get_data_bcra('BCRA_cer.html',bcra_url,bcra_cer_qparams)
bcra_cer.rename(columns={'VALOR':'cer'},inplace=True)
bcra_cer.head()

bcra_tcref_qparams = {'desde': '17/12/2015', 'hasta': '27/03/2018','fecha':'Fecha_Ref','descri':19,'campo':'Lebac','primeravez':1,'alerta':5}
bcra_tcref= get_data_bcra('BCRA_tc_ref.html',bcra_url,bcra_cer_qparams)
bcra_tcref.rename(columns={'VALOR':'tcref'},inplace=True)
bcra_tcref.head()
bcra_rlebac_qparams = {'desde': '17/12/2015', 'hasta': '27/03/2018','fecha':'Fecha_Ref','descri':19,'campo':'Lebac','primeravez':1,'alerta':5}
bcra_rlebac= get_data_bcra('BCRA_lebac_tasas_primaria.html',bcra_url,bcra_rlebac_qparams)
bcra_rlebac.rename(columns={'VALOR':'rlebac'},inplace=True)
bcra_rlebac.head()
ax= bcra_cer.plot(color='red',grid=True, title= 'Dolar de referencia vs. CER')
bcra_tcref.plot(ax= ax, color='green',grid=True,secondary_y=True)
#bcra_rlebac.plot(ax=ax, color='orange',grid=True, secondary_y=True)
# t0= pd.Timestamp('2015-01-01') #A: Dolar "barato" de CFK
t0= pd.Timestamp('2015-12-18') #A: Dolar "sincerado" de MM
t0_cer= bcra_cer.at[t0,'cer']
t0_tcref= bcra_tcref.at[t0,'tcref']
t0_cerAtcref= t0_tcref / t0_cer
print(f'La relacion al {t0} tcref/cer= {t0_cerAtcref}= {t0_tcref} / {t0_cer}')

ax= bcra_cer.apply(lambda x: x*t0_cerAtcref).plot(color='red',grid=True, title=  f'Dolar de referencia vs. CER AJUSTADO a {t0}')
bcra_tcref.plot(ax= ax, color='green',grid=True)

ax= bcra_cer.apply(lambda x: x*t0_cerAtcref).plot(color='red',grid=True, title=  f'Dolar de referencia vs. CER AJUSTADO a {t0}')
bcra_tcref.plot(ax= ax, color='green',grid=True)
ax.set_xlim(xmin=t0)
ax.set_ylim(12)
bcra_tcref_MM= bcra_tcref[t0:]
bcra_tcref_d1= bcra_tcref_MM.diff() #A: la diferencia entre un dia y el otro
ax= bcra_tcref_d1.plot(title='diferencia diaria en la cotizacion del dolar')
ax.set_ylim(-1)
bcra_tcref_d1_pct= bcra_tcref_d1 / bcra_tcref

bcra_tcref_d1_bajaMayor1pct= bcra_tcref_d1_pct[bcra_tcref_d1_pct['tcref']<-.01]
bcra_tcref_d1_bajaMayor1pct.plot(kind='hist', bins=10, title="Veces que podria haber ganado comprando mas de 1% mas barato al dia siguiente")

cnt= bcra_tcref_d1_bajaMayor1pct.shape[0]
print(f'Podria haber ganado mas de un 1% vendiendo CARO y recomprando barato de un dia para el otro {cnt} veces')

tasa_equiv_1pct= ((1.01 ** cnt) - 1) * 100
print(f'La tasa equivalente, considerando que fuera solo el 10% cada vez (es mas) es de {tasa_equiv_1pct}')

def rate_compose(series):
    return reduce(lambda x, y: x * (1-y), series,1)

tasa_equiv= (bcra_tcref_d1_bajaMayor1pct['tcref'].agg(rate_compose)-1)*100
print(f'La tasa equivalente por las veces que vendia y recompraba al dia siguiente 1% menos o mas barato hubiera sido {tasa_equiv}')
#Buscando extremos donde comprar y vender hubiera dado mas diferencia
from scipy.signal import argrelextrema #U: para encontrar maximos y minimos aunque no sean intradiarios
from scipy.signal import find_peaks_cwt #U: con smoothing y mas parametros
#SEE: https://stackoverflow.com/questions/4624970/finding-local-maxima-minima-with-numpy-in-a-1d-numpy-array
peak_order= 10
max_idx= argrelextrema(bcra_tcref_MM['tcref'].values,np.greater,order=peak_order) #A: encuentra TODOS los piquitos arriba
min_idx= argrelextrema(bcra_tcref_MM['tcref'].values,np.less,order=peak_order) #A: encuentra TODOS los piquitos abajo
#max_idx= find_peaks_cwt(bcra_tcref_MM['tcref'].values,np.arange(1,30))
ax= bcra_tcref_MM.plot(color='green', title='Buscando extremos donde comprar y vender hubiera dado mas diferencia')
bcra_tcref_MM.iloc[max_idx].plot(ax= ax, color='orange',linewidth=0,marker="x",markersize=10)
bcra_tcref_MM.iloc[min_idx].plot(ax= ax, color='blue',linewidth=0,marker="x",markersize=10)

#Funciona mas o menos... parece que la posta es usar Prophet
#SEE: https://www.kaggle.com/attollos/time-series-forecast-example-with-prophet
from fbprophet import Prophet

bcra_tcref_MM_train= bcra_tcref_MM.copy()
#SEE: http://pandas.pydata.org/pandas-docs/stable/indexing.html#indexing-view-versus-copy
bcra_tcref_MM_train.reset_index(level=0, inplace=True)
bcra_tcref_MM_train.rename(columns={'FECHA':'ds','tcref':'y'},inplace=True)

#A: tengo los nombres de cols e index que quiere Prophet

bcra_tcref_MM_prophet= Prophet()
bcra_tcref_MM_prophet.fit(bcra_tcref_MM_train)
#A: ajuste el modelo con los datos que tenia
future = bcra_tcref_MM_prophet.make_future_dataframe(periods=2)
#A: prepare el futuro
bcra_tcref_MM_prophet_forecast = bcra_tcref_MM_prophet.predict(future)
bcra_tcref_MM_prophet_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
#A: tengo un pronostico segun las tendencias que detecto prophet
bcra_tcref_MM_prophet.plot(bcra_tcref_MM_prophet_forecast) 
plt.title('Ajuste de Prophet Y PREDICCION (notar meses futuros)') #A: el ; es para que no aparezca el grafico dos veces, una por el comando y otra como resultado
for changepoint in bcra_tcref_MM_prophet.changepoints:
    plt.axvline(changepoint,ls='--', lw=1)
print('Componentes del Ajuste de Prophet Y PREDICCION (notar meses futuros)')
bcra_tcref_MM_prophet.plot_components(bcra_tcref_MM_prophet_forecast);

 #A: el ; es para que no aparezca el grafico dos veces, una por el comando y otra como resultado
bcra_tcref_MM_prophet= Prophet(daily_seasonality=False, yearly_seasonality=False)
bcra_tcref_MM_prophet.fit(bcra_tcref_MM_train)
#A: ajuste el modelo con los datos que tenia
future = bcra_tcref_MM_prophet.make_future_dataframe(periods=2)
#A: prepare el futuro
bcra_tcref_MM_prophet_forecast = bcra_tcref_MM_prophet.predict(future)
bcra_tcref_MM_prophet_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
#A: tengo un pronostico segun las tendencias que detecto prophet
bcra_tcref_MM_prophet.plot(bcra_tcref_MM_prophet_forecast) 
plt.title('Ajuste de Prophet Y PREDICCION (notar meses futuros)') #A: el ; es para que no aparezca el grafico dos veces, una por el comando y otra como resultado
for changepoint in bcra_tcref_MM_prophet.changepoints:
    plt.axvline(changepoint,ls='--', lw=1)
print('Componentes del Ajuste de Prophet Y PREDICCION (notar meses futuros)')
bcra_tcref_MM_prophet.plot_components(bcra_tcref_MM_prophet_forecast);
