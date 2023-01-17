import os
os.listdir("../input")
import pandas as pd
datos= pd.read_csv('../input/Consumo_cerveja.csv',index_col='Data')
datos.info()
datos=datos.dropna()
import pandas as pd
def convertir(x):
    y=x.replace(',','.')
    return pd.to_numeric(y)
datos[datos.columns[1]].head()
datos[datos.columns[1]].apply(convertir).head()
var=datos.columns[:4]
for col in var:
    datos[col]=datos[col].apply(convertir)
datos.info()
import calendar
fecha=pd.to_datetime('2019-01-15',format='%Y-%m-%d')
calendar.day_name[fecha.weekday()]
fecha2=pd.to_datetime(datos.index,format='%Y-%m-%d')
#datos['NomDia']=fecha2.weekday_name.values
datos['numDia']=fecha2.dayofweek
days = {0:'1.Lun',1:'2.Mar',2:'3.Mie',3:'4.Jue',4:'5.Vie',5:'6.Sab',6:'7.Dom'}
datos['NomDia'] =datos['numDia'].apply(lambda x: days[x])
datos['NomDia']=pd.Categorical(datos['NomDia'])
datos.info()
series=datos['Consumo de cerveja (litros)']
import matplotlib.pyplot as plt
plt.figure(figsize=(14,6))
plt.plot(series)
plt.title('Consumo de cerveja (litros)')
plt.show()
import statsmodels.api as sm
import matplotlib.pyplot as plt
res = sm.tsa.seasonal_decompose(series,freq=60)
fig = res.plot()
fig.set_figheight(8)
fig.set_figwidth(15)
plt.show()
window=3
y_ajust=series.rolling(window=window).mean()
pd.DataFrame({'Y':series.head(6).values,'Y_ajust':y_ajust.head(6)})
import matplotlib.pyplot as plt 
window=60
series=datos['Consumo de cerveja (litros)']
rolling_mean = series.rolling(window=window).mean()
plt.figure(figsize=(18,4))
plt.plot(series.values, "c", label = "Actual")
plt.title("Moving average\n window size = {}".format(window))
plt.plot(rolling_mean,color='red',label="MA")
plt.legend(loc="best")
plt.show()
def SuavizacionExponencialSimple(series, alpha):
    result = [series[0]]
    for n in range(1, len(series)):
        result.append(alpha * series[n] + (1 - alpha) * result[n-1])
    return result
def plotSuavizacionExponencialSimple(series, alphas):
    plt.figure(figsize=(18,4))
    plt.plot(series.values, "c", label = "Actual")
    for alpha in alphas:
        plt.plot(SuavizacionExponencialSimple(series, alpha), label="Alpha {}".format(alpha),color="red")    
    plt.legend(loc="best")
    plt.title("Suavizacion Exponencial Simple",fontsize=16)
plotSuavizacionExponencialSimple(series,[0.3])
def SuavizacionHolt(series, alpha, beta):
    result = [series[0]]
    for n in range(1, len(series)+1):
        if n == 1:
            level, trend = series[0], series[1] - series[0]
        if n >= len(series):
            value = result[-1]
        else:
            value = series[n]
        last_level, level = level, alpha*value + (1-alpha)*(level+trend)
        trend = beta*(level-last_level) + (1-beta)*trend
        result.append(level+trend)
    return result
def plotSuavizacionHolt(series, alphas, betas):
    plt.figure(figsize=(18,4))
    plt.plot(series.values, "c", label = "Actual")
    for alpha in alphas:
        for beta in betas:
            plt.plot(SuavizacionHolt(series, alpha, beta), label="Alpha {}, beta {}".format(alpha, beta),color="red")
    plt.title("Suavización mediante el método de Holt",fontsize=16)
    plt.legend(loc="best")
    plt.show()
plotSuavizacionHolt(series,alphas=[0.2],betas=[0.1])
import matplotlib.pyplot as plt

series=datos['Consumo de cerveja (litros)']
plt.figure(figsize=(20,5))
plt.plot(series)
plt.title('Consumo de cerveja (litros)')
plt.show()
import seaborn as sns 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
fig,axes = plt.subplots(1,3,figsize=(18,6))
#Gráfica01
sns.distplot(datos['Consumo de cerveja (litros)'],ax=axes[0],hist_kws=dict(edgecolor="b", linewidth=2))
axes[0].set_title('Histograma del Consumo', fontsize=16)
axes[0].set_xlabel('Consumo de cerveja (litros)',fontsize=16)
#Gráfica02
color=sns.color_palette('husl',7)
for i in range(7):
    sns.kdeplot(datos['Consumo de cerveja (litros)'][(datos["numDia"] ==i)],shade=True,color=color[i],label=days[i],ax=axes[1])
axes[1].set_title('Consumo por nombre de día', fontsize=16)
axes[1].set_xlabel('Consumo de cerveja (litros)',fontsize=16)
#Gráfica03
sns.boxplot(y='NomDia',x='Consumo de cerveja (litros)',data=datos,ax=axes[2],palette=color)
axes[2].set_title('Consumo por nombre de día', fontsize=16)
axes[2].set_ylabel('',fontsize=18)
axes[2].set_xlabel('Consumo de cerveja (litros)',fontsize=16)
plt.show()
from scipy.stats import skew
from scipy.stats import kurtosis
import numpy as np
Resumen2=datos.groupby(['NomDia'])['Consumo de cerveja (litros)'].agg([np.mean,np.median,np.std,skew,kurtosis])
Resumen2
def inversa_boxcox(y, lambda_):
    return np.exp(y) if lambda_ == 0 else np.exp(np.log(lambda_ * y + 1) / lambda_)
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split

train,test = train_test_split(series, shuffle=False)
train_data,fitted_lambda = stats.boxcox(train)

test_data = stats.boxcox(test, fitted_lambda)
test_datat1= inversa_boxcox(test_data,fitted_lambda)
fig,ax = plt.subplots(1,2,figsize=(18,6))
ax[0].plot(test)
ax[1].plot(test_data)
plt.show()
import pandas as pd
pd.DataFrame({'Desviación':['Datos Originales','Transformados'],
              'Valor':[np.std(test),np.std(test_data)]})
fig,ax = plt.subplots(1,2,figsize=(16,6))
sns.distplot(test, ax=ax[0],hist_kws=dict(edgecolor="b", linewidth=2))
sns.distplot(test_data, ax=ax[1],hist_kws=dict(edgecolor="b", linewidth=2))
plt.show()
import pandas as pd
rolmean = series.rolling(window=60).mean()
rolstd = series.rolling(window=60).std()

fig,ax = plt.subplots(figsize=(16,6))
orig = plt.plot(series, color='blue',label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std  = plt.plot(rolstd, color='black', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show()
from statsmodels.tsa.stattools import adfuller
test1 = adfuller(series, autolag='AIC')
test1
dfoutput = pd.Series(test1[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
print (dfoutput)
import numpy as np
np.random.seed(1)
n_samples = 1000
x1=w=np.random.normal(size=n_samples)
import seaborn as sns 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

fig,ax=plt.subplots(figsize=(10,6))
sns.distplot(x1,hist_kws=dict(edgecolor="b", linewidth=2))
plt.show()
for t in range(n_samples):
    x1[t] = 0.69*x1[t-1]+ w[t]
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

fig,ax = plt.subplots(1,2,figsize=(20,6))
fig = sm.graphics.tsa.plot_acf(x1, lags=60,ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(x1,lags=60,ax=ax[1])
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

x2=w=np.random.normal(size=n_samples)
for t in range(n_samples):
    x2[t] = 0.39*x2[t-1]+0.49*x2[t-2]+w[t]

fig,ax = plt.subplots(1,2,figsize=(20,6))
fig = sm.graphics.tsa.plot_acf(x2, lags=60,ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(x2,lags=60,ax=ax[1])
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

x3=w=np.random.normal(size=n_samples)
for t in range(n_samples):
    x3[t] = 0.39*x3[t-1]+0.49*x3[t-7]+w[t]

fig,ax = plt.subplots(1,2,figsize=(20,6))
fig = sm.graphics.tsa.plot_acf(x3, lags=60,ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(x3,lags=60,ax=ax[1])
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

x4=w=np.random.normal(size=n_samples)
for t in range(n_samples):
    x4[t] = 0.49*x4[t-1]+0.2*x4[t-7]+0.2*x4[t-14]+w[t]

fig,ax = plt.subplots(1,2,figsize=(20,6))
fig = sm.graphics.tsa.plot_acf(x4, lags=60,ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(x4,lags=60,ax=ax[1])
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

x5=w=np.random.normal(size=n_samples)
for t in range(n_samples):
    x5[t] = 0.4*x5[t-1]+0.15*x5[t-7]+0.15*x5[t-14]+0.15*x5[t-21]+w[t]

fig,ax = plt.subplots(1,2,figsize=(20,6))
fig = sm.graphics.tsa.plot_acf(x5, lags=60,ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(x5,lags=60,ax=ax[1])
import statsmodels.api as sm
import matplotlib.pyplot as plt

fig,ax = plt.subplots(2,1,figsize=(20,8))
fig = sm.graphics.tsa.plot_acf(series, lags=120, ax=ax[0])
fig = sm.graphics.tsa.plot_pacf(series, lags=120, ax=ax[1])
plt.show()
import statsmodels.tsa.api as smt
modelo1 = smt.AR(x1).fit(maxlag=30, ic='aic', trend='nc')
print(modelo1.params)
import statsmodels.tsa.api as smt
modelo2 = smt.AR(x2).fit(maxlag=30, ic='aic', trend='nc')
print(modelo2.params[0],modelo2.params[1])
max_lag = 30
modelo1 = smt.ARMA(x1, order=(1,0)).fit(maxlag=max_lag, method='mle', trend='nc')
modelo1.summary()
max_lag = 30
modelo2 = smt.ARMA(x2, order=(2,0)).fit(maxlag=max_lag, method='mle', trend='nc')
modelo2.summary()
import pandas as pd
datos1= pd.read_csv('../input/DatosEjemploSeries.csv',sep=";")#index_col='Data'
fechas= pd.read_csv('../input/fechas.csv',sep=";")#index_col='Data'
datos1=datos1.drop('Date',axis=1).rename(pd.to_datetime(datos1['Date']),axis='index')
datos1.head()
datos1.info()
fechas.head()