# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))  
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import numpy as np
import itertools
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import plotly.express as px
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, acf, pacf,arma_order_select_ic
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_model import ARIMA
import warnings
warnings.simplefilter('ignore')
data=pd.read_csv(r"/kaggle/input/covid19-in-italy/covid19_italy_region.csv")
test=pd.read_csv(r"/kaggle/input/covid19-in-italy/covid19_italy_province.csv")
data.sample(6)
test.sample(6)
data.corr()
df = data.fillna('NA').groupby(['Country','RegionName','Date'])['CurrentPositiveCases'].sum() \
                          .groupby(['Country','RegionName']).max().sort_values() \
                          .groupby(['Country']).sum().sort_values(ascending = False)

top10 = pd.DataFrame(df).head(10)
top10
fig = px.bar(top10, x=top10.index, y='CurrentPositiveCases', labels={'x':'RegionName'},
             color="CurrentPositiveCases", color_continuous_scale=px.colors.sequential.Brwnyl)
fig.update_layout(title_text='Confirmed COVID-19 cases by country')
fig.show()
df_by_date = pd.DataFrame(data.fillna('NA').groupby(['Country','Date'])['CurrentPositiveCases'].sum().sort_values().reset_index())

fig = px.bar(df_by_date.loc[(df_by_date['Country'] == 'ITA') &(df_by_date.Date >= '2020-03-01')].sort_values('CurrentPositiveCases',ascending = False), 
             x='Date', y='CurrentPositiveCases', color="CurrentPositiveCases", color_continuous_scale=px.colors.sequential.BuGn)
fig.update_layout(title_text='İtalya da günde onaylanmış COVID-19 vakaları')
fig.show()
df=data.groupby(['Date','Country']).agg('sum').reset_index()
df.tail(5)
def pltCountry_cases(TotalPositiveCases,*argv):
    f, ax=plt.subplots(figsize=(16,5))
    labels=argv
    for a in argv: 
        country=df.loc[(df['Country']==a)]
        plt.plot(country['Date'],country['TotalPositiveCases'],linewidth=3)
        plt.xticks(rotation=40)
        plt.legend(labels)
        ax.set(title='Dava sayısının gelişimi' )
def pltCountry_fatalities(Deaths,*argv):
    f, ax=plt.subplots(figsize=(16,5))
    labels=argv
    for a in argv: 
        country=df.loc[(df['Country']==a)]
        plt.plot(country['Date'],country['Deaths'],linewidth=3)
        plt.xticks(rotation=40)
        plt.legend(labels)
        ax.set(title='Ölüm sayısının gelişimi' )
pltCountry_cases('TotalPositiveCases','ITA')
pltCountry_fatalities('Deaths','ITA')
TotalPositiveCases_Lazio = data[data['RegionName']=='Lazio'].groupby(['Date']).agg({'TotalPositiveCases':['sum']})
Deaths_Lazio = data[data['RegionName']=='Lazio'].groupby(['Date']).agg({'Deaths':['sum']})
total_Lazio = TotalPositiveCases_Lazio.join(Deaths_Lazio)

TotalPositiveCases_Veneto = data[data['RegionName']=='Veneto'].groupby(['Date']).agg({'TotalPositiveCases':['sum']})
Deaths_Veneto = data[data['RegionName']=='Veneto'].groupby(['Date']).agg({'Deaths':['sum']})
total_Veneto = TotalPositiveCases_Veneto.join(Deaths_Veneto)

TotalPositiveCases_Toscana = data[data['RegionName']=='Toscana'].groupby(['Date']).agg({'TotalPositiveCases':['sum']})
Deaths_Toscana = data[data['RegionName']=='Toscana'].groupby(['Date']).agg({'Deaths':['sum']})
total_Toscana = TotalPositiveCases_Toscana.join(Deaths_Toscana)

TotalPositiveCases_Lambordia = data[data['RegionName']=='Lambordia'].groupby(['Date']).agg({'TotalPositiveCases':['sum']})
Deaths_Lambordia = data[data['RegionName']=='Lambordia'].groupby(['Date']).agg({'Deaths':['sum']})
total_Lambordia = TotalPositiveCases_Lambordia.join(Deaths_Lambordia)

TotalPositiveCases_Marche = data[data['RegionName']=='Marche'].groupby(['Date']).agg({'TotalPositiveCases':['sum']})
Deaths_Marche = data[data['RegionName']=='Marche'].groupby(['Date']).agg({'Deaths':['sum']})
total_Marche = TotalPositiveCases_Marche.join(Deaths_Marche)
plt.figure(figsize=(24,18))

plt.subplot(3, 3, 1)
total_Lazio.plot(ax=plt.gca(), title='Lazio')
plt.ylabel("Confirmed infection cases", size=13)

plt.subplot(3, 3, 2)
total_Veneto.plot(ax=plt.gca(), title='Veneto')

plt.subplot(3, 3, 3)
total_Toscana.plot(ax=plt.gca(), title='Toscana')

plt.subplot(3, 3, 4)
total_Lambordia.plot(ax=plt.gca(), title='Lambordia')
plt.ylabel("Confirmed infection cases", size=13)

plt.subplot(3, 3, 5)
total_Marche.plot(ax=plt.gca(), title='Marche')

sns.set(palette = 'Set1',style='darkgrid')
#Function for making a time serie on a designated country and plotting the rolled mean and standard 
def roll(country,case='TotalPositiveCases'):
    ts=df.loc[(df['Country']==country)]  
    ts=ts[['Date',case]]
    ts=ts.set_index('Date')
    ts.astype('int64')
    a=len(ts.loc[(ts['TotalPositiveCases']>=10)])
    ts=ts[-a:]
    return (ts.rolling(window=4,center=False).mean().dropna())


def rollPlot(country, case='TotalPositiveCases'):
    ts=df.loc[(df['Country']==country)]  
    ts=ts[['Date',case]]
    ts=ts.set_index('Date')
    ts.astype('int64')
    a=len(ts.loc[(ts['TotalPositiveCases']>=10)])
    ts=ts[-a:]
    plt.figure(figsize=(16,6))
    plt.plot(ts.rolling(window=7,center=False).mean().dropna(),label='Rolling Mean')
    plt.plot(ts[case])
    plt.plot(ts.rolling(window=7,center=False).std(),label='Rolling std')
    plt.legend()
    plt.title('Haddeleme ortalaması ve standart ile% s cinsinden vaka dağılımı' %country)
    plt.xticks([])
tsC1=roll('ITA')
rollPlot('ITA')
fig=sm.tsa.seasonal_decompose(tsC1.values,freq=7).plot()
#Function to check the stationarity of the time serie using Dickey fuller test
def stationarity(ts):
    print('Results of Dickey-Fuller Test:')
    test = adfuller(ts, autolag='AIC')
    results = pd.Series(test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    for i,val in test[4].items():
        results['Critical Value (%s)'%i] = val
    print (results)

#For China
tsC=tsC1['TotalPositiveCases'].values
stationarity(tsC)
def corr(ts):
    plot_acf(ts,lags=12,title="ACF")
    plot_pacf(ts,lags=12,title="PACF")
    

#For China
corr(tsC1)
#test['Date'] = pd.to_datetime(test['Date'])
#train['Date'] = pd.to_datetime(train['Date'])
data = data.set_index(['Date'])
test = test.set_index(['Date'])
def create_features(df,label=None):
    """
    Creates time series features from datetime index.
    """
    df = df.copy()
    df['Date'] = df.index
    df['hour'] = df['Date'].dt.hour
    df['dayofweek'] = df['Date'].dt.dayofweek
    df['quarter'] = df['Date'].dt.quarter
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['dayofyear'] = df['Date'].dt.dayofyear
    df['dayofmonth'] = df['Date'].dt.day
    df['weekofyear'] = df['Date'].dt.weekofyear
    
    X = df[['hour','dayofweek','quarter','month','year',
           'dayofyear','dayofmonth','weekofyear']]
   
    return X
#Mean absolute percentage error
def mape(y1, y_pred): 
    y1, y_pred = np.array(y1), np.array(y_pred)
    return np.mean(np.abs((y1 - y_pred) / y1)) * 100

def split(ts):
    #splitting 85%/15% because of little amount of data
    size = int(len(ts) * 0.85)
    data= ts[:size]
    test = ts[size:]
    return(data,test)


#Arima modeling for ts
def arima(ts,test):
    p=d=q=range(0,6)
    a=99999
    pdq=list(itertools.product(p,d,q))
    
    #Determining the best parameters
    for var in pdq:
        try:
            model = ARIMA(ts, order=var)
            result = model.fit()

            if (result.aic<=a) :
                a=result.aic
                param=var
        except:
            continue
            
    #Modeling
    model = ARIMA(ts, order=param)
    result = model.fit()
    result.plot_predict(start=int(len(ts) * 0.7), end=int(len(ts) * 1.2))
    pred=result.forecast(steps=len(test))[0]
    #Plotting results
    f,ax=plt.subplots()
    plt.plot(pred,c='green', label= 'predictions')
    plt.plot(test, c='red',label='real values')
    plt.legend()
    plt.title('True vs predicted values')
    #Printing the error metrics
    print(result.summary())        
    
    print('\nMean absolute percentage error: %f'%mape(test,pred))
    return (pred)



data,test=split(tsC)
pred=arima(data,test)
train=pd.read_csv("/kaggle/input/covid19-in-italy/covid19_italy_region.csv")
train.head()

feature_cols = train.columns[:-1]
corr_values = train[feature_cols].corr()

tril_index = np.tril_indices_from(corr_values)

for coord in zip(*tril_index):
    corr_values.iloc[coord[0], coord[1]] = np.NaN

corr_values = (corr_values.stack().to_frame().reset_index().rename(columns={'level_0':'feature1','level_1':'feature2',0:'correlation'}))

corr_values['abs_correlation'] = corr_values.correlation.abs()
sns.set_context('talk')
sns.set_style('white')
sns.set_palette('dark')

ax = corr_values.abs_correlation.hist(bins=50)

ax.set(xlabel='Mutlak Korelasyon', ylabel='Sıklık');
y = (train['Country'] == 'red').astype(int)
fields = list(train.columns[:-1])
correlations = train[fields].corrwith(y)
correlations.sort_values(inplace=True)
correlations
sns.set_context('talk')
sns.set_palette('dark')
sns.set_style('white')
sns.pairplot(train, hue='Country')
plt.figure(figsize=(23,6))

plt.plot(train.Date,train.IntensiveCarePatients,color="red")

plt.title("İtalya Günlere Göre Yoğun Bakım Hastaları")

plt.xlabel("Tarih")

plt.ylabel("Yoğun Bakım Hastaları")

plt.show()
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
X = train[train.columns[:-1]]
y = train.NewPositiveCases

GNB = GaussianNB()
cv_N = 4
scores = cross_val_score(GNB, X, y, n_jobs=cv_N, cv=cv_N)
print(scores)
np.mean(scores)
ax = plt.axes()

ax.scatter(train.NewPositiveCases, train.TestsPerformed)

ax.set(xlabel='Yeni Olumlu vakalar (gün)',
       ylabel='Yapılan Testler (adet)',
       title='Yeni Olumlu vakalar vs Yapılan Testler');
plt.axes().set(xlabel='Değerler',
       ylabel='Sıklık',
       title='Toplam Hastanede Yatan Hasta');
train.TotalHospitalizedPatients.plot(kind = 'hist',bins = 80,figsize = (9,5))
plt.show()
plt.boxplot ([train.TestsPerformed,train.Deaths,train.HospitalizedPatients,train.Recovered])
train.plot(subplots = True)
train.TotalHospitalizedPatients.plot(kind = 'hist',figsize = (6,95))
plt.show()
train[['RegionName','TestsPerformed','Date']].describe()
plt.style.use(['tableau-colorblind10'])

df_Country = train.groupby(['RegionName'])[["Deaths","TotalPositiveCases"]].max().nlargest(8,'Deaths')

df_Country['Fatality_Percentage'] = df_Country['Deaths']/ df_Country['TotalPositiveCases']
df_Country = df_Country.reset_index()
df_Country.sort_values('Fatality_Percentage',inplace=True)
figure, axes = plt.subplots(1, 2,figsize=(12,4))
df_Country.plot(ax= axes[0],x = 'RegionName', y = ["Deaths","TotalPositiveCases"],kind='bar', title = 'Ölüm ve Vaka sayısı')
df_Country.plot(ax= axes[1],x = 'RegionName', y = ["Fatality_Percentage"],kind='bar', title = 'Vakaların ölümleri')
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()
train.style.background_gradient(cmap='Reds')
train.info()
train.plot(kind='scatter', x='HomeConfinement', y='TestsPerformed',alpha = 0.5,color = 'red')
plt.xlabel('Evde Karantinada olanlar')
plt.ylabel('Test Sayıları')
plt.title('Test Sayılarına Göre Evde Karantinada olanlar')  
train.Recovered.plot(kind = 'line', color = 'g',label = 'Recovered',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
train.TotalHospitalizedPatients.plot(color = 'r',label = 'TotalHospitalizedPatients',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     
plt.xlabel('Günler')              
plt.ylabel('Kişi Sayısı')
plt.title('Grafiğin Başlığı')            
plt.show()