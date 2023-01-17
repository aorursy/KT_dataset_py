from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from scipy.optimize import curve_fit

import seaborn as sns

sns.set()

import warnings

warnings.filterwarnings('ignore')
deaths_global_df = pd.read_csv('../input/covid-19/time_series_covid19_deaths_global.csv')
deaths_global_df.head(5)
deaths_global_df = deaths_global_df.groupby(['Country/Region']).sum().reset_index()

deaths_global_df = deaths_global_df.drop(['Lat','Long'], axis=1)

deaths_global_df = deaths_global_df.transpose()

deaths_global_df.columns = deaths_global_df.iloc[0]

deaths_global_df = deaths_global_df[1:]

deaths_global_df = deaths_global_df[['US','United Kingdom', 'Italy', 'France',  'Spain', 'Brazil','Belgium','Germany', 'China']]

deaths_global_df['China'] = np.where(deaths_global_df['China'] > 4630, 3326,deaths_global_df['China'] )

deaths_global_df.index = range(len(deaths_global_df.index))

deaths_global_df
deaths_global_zero = deaths_global_df.apply(lambda col: col.drop_duplicates().reset_index(drop=True))

deaths_global_zero.head(10)
# same plotting code as above!

plt.figure(figsize=(15,10))

plt.plot(deaths_global_df.index, deaths_global_df)

plt.legend(deaths_global_df.columns, ncol=2, loc='upper left');
# same plotting code as above!

plt.figure(figsize=(15,10))

plt.plot(deaths_global_zero.index, deaths_global_zero)

plt.legend(deaths_global_zero.columns, ncol=2, loc='upper left');
deaths_global_difference = deaths_global_zero.diff().dropna(how='all')

deaths_global_difference.head(5)
deaths_global_difference.plot(subplots=True, kind='bar' , figsize=(20,30))
plt.figure(figsize=(15,10))

sns.set(style="whitegrid")

ax = sns.barplot(x = deaths_global_difference.sum().index, y = deaths_global_difference.sum().sort_values(ascending=False))
sns.pairplot(deaths_global_zero.dropna(), size=3);
## FUNÇÃO SIGMÓIDE DE Gompertz

a = b = g =1

def sigGompertz (x, a, b, g):

        return a*np.exp(-b*np.exp(-g*x))

#CÁLCULO DA DERIVADA DA SIGMÓIDE

def devGompertz(x, a,b,g):

    return np.array(a*b*g*np.exp(-b*np.exp(-g*x))*np.exp(-g*x))



x=np.arange(-10,10,0.1)

sigGompertz (x, a, b, g)

plt.figure(figsize=(20,10))

plt.plot(x, sigGompertz (x, a, b, g), lw=4, label=' SIGMOID FUNCTION GOMPERTZ')

plt.plot(x, devGompertz(x, a,b,g), lw=4, label=' DERIVED FUNCTION GOMPERTZ')

plt.axvline(0, color='b', ls ='-.')

plt.axhline(0.5, color='b', ls ='-.')

plt.legend(loc='upper center', fontsize=15)

plt.grid(True)

plt.show()
#Moving average function

def moving_average(data, d_mediamovel):

    

    media_movel = []

    

    for i in range(len(data)+1-d_mediamovel):

        arr = data[i:i+d_mediamovel]

        media_movel.append(arr.mean())

    

    x = range(d_mediamovel,len(data)+1,1)

    y = media_movel

    

    return x , y 
def SigmoideGompertz(x,y,country,d_mediamovel, previsao):



    y_med_movel_d = []

    

    if (d_mediamovel>0):

        x,y = moving_average(y, d_mediamovel)

    

        for count in range (len(y)-1):

            y_med_movel_d.append(y[count+1]-y[count])



    y_med_movel = y    

    

    a = b = g = 0

    try:

        

        def sigGompertz (x, a, b, g):

            return a*np.exp(-b*np.exp(-g*x))



        popt, pcov = curve_fit(sigGompertz, x, y,method='lm',maxfev = 8000)

        popt

        a=popt[0]

        b=popt[1]

        g=popt[2]

    

    except:

        popt, pcov = curve_fit(sigGompertz, x, y,method='trf')

        a=popt[0]

        b=popt[1]

        g=popt[2]

        

    if (b<0):

        popt, pcov = curve_fit(sigGompertz, x, y,method='trf')

        a=popt[0]

        b=popt[1]

        g=popt[2]

    

    def devGompertz(x, a,b,g):

        return np.array(a*b*g*np.exp(-b*np.exp(-g*x))*np.exp(-g*x))

    

    sig_inf = int((1/g)*np.log(b))

    

    n_max_deaths = a

        

    diascomprevisao = range(len(y)+previsao+d_mediamovel)

    ypred_ssigGompertz = sigGompertz(diascomprevisao, a,b,g)

    ypred_devsigGompertz = devGompertz(diascomprevisao, a,b,g)

    

    def rmse(predictions, targets):

        return np.sqrt(((predictions - targets) ** 2).mean())

    

    err = rmse(y, ypred_ssigGompertz [d_mediamovel:len(y)+d_mediamovel])

    

    return ypred_ssigGompertz, ypred_devsigGompertz, (1/g)*np.log(b), a*np.exp(-1), n_max_deaths, err, y_med_movel,y_med_movel_d
forecasting = pd.DataFrame()

Dias_previsao = 25

Media_movel = 0



for i in deaths_global_zero.columns:

    treino = deaths_global_zero[i][~pd.isnull(deaths_global_zero[i])]   

    forecasting[i] = SigmoideGompertz(treino.index, treino , i ,Media_movel, Dias_previsao) 
plt.figure(figsize=(30,30))

subplt = 1

for i in deaths_global_zero.columns:

    plt.subplot(3, 3, subplt)

    plt.plot(forecasting[i][0])

    plt.plot(deaths_global_zero[i])

    plt.axhline(forecasting[i][4], color='g', ls ='-.')

    plt.axvline(forecasting[i][2], color='b', ls ='-.')

    plt.axhline(forecasting[i][3], color='b', ls ='-.')

    plt.title(i, size=25)

    subplt+=1
plt.figure(figsize=(30,30))

subplt = 1

for i in deaths_global_zero.columns:

    plt.subplot(3, 3, subplt)

    plt.plot(forecasting[i][1])

    plt.plot(deaths_global_difference[i])

    plt.axvline(forecasting[i][2], color='b', ls ='-.')

    plt.title(i, size=25)

    subplt+=1
plt.figure(figsize=(30,30))



for i in deaths_global_zero.columns:

    plt.bar(i, forecasting[i][5])

    plt.title("Root Mean Squared Error (RMSE)", size=25)

    plt.rcParams['xtick.labelsize'] = 25

    plt.rcParams['ytick.labelsize'] = 25
forecasting_med_mov = pd.DataFrame()

Dias_previsao = 50

Media_movel = 5



for i in deaths_global_zero.columns:

    treino = deaths_global_zero[i][~pd.isnull(deaths_global_zero[i])] 

    forecasting_med_mov[i] = SigmoideGompertz(treino.index, treino ,i,Media_movel, Dias_previsao) 
plt.figure(figsize=(30,30))

subplt = 1

for i in deaths_global_zero.columns:

    plt.subplot(3, 3, subplt)

    plt.plot(forecasting_med_mov[i][0])

    plt.plot(range(Media_movel,len(forecasting_med_mov[i][6])+Media_movel), forecasting_med_mov[i][6])

    plt.axhline(forecasting_med_mov[i][4], color='g', ls ='-.')

    plt.axvline(forecasting_med_mov[i][2], color='b', ls ='-.')

    plt.axhline(forecasting_med_mov[i][3], color='b', ls ='-.')

    plt.title(i, size=25)

    subplt+=1
plt.figure(figsize=(30,30))

subplt = 1

for i in deaths_global_zero.columns:

    plt.subplot(3, 3, subplt)

    plt.plot(forecasting_med_mov[i][1])

    plt.plot(range(Media_movel,len(forecasting_med_mov[i][7])+Media_movel),forecasting_med_mov[i][7])

    plt.axvline(forecasting_med_mov[i][2], color='b', ls ='-.')

    plt.title(i, size=25)

    subplt+=1
erro1 = []

erro2 = []

for i in deaths_global_zero.columns:

    erro1.append(forecasting[i][5])

    erro2.append(forecasting_med_mov[i][5])



plt.figure(figsize=(30,30))

w=0.8

ax = plt.subplot(111)

ax.bar(deaths_global_zero.columns,erro1, width=w,color='b', align='center')

ax.bar(deaths_global_zero.columns,erro2, width=w,color='r', align='center')   

ax.legend(['Moving Average Not Applied','Applied Moving Average'], loc='upper left', fontsize=20)

plt.title("Root Mean Squared Error (RMSE)", size=25)

ax.autoscale(tight=True)
forecasting_med_mov_inflex = pd.DataFrame()



d_pre_inflex = 5

Dias_previsao = 7

Media_movel = 5



for i in deaths_global_zero.columns:

    treino = deaths_global_zero[i][~pd.isnull(deaths_global_zero[i])] 

    treino = treino[:int(forecasting_med_mov[i][2])-d_pre_inflex]

    forecasting_med_mov_inflex[i] = SigmoideGompertz(treino.index, treino ,i,Media_movel, Dias_previsao) 
plt.figure(figsize=(30,30))

subplt = 1

for i in deaths_global_zero.columns:

    plt.subplot(3, 3, subplt)

    plt.plot(forecasting_med_mov_inflex[i][0])

    plt.plot(range(Media_movel,len(forecasting_med_mov_inflex[i][6])+Media_movel), forecasting_med_mov_inflex[i][6])

    plt.axhline(forecasting_med_mov_inflex[i][4], color='g', ls ='-.')

    plt.axvline(forecasting_med_mov_inflex[i][2], color='b', ls ='-.')

    plt.axhline(forecasting_med_mov_inflex[i][3], color='b', ls ='-.')

    plt.title(i, size=25)

    subplt+=1
plt.figure(figsize=(30,30))

subplt = 1

for i in deaths_global_zero.columns:

    plt.subplot(3, 3, subplt)

    plt.plot(forecasting_med_mov_inflex[i][1])

    plt.plot(range(Media_movel,len(forecasting_med_mov_inflex[i][7])+Media_movel),forecasting_med_mov_inflex[i][7])

    plt.axvline(forecasting_med_mov_inflex[i][2], color='b', ls ='-.')

    plt.title(i, size=25)

    subplt+=1
dados_erros={}



for i in forecasting_med_mov.columns:

    erro_previsao = []

    if (i != 'Brazil'):

        reality = forecasting_med_mov[i][6][len(forecasting_med_mov_inflex[i][6]):len(forecasting_med_mov_inflex[i][6])+Dias_previsao]

        forecast7daysbefore = forecasting_med_mov_inflex[i][0][-Dias_previsao:]

        

        for day in range(Dias_previsao):

            erro_previsao.append((100*abs(reality[day]-forecast7daysbefore[day])/reality[day]))

        dados_erros[i]=erro_previsao

        

dados_erros = pd.DataFrame(dados_erros)

dados_erros
dados_erros.plot.bar(rot=90,figsize=(15,8),title='7-Day Forecast Error Rate',fontsize=12);
forecasting_med_mov_inflex = pd.DataFrame()

d_pos_inflex = 5

Dias_previsao = 7

Media_movel = 5



for i in deaths_global_zero.columns:

    treino = deaths_global_zero[i][~pd.isnull(deaths_global_zero[i])] 

    treino = treino[:int(forecasting_med_mov[i][2])+d_pos_inflex]

    forecasting_med_mov_inflex[i] = SigmoideGompertz(treino.index, treino ,i,Media_movel, Dias_previsao) 
plt.figure(figsize=(30,30))

subplt = 1

for i in deaths_global_zero.columns:

    plt.subplot(3, 3, subplt)

    plt.plot(forecasting_med_mov_inflex[i][0])

    plt.plot(range(Media_movel,len(forecasting_med_mov_inflex[i][6])+Media_movel), forecasting_med_mov_inflex[i][6])

    plt.axhline(forecasting_med_mov_inflex[i][4], color='g', ls ='-.')

    plt.axvline(forecasting_med_mov_inflex[i][2], color='b', ls ='-.')

    plt.axhline(forecasting_med_mov_inflex[i][3], color='b', ls ='-.')

    plt.title(i, size=25)

    subplt+=1
plt.figure(figsize=(30,30))

subplt = 1

for i in deaths_global_zero.columns:

    plt.subplot(3, 3, subplt)

    plt.plot(forecasting_med_mov_inflex[i][1])

    plt.plot(range(Media_movel,len(forecasting_med_mov_inflex[i][7])+Media_movel),forecasting_med_mov_inflex[i][7])

    plt.axvline(forecasting_med_mov_inflex[i][2], color='b', ls ='-.')

    plt.title(i, size=25)

    subplt+=1
dados_erros={}



for i in forecasting_med_mov.columns:

    erro_previsao = []

    if (i != 'Brazil'):

        reality = forecasting_med_mov[i][6][len(forecasting_med_mov_inflex[i][6]):len(forecasting_med_mov_inflex[i][6])+Dias_previsao]

        forecast7daysbefore = forecasting_med_mov_inflex[i][0][-Dias_previsao:]

        

        for day in range(Dias_previsao):

            erro_previsao.append((100*abs(reality[day]-forecast7daysbefore[day])/reality[day]))

        dados_erros[i]=erro_previsao

        

dados_erros = pd.DataFrame(dados_erros)

dados_erros
dados_erros.plot.bar(rot=90,figsize=(15,8),title='7-Day Forecast Error Rate',fontsize=12);
total_mortos = {}

for i in forecasting_med_mov.columns:

    lista_mortos=[]

    lista_mortos.append(forecasting_med_mov[i][4])

    total_mortos[i]=lista_mortos

    

total_mortos = pd.DataFrame(total_mortos)

total_mortos = total_mortos.append(deaths_global_df.iloc[-1:])

total_mortos.index.names = ['Comparison Prediction and Reality']

total_mortos = total_mortos.rename(index={0: 'No. Max. Dead Forecast Model',120:'Current Values'})

total_mortos
total_mortos.T.plot.bar(rot=0,figsize=(15,8),title='Comparison of the Maximum Value Predicted in the Model and Current Value',fontsize=12)