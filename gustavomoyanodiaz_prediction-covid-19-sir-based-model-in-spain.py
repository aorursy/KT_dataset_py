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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
import time
from datetime import datetime
from scipy import integrate, optimize
import warnings
warnings.filterwarnings('ignore')
es_confirmed = pd.read_csv("/kaggle/input/covid19-in-spain/ccaa_covid19_casos_long.csv")
es_confirmed_pcr = pd.read_csv("/kaggle/input/covid19inspain2/ccaa_covid19_confirmados_pcr_long.csv")
es_confirmed_test = pd.read_csv("/kaggle/input/covid19inspain2/ccaa_covid19_confirmados_test_long.csv")
es_deaths = pd.read_csv("/kaggle/input/covid19-in-spain/ccaa_covid19_fallecidos_long.csv")
es_UCI = pd.read_csv("/kaggle/input/covid19-in-spain/ccaa_covid19_uci_long.csv")
es_hospitalized = pd.read_csv("/kaggle/input/covid19-in-spain/ccaa_covid19_hospitalizados_long.csv")
es_recovered = pd.read_csv("/kaggle/input/covid19-in-spain/ccaa_covid19_altas_long.csv")
es_covid = pd.merge(es_confirmed,es_confirmed_pcr, how='outer', left_on=['fecha','cod_ine','CCAA'], right_on=['fecha','cod_ine','CCAA'],suffixes=('_confirmed', '_pcr')).merge(es_confirmed_test, how='outer', left_on=['fecha','cod_ine','CCAA'], right_on=['fecha','cod_ine','CCAA']).merge(es_deaths, how='outer', left_on=['fecha','cod_ine','CCAA'], right_on=['fecha','cod_ine','CCAA'],suffixes=('_test', '_deaths')).merge(es_UCI, how='outer', left_on=['fecha','cod_ine','CCAA'], right_on=['fecha','cod_ine','CCAA']).merge(es_hospitalized, how='outer', left_on=['fecha','cod_ine','CCAA'], right_on=['fecha','cod_ine','CCAA'],suffixes=('_uci', '_hosp')).merge(es_recovered, how='outer', left_on=['fecha','cod_ine','CCAA'], right_on=['fecha','cod_ine','CCAA'])
es_covid["fecha"] = pd.to_datetime(es_covid['fecha'])
del es_confirmed, es_confirmed_pcr, es_confirmed_test, es_deaths, es_UCI, es_hospitalized, es_recovered
es_covid = es_covid.sort_values(by=['CCAA', 'fecha'])
es_covid['diff_total_confirmed'] = es_covid.groupby(['CCAA'])['total_confirmed'].diff().fillna(es_covid['total_confirmed'])
es_covid['diff_total_deaths'] = es_covid.groupby(['CCAA'])['total_deaths'].diff().fillna(es_covid['total_deaths'])
es_covid['diff_total_recovered'] = es_covid.groupby(['CCAA'])['total'].diff().fillna(es_covid['total'])

es_covid['diff_total_confirmed'].fillna(0, inplace=True)
es_covid['diff_total_deaths'].fillna(0, inplace=True)
es_covid['diff_total_recovered'].fillna(0, inplace=True)

es_covid['day_num'] = preprocessing.LabelEncoder().fit_transform(es_covid.fecha)

display(es_covid.loc[es_covid['fecha'] > '2020-05-20'])
print(datetime.now().strftime("%d%b%Y %H:%M"))
print("Dates go from day", min(es_covid['fecha']), "to day", max(es_covid['fecha']), ", a total of", es_covid['fecha'].nunique(), "days")
print("Communities informed: ", es_covid.loc[es_covid['CCAA']!='None']['CCAA'].unique())
missings_count = {col:es_covid[col].isnull().sum() for col in es_covid.columns}
print(pd.DataFrame.from_dict(missings_count, orient='index').nlargest(30, 0))
del missings_count
confirmed_total_date = es_covid.groupby(['fecha']).agg({'total_confirmed':['sum']})
fatalities_total_date = es_covid.groupby(['fecha']).agg({'total_deaths':['sum']})
recovered_total_date = es_covid.groupby(['fecha']).agg({'total':['sum']})
total_date = confirmed_total_date.join(fatalities_total_date).join(recovered_total_date)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17,7))
total_date.plot(ax=ax1)
ax1.set_title("Total confirmed cases", size=13)
ax1.set_ylabel("Number of cases", size=13)
ax1.set_xlabel("Date", size=13)
fatalities_total_date.plot(ax=ax2, color='orange')
ax2.set_title("Total deceased cases", size=13)
ax2.set_ylabel("Number of cases", size=13)
ax2.set_xlabel("Date", size=13)

del confirmed_total_date, fatalities_total_date, recovered_total_date, total_date
del fig, ax1, ax2
grouped = es_covid.groupby('CCAA')
for name, group in grouped:
    fig, (ax_confirmed,ax_death,ax_rec) = plt.subplots(1, 3, figsize=(15,5))
    group.groupby(['fecha','CCAA']).max()['diff_total_confirmed'].unstack().plot(ax=ax_confirmed)
    ax_confirmed.set_title("Daily confirmed cases", size=13)
    group.groupby(['fecha','CCAA']).max()['diff_total_deaths'].unstack().plot(ax=ax_death)
    ax_death.set_title("Daily Death cases", size=13)
    group.groupby(['fecha','CCAA']).max()['diff_total_recovered'].unstack().plot(ax=ax_rec)
    ax_rec.set_title("Daily Recovered cases", size=13)
    

del fig,ax_confirmed,ax_death,ax_rec
del grouped, group, name
sns.set(style="darkgrid")
plt.figure(figsize=(12,7))
sns.lineplot(x='fecha', y='total_confirmed', hue='CCAA', data=es_covid)
plt.xticks(rotation=70)
plt.tight_layout()
sns.set(style="darkgrid")
plt.figure(figsize=(12,7))
sns.lineplot(x='fecha', y='total_deaths', hue='CCAA', data=es_covid)
plt.xticks(rotation=70)
plt.tight_layout()
sns.set(style="darkgrid")
plt.figure(figsize=(12,7))
sns.lineplot(x='fecha', y='total', hue='CCAA', data=es_covid)
plt.xticks(rotation=70)
plt.tight_layout()
es_dic_pob = {'CCAA': ['Andalucía','Aragón','Asturias','Baleares','Canarias','Cantabria','Castilla y León','Castilla La Mancha','Cataluña','C. Valenciana','Extremadura','Galicia','Madrid','Murcia','Navarra','País Vasco','La Rioja','Ceuta','Melilla'],
          'hombres': [4147167,650694,488137,572757,1065971,281801,1181401,1016954,3770123,2465342,5285,1298964,3187312,747615,323631,1073074,156179,42912,43894],
          'mujeres': [4267073,668597,534663,576703,1087418,299277,1218147,1015909,3905094,2538427,53921,1400535,3476082,746283,330583,1134702,160619,41865,42593]
               }       
es_poblacion = pd.DataFrame(es_dic_pob, columns = ['CCAA','hombres', 'mujeres'])
es_poblacion.reset_index().set_index('CCAA')
es_poblacion['total'] = es_poblacion['hombres'] + es_poblacion['mujeres']

del es_dic_pob
es_poblacion
# Susceptible equation
def susceptibility(N, s, i, beta):
    si = -beta*s*i
    return si

# Infected equation
def infection(N, s, i, beta, gamma):
    inf = beta*s*i - gamma*i
    return inf

# Recovered/deceased equation
def recovery(N, i, gamma):
    rec = gamma*i
    return rec
# Runge-Kutta method of 4rth order for 3 dimensions (susceptible s, infected i snd recovered r)
def rK4(N, s, i, r, susceptibility, infection, recovery, beta, gamma, hs):
    s1 = susceptibility(N, s, i, beta)*hs
    i1 = infection(N, s, i, beta, gamma)*hs
    r1 = recovery(N, i, gamma)*hs
    sk = s + s1*0.5
    ik = i + i1*0.5
    rk = r + r1*0.5
    s2 = susceptibility(N, sk, ik, beta)*hs
    i2 = infection(N, sk, ik, beta, gamma)*hs
    r2 = recovery(N, ik, gamma)*hs
    sk = s + s2*0.5
    ik = i + i2*0.5
    rk = r + r2*0.5
    s3 = susceptibility(N, sk, ik, beta)*hs
    i3 = infection(N, sk, ik, beta, gamma)*hs
    r3 = recovery(N, ik, gamma)*hs
    sk = s + s3
    ik = i + i3
    rk = r + r3
    s4 = susceptibility(N, sk, ik, beta)*hs
    i4 = infection(N, sk, ik, beta, gamma)*hs
    r4 = recovery(N, ik, gamma)*hs
    s = s + (s1 + 2*(s2 + s3) + s4)/6
    i = i + (i1 + 2*(i2 + i3) + i4)/6
    r = r + (r1 + 2*(r2 + r3) + r4)/6
    return s, i, r
def SIR(N, b0, beta, gamma, hs):

    # Initial condition
    s = float(N-1)/N -b0
    i = float(1)/N +b0
    r = 0.

    sus, inf, rec= [],[],[]
    for j in range(10000): # Run for a certain number of time-steps
        sus.append(s)
        inf.append(i)
        rec.append(r)
        s,i,r = rK4(N, s, i, r, susceptibility, infection, recovery, beta, gamma, hs)

    return sus, inf, rec
N = es_poblacion['total'].sum()
b0 = 0
beta = 0.7
gamma = 0.2
hs = 0.1

sus, inf, rec = SIR(N, b0, beta, gamma, hs)
f = plt.figure(figsize=(8,5)) 
plt.plot(sus, 'b.', label='susceptible');
plt.plot(inf, 'r.', label='infected');
plt.plot(rec, 'c.', label='recovered/deceased');
plt.title('SIR Model')
plt.xlabel("time", fontsize=10);
plt.ylabel("Fraction of population", fontsize=10);
plt.legend(loc='best')
plt.xlim(0,1000)
plt.savefig('SIR_example.png')
plt.show()

del N, b0, beta, gamma, hs, sus, inf, rec, f
def sir_model(y, x, beta, gamma):
    sus = -beta * y[0] * y[1] / N
    rec = gamma * y[1] 
    inf = -(sus + rec)
    return sus, inf, rec
def estimateParametersSIR(ccaa, initialDay):
    country_df = pd.DataFrame()
    country_df['ConfirmedCases'] = es_covid.loc[es_covid['CCAA']==ccaa].total_confirmed.diff().fillna(0)
    # This cut it's caused by try visual fits over results
    country_df =  country_df[initialDay:]
    country_df['day_count'] = list(range(1,len(country_df)+1))

    ydata = [i for i in country_df.ConfirmedCases]
    xdata = country_df.day_count
    ydata = np.array(ydata, dtype=float)
    xdata = np.array(xdata, dtype=float)

    N = es_poblacion.loc[es_poblacion['CCAA']==ccaa].total
    inf0 = ydata[0]
    sus0 = N - inf0
    rec0 = 0.0

    def sir_model(y, x, beta, gamma):
        sus = -beta * y[0] * y[1] / N
        rec = gamma * y[1]
        inf = -(sus + rec)
        return sus, inf, rec

    def fit_odeint(x, beta, gamma):
        return integrate.odeint(sir_model, (sus0, inf0, rec0), x, args=(beta, gamma))[:,1]

    popt, pcov = optimize.curve_fit(fit_odeint, xdata, ydata)
    fitted = fit_odeint(xdata, *popt)

    plt.plot(xdata, ydata, 'o')
    plt.plot(xdata, fitted)
    plt.title("Fit of SIR model for " +ccaa + " infected cases")
    plt.ylabel("Population infected")
    plt.xlabel("Days")
    plt.show()
    print("Optimal parameters: \nbeta =", popt[0], " \ngamma = ", popt[1])
    es_poblacion.at[es_poblacion['CCAA'] == ccaa,'ini_day'] = initialDay
    es_poblacion.at[es_poblacion['CCAA'] == ccaa,'beta'] = popt[0]
    es_poblacion.at[es_poblacion['CCAA'] == ccaa,'gamma'] = popt[1]
estimateParametersSIR('Andalucía', 16)
estimateParametersSIR('Aragón', 18)
estimateParametersSIR('Asturias', 15)
estimateParametersSIR('Baleares', 17)
estimateParametersSIR('C. Valenciana', 14)
estimateParametersSIR('Canarias', 16)
estimateParametersSIR('Cantabria', 10)
estimateParametersSIR('Castilla La Mancha', 9)
estimateParametersSIR('Castilla y León', 9)
estimateParametersSIR('Cataluña', 5)
estimateParametersSIR('Ceuta', 23)
estimateParametersSIR('Extremadura', 9)
estimateParametersSIR('Galicia', 18)
estimateParametersSIR('La Rioja', 12)
estimateParametersSIR('Madrid', 10)
estimateParametersSIR('Melilla', 22)
estimateParametersSIR('Murcia', 17)
estimateParametersSIR('Navarra', 12)
estimateParametersSIR('País Vasco', 9)
from scipy.integrate import odeint
# Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
def SIR_CCAA_Adjust(ccaa, initialDay, population, beta, gamma):
    # Total population, N.
    N = population
    # Initial number of infected and recovered individuals, I0 and R0.
    I0, R0 = 1, 0
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0
    # A grid of time points (in days)
    t = np.linspace(0, 160, 160)

    # The SIR model differential equations.
    def deriv(y, t, N, beta, gamma):
        S, I, R = y
        dSdt = -beta * S * I / N
        dIdt = beta * S * I / N - gamma * I
        dRdt = gamma * I
        return dSdt, dIdt, dRdt

    # Initial conditions vector
    y0 = S0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T

    # Plot the data on three separate curves for S(t), I(t) and R(t)
    fig = plt.figure(figsize=(10,7), facecolor='w')
    ax = fig.add_subplot(111, facecolor='#dddddd', axisbelow=True)
    ax.plot(t, S/500, 'b', alpha=0.5, lw=2, label='Susceptible')
    ax.plot(t, I/500, 'r', alpha=0.5, lw=2, label='Infected')
    ax.plot(t, R/500, 'g', alpha=0.5, lw=2, label='Recovered with immunity')
    ax.set_xlabel('Time /days')
    ax.set_ylabel('Number (500s)')
    ax.set_ylim(0,1.2)
    ax.set_title(ccaa + ' first adjusted day: ' + str(initialDay))
    ax.yaxis.set_tick_params(length=0)
    ax.xaxis.set_tick_params(length=0)
    ax.grid(b=True, which='major', c='w', lw=2, ls='-')
    legend = ax.legend()
    legend.get_frame().set_alpha(0.5)
    for spine in ('top', 'right', 'bottom', 'left'):
        ax.spines[spine].set_visible(False)
    plt.show()
for index, row in es_poblacion.iterrows():
    ds_ccaa = es_covid.loc[(es_covid["CCAA"] == row['CCAA'])]
    ds_ccaa = ds_ccaa.loc[(ds_ccaa["total_confirmed"] > 0)]
    mini = ds_ccaa["fecha"].min()
    maxi = ds_ccaa["fecha"].max()
    es_poblacion.at[es_poblacion['CCAA'] == row['CCAA'],'fecha_inicio'] = mini
    es_poblacion.at[es_poblacion['CCAA'] == row['CCAA'],'fecha_actual'] = maxi
    SIR_CCAA_Adjust(row['CCAA'],row['ini_day'],row['total'],row['beta'],row['gamma'])
del index, row, mini, maxi
es_poblacion.at[es_poblacion['CCAA'] == 'Andalucía','day_fin'] = 80
es_poblacion.at[es_poblacion['CCAA'] == 'Aragón','day_fin'] = 140
es_poblacion.at[es_poblacion['CCAA'] == 'Asturias','day_fin'] = 100
es_poblacion.at[es_poblacion['CCAA'] == 'Baleares','day_fin'] = 70
es_poblacion.at[es_poblacion['CCAA'] == 'C. Valenciana','day_fin'] = 90
es_poblacion.at[es_poblacion['CCAA'] == 'Canarias','day_fin'] = 70
es_poblacion.at[es_poblacion['CCAA'] == 'Cantabria','day_fin'] = 140
es_poblacion.at[es_poblacion['CCAA'] == 'Castilla La Mancha','day_fin'] = 100
es_poblacion.at[es_poblacion['CCAA'] == 'Castilla y León','day_fin'] = 120
es_poblacion.at[es_poblacion['CCAA'] == 'Cataluña','day_fin'] = 100
es_poblacion.at[es_poblacion['CCAA'] == 'Ceuta','day_fin'] = 60
es_poblacion.at[es_poblacion['CCAA'] == 'Extremadura','day_fin'] = 120
es_poblacion.at[es_poblacion['CCAA'] == 'Galicia','day_fin'] = 90
es_poblacion.at[es_poblacion['CCAA'] == 'La Rioja','day_fin'] = 100
es_poblacion.at[es_poblacion['CCAA'] == 'Madrid','day_fin'] = 80
es_poblacion.at[es_poblacion['CCAA'] == 'Melilla','day_fin'] = 40
es_poblacion.at[es_poblacion['CCAA'] == 'Murcia','day_fin'] = 60
es_poblacion.at[es_poblacion['CCAA'] == 'Navarra','day_fin'] = 140
es_poblacion.at[es_poblacion['CCAA'] == 'País Vasco','day_fin'] = 100
from datetime import timedelta
for index, row in es_poblacion.iterrows():
    ini = pd.to_numeric(row["ini_day"])
    fin = pd.to_numeric(row["day_fin"])
    es_poblacion.at[es_poblacion['CCAA'] == row['CCAA'],'fecha_fin'] = row['fecha_inicio'] + timedelta(days=ini) + timedelta(days=fin)
display(es_poblacion)
