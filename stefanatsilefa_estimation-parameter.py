import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from scipy import integrate, optimize

import warnings

warnings.filterwarnings('ignore')

import seaborn as sns

sns.set_style("darkgrid") 
data = pd.read_csv("../input/corona-virus-report/full_grouped.csv")

data.rename(columns={'New cases':'New_cases'},inplace=True)

data.rename(columns={'New deaths':'New_deaths'},inplace=True)

data.rename(columns={'New recovered':'New_recovered'},inplace=True)



data_df = pd.DataFrame()

data_df['Confirmé'] = data.loc[data['Country/Region']=='Spain'].New_cases

data_df['Guerri'] = data.loc[data['Country/Region']=='Spain'].Recovered

data_df['mor'] = data.loc[data['Country/Region']=='Spain'].Deaths



data_df = data_df[10:]

data_df['day_count'] = list(range(1,len(data_df)+1))



ydata = [i for i in data_df.Confirmé]

ydata1 = [i for i in data_df.Guerri]

ydata2 = [i for i in data_df.mor]



xdata = data_df.day_count

y = np.array(ydata, dtype=float)

y1 = np.array(ydata1, dtype=float)

y2 = np.array(ydata2, dtype=float)

x = np.array(xdata, dtype=float)
data_df.head()
population = float(46094000 )

N = population

inf0 = ydata[0]

sus0 = N - inf0

rec0 = 0.0

mor0 = 0.0
def sir_model(y, x, alpha, p):

    S,I,R,D = y

    sus = -alpha * S * I/ N

    inf = (alpha * S * I/ N) - (p*0.05 * I) - ((1-p)*0.04 * I)

    rec = p*0.05 * I

    mor = (1-p)*0.04 * I

    return sus, inf, rec,mor



def fit_odeint(x, alpha, p):

    return integrate.odeint(sir_model, (sus0, inf0, rec0,mor0), x, args=(alpha, p))[:,2]



def fit_odeint1(x, alpha, p):

    return integrate.odeint(sir_model, (sus0, inf0, rec0,mor0), x, args=(alpha, p))[:,3]



popt, pcov = optimize.curve_fit(fit_odeint, x, y1)

popt1, pcov1 = optimize.curve_fit(fit_odeint1, x, y2)



fitted = fit_odeint(xdata, *popt)

fitted1 = fit_odeint1(xdata, *popt1)

print("Paramètre optimale: alpha =", popt[0], " and p = ", 1-popt[1])

print("Paramètre optimale: alpha =", popt1[0], " and 1-p = ", 1-popt1[1])
plt.figure(figsize=(15,4))

plt.subplot(1,2,1)

plt.plot(x, y1, 'o',label='Données réel guerrison')

plt.plot(x, fitted,c='r',label='Estimation')

plt.xlabel('Jours')

plt.ylabel('Population')

plt.title('Paramètre optimale: alpha = {0:.4f}  et p = {1:.4f}'.format(popt[0],1-popt[1]))

plt.legend()



plt.subplot(1,2,2)

plt.plot(x, y2, 'o',label='Données réel mortalité')

plt.plot(x, fitted1,c='r',label='Estimation')

plt.xlabel('Jours')

plt.ylabel('Population')

plt.title('Paramètre optimale: alpha = {0:.4f}  et 1-p = {1:.4f}'.format(popt1[0],1-popt1[1]))

plt.legend()

df = pd.read_csv('../input/covid19-madagacar/Mada-COVID.csv')

df.head()
mada_df = pd.DataFrame()

mada_df['Date'] = df.Date.unique()

mada_df['Cas_journalier'] = df.groupby('Date').agg({'Nouveau_cas':['sum']}).values

mada_df['Guerrison_journalier'] = df.groupby('Date').agg({'cas_Guerri':['sum']}).values

mada_df['Mort_journalier'] = df.groupby('Date').agg({'cas_mor':['sum']}).values

mada_df['Cas_cumuler'] = df.groupby('Date').agg({'Confirmé':['mean']}).values

mada_df['Guerrison_cumuler'] = df.groupby('Date').agg({'Guerrison':['mean']}).values

mada_df['Mort_cumuler'] = df.groupby('Date').agg({'Mortalite':['mean']}).values

mada_df = mada_df.set_index('Date')



mada_df['day_count'] = list(range(1,len(mada_df)+1))



ydata = [i for i in mada_df.Cas_journalier]

ydata1 = [i for i in mada_df.Guerrison_cumuler]

ydata2 = [i for i in mada_df.Mort_cumuler]



xdata = mada_df.day_count

y = np.array(ydata, dtype=float)

y1 = np.array(ydata1, dtype=float)

y2 = np.array(ydata2, dtype=float)

x = np.array(xdata, dtype=float)
population = float(26026000)

N = population

inf0 = ydata[0]

sus0 = N - inf0

rec0 = 0.0

mor0 = 0.0
def sir_model(y, x, alpha, p):

    S,I,R,D = y

    sus = -alpha * S * I/ N

    inf = (alpha * S * I/ N) - (p*0.05 * I) - ((1-p)*0.04 * I)

    rec = p*0.05 * I

    mor = (1-p)*0.04 * I

    return sus, inf, rec,mor



def fit_odeint(x, alpha, p):

    return integrate.odeint(sir_model, (sus0, inf0, rec0,mor0), x, args=(alpha, p))[:,2]



def fit_odeint1(x, alpha, p):

    return integrate.odeint(sir_model, (sus0, inf0, rec0,mor0), x, args=(alpha, p))[:,3]



popt, pcov = optimize.curve_fit(fit_odeint, x, y1)

popt1, pcov1 = optimize.curve_fit(fit_odeint1, x, y2)



fitted = fit_odeint(xdata, *popt)

fitted1 = fit_odeint1(xdata, *popt1)

print("Paramètre optimale: alpha =", popt[0], " and p = ", 1-popt[1])

print("Paramètre optimale: alpha =", popt1[0], " and 1-p = ", 1-popt1[1])
plt.figure(figsize=(15,4))

plt.subplot(1,2,1)

plt.plot(x, y1, 'o',label='Données réel guerrison')

plt.plot(x, fitted,c='r',label='Estimation')

plt.xlabel('Jours')

plt.ylabel('Population')

plt.title('Paramètre optimale: alpha = {0:.4f}  et p = {1:.4f}'.format(popt[0],1-popt[1]))

plt.legend()



plt.subplot(1,2,2)

plt.plot(x, y2, 'o',label='Données réel mortalité')

plt.plot(x, fitted1,c='r',label='Estimation')

plt.xlabel('Jours')

plt.ylabel('Population')

plt.title('Paramètre optimale: alpha = {0:.4f}  et 1-p = {1:.4f}'.format(popt1[0],1-popt1[1]))

plt.legend()
df_tana = df[df['Region']=='Analamanga']

df_tamaga = df[df['Region']=='Atsinanana']



tana_df = pd.DataFrame()

tana_df['Date'] = df_tana.Date.unique()

tana_df['Cas_journalier'] = df_tana.groupby('Date').agg({'Nouveau_cas':['sum']}).values

tana_df['Mort_cumuler'] = df_tana.groupby('Date').agg({'cas_mor':['sum']}).cumsum().values

tana_df = tana_df.set_index('Date')



tana_df['day_count'] = list(range(1,len(tana_df)+1))



ydata = [i for i in tana_df.Cas_journalier]

ydata1 = [i for i in tana_df.Mort_cumuler]



xdata = tana_df.day_count

y = np.array(ydata, dtype=float)

y1 = np.array(ydata1, dtype=float)

x = np.array(xdata, dtype=float)
population = float(3618000)

N = population

inf0 = ydata[0]

sus0 = N - inf0

rec0 = 0.0

mor0 = 0.0
def sir_model(y, x, alpha, p):

    S,I,R,D = y

    sus = -alpha * S * I/ N

    inf = (alpha * S * I/ N) - (p*0.05 * I) - ((1-p)*0.04 * I)

    rec = p*0.05 * I

    mor = (1-p)*0.04 * I

    return sus, inf, rec,mor



def fit_odeint(x, alpha, p):

    return integrate.odeint(sir_model, (sus0, inf0, rec0,mor0), x, args=(alpha, p))[:,3]



popt, pcov = optimize.curve_fit(fit_odeint, x, y1)



fitted = fit_odeint(xdata, *popt)

fitted1 = fit_odeint1(xdata, *popt1)

print("Paramètre optimale: beta =", popt[0], " and 1-p = ", 1-popt[1])
plt.figure(figsize=(10,4))

plt.plot(x, y1, 'o',label='Données réel mort cummulée')

plt.plot(x, fitted,c='r',label='Estimation')

plt.xlabel('Jours')

plt.ylabel('Population')

plt.title('Paramètre optimale: alpha = {0:.4f}  et 1-p = {1:.4f}'.format(popt[0],1-popt[1]))

plt.legend()
tamaga_df = pd.DataFrame()

tamaga_df['Date'] = df_tamaga.Date.unique()

tamaga_df['Cas_journalier'] = df_tamaga.groupby('Date').agg({'Nouveau_cas':['sum']}).values

tamaga_df['Mort_cumuler'] = df_tamaga.groupby('Date').agg({'cas_mor':['sum']}).cumsum().values

tamaga_df = tamaga_df.set_index('Date')



tamaga_df['day_count'] = list(range(1,len(tamaga_df)+1))



ydata = [i for i in tamaga_df.Cas_journalier]

ydata1 = [i for i in tamaga_df.Mort_cumuler]



xdata = tamaga_df.day_count

y = np.array(ydata, dtype=float)

y1 = np.array(ydata1, dtype=float)

x = np.array(xdata, dtype=float)
population = float(1484000)

N = population

inf0 = ydata[0]

sus0 = N - inf0

rec0 = 0.0

mor0 = 0.0
def sir_model(y, x, alpha, p):

    S,I,R,D = y

    sus = -alpha * S * I/ N

    inf = (alpha * S * I/ N) - (p*0.05 * I) - ((1-p)*0.04 * I)

    rec = p*0.05 * I

    mor = (1-p)*0.04 * I

    return sus, inf, rec,mor



def fit_odeint(x, alpha, p):

    return integrate.odeint(sir_model, (sus0, inf0, rec0,mor0), x, args=(alpha, p))[:,3]



popt, pcov = optimize.curve_fit(fit_odeint, x, y1)



fitted = fit_odeint(xdata, *popt)

fitted1 = fit_odeint1(xdata, *popt1)

print("Paramètre optimale: alpha =", popt[0], " and 1-p = ", 1-popt[1])
plt.figure(figsize=(10,4))

plt.plot(x, y1, 'o',label='Données réel mort cummulée')

plt.plot(x, fitted,c='r',label='Estimation')

plt.xlabel('Jours')

plt.ylabel('Population')

plt.title('Paramètre optimale: alpha = {0:.4f}  et 1-p = {1:.4f}'.format(popt[0],1-popt[1]))

plt.legend()