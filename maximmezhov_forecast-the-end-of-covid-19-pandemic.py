import pandas as pd

import math

import datetime

import matplotlib.pyplot as plt

import matplotlib.dates as mdates
url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSe-8lf6l_ShJHvd126J-jGti992SUbNLu-kmJfx1IRkvma_r4DHi0bwEW89opArs8ZkSY5G2-Bc1yT/pub?gid=0&single=true&output=csv"

df = pd.read_csv(url, index_col = 'ADM0_NAME', parse_dates = ['date_epicrv'], dayfirst=True)
df.info()
df_SingleCountry = df.loc['Russian Federation'][['date_epicrv', 'NewCase','CumCase']]
df_SingleCountry
df_SingleCountry['index'] = range(len(df_SingleCountry))
df_SingleCountry.info()
df_SingleCountry.set_index('index', inplace = True)
df_SingleCountry.info()
df_SingleCountry.drop(len(df_SingleCountry)-1, inplace = True)
fig, ax = plt.subplots(figsize=(15, 7))

ax.set_title("Dynamic of new cases", fontsize=16)

ax.set_xlabel("Date", fontsize=14)

ax.set_ylabel("Cases per day", fontsize=14)



ax.plot_date(df_SingleCountry['date_epicrv'], df_SingleCountry['NewCase'], fmt='.-')

plt.show()
t = len(df_SingleCountry)-1

No = df_SingleCountry.at[0,'CumCase']

N = df_SingleCountry.at[t,'CumCase']

k = math.log(N/No)/t

print("Current illness coefficient is {}".format(k))
df_k = pd.DataFrame()



for i in range(t):

    if(i > 0):

        N = df_SingleCountry.at[i,'CumCase']

        df_k.at[i, 1] = math.log(N/No)/i

    else:

        df_k.at[i, 1] = 0

    df_k.at[i, 0] = df_SingleCountry.at[i, 'date_epicrv']

fig, ax = plt.subplots(figsize=(15, 7))

ax.set_title("Dynamic of illness coefficient", fontsize=16)

ax.set_xlabel("Date", fontsize=14)

ax.set_ylabel("Coefficient value", fontsize=14)



ax.plot_date(df_k[0], df_k[1], fmt='.-')

plt.show()
to_end_July = datetime.date(2020, 7, 31) - datetime.date.today() - datetime.timedelta(days = 1)



if to_end_July.days > 0:

    Percent = round(math.exp(-k * to_end_July.days) * 100, 1)

    print("Forecast percent of illness people that will be at the end of July is {} %".format(Percent))

else:

    print('July has gone...')
T = round(math.log((N*0.01)/N)/-k, 1)

end_date_to1per = datetime.date.today() + datetime.timedelta(days=T)

print("Forecast date when only 1% of cases will remain is {}".format(end_date_to1per.strftime("%d.%m.%Y")))
T = round(math.log(1/N)/-k, 1)

end_date = datetime.date.today() + datetime.timedelta(days=T)

print("Forecast date when last case will remain is {}".format(end_date.strftime("%d.%m.%Y")))
print("This calculation was completed on {}".format(datetime.date.today().strftime("%d.%m.%Y")))

print("Current illness coefficient is {}".format(k))

if to_end_July.days > 0:

    print("Forecast percent of illness people that will be at the end of July is {} %".format(Percent))

print("Forecast date when only 1% of cases will remain is {}".format(end_date_to1per.strftime("%d.%m.%Y")))

print("Forecast date when last case will remain is {}".format(end_date.strftime("%d.%m.%Y")))
fig, ax1 = plt.subplots(figsize=(15, 5))



ax1.set_title("Dynamic of new cases", fontsize=16)

ax1.set_ylabel("Cases per day", fontsize=14)

ax1.plot_date(df_SingleCountry['date_epicrv'], df_SingleCountry['NewCase'], fmt='.-')

ymin = 0

ymax = 12000

ax1.vlines(datetime.date(2020, 5, 1), ymin, ymax, linestyles='dotted', label="Start May holidays", color='r')

ax1.vlines(datetime.date(2020, 5, 22), ymin, ymax, linestyles='dotted', label="End of self-isolation in Moscow region", color='g') 

ax1.vlines(datetime.date(2020, 6, 8), ymin, ymax, linestyles='dotted', label="End of self-isolation in Moscow", color='b')

ax1.vlines(datetime.date(2020, 7, 1), ymin, ymax, linestyles='dotted', label="Voiting")

plt.legend(loc='upper left')





fig, ax2 = plt.subplots(figsize=(15, 5))

ax2.set_title("Dynamic of illness coefficient", fontsize=16)

#ax2.set_xlabel("Date", fontsize=14)

ax2.set_ylabel("Coefficient value", fontsize=14)

ax2.plot_date(df_k[0], df_k[1], fmt='.-')





# View period since 01.05.2020

df2_k = df_k[df_k[0] >= datetime.date(2020, 5, 1)]



fig, ax3 = plt.subplots(figsize=(15, 5))

ax3.set_title("Dynamic of illness coefficient since May", fontsize=16)

ax3.set_xlabel("Date", fontsize=14)

ax3.set_ylabel("Coefficient value", fontsize=14)

ax3.plot_date(df2_k[0], df2_k[1], fmt='.-')

ymin = 0

ymax = 0.12

ax3.vlines(datetime.date(2020, 5, 1), ymin, ymax, linestyles='dotted', label="Start May holidays", color='r')

ax3.vlines(datetime.date(2020, 5, 22), ymin, ymax, linestyles='dotted', label="End of self-isolation in Moscow region", color='g') 

ax3.vlines(datetime.date(2020, 6, 8), ymin, ymax, linestyles='dotted', label="End of self-isolation in Moscow", color='b')

ax3.vlines(datetime.date(2020, 7, 1), ymin, ymax, linestyles='dotted', label="Voiting")



plt.show()