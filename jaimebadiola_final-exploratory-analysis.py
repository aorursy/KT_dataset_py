# %matplotlib notebook

%matplotlib inline

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

# plt.style.use('seaborn-deep')

#Load dataset



df = pd.read_csv('../input/df_Final.csv', sep = ';')[['Date','Close', 'Compound_Score', 'Total Volume of Tweets', 'Count_Negatives',

       'Count_Positives', 'Count_Neutrals', 'Sent_Negatives', 'Sent_Positives',

       'Count_News', 'Count_Bots']]

df = df.set_index('Date')

df.index = pd.to_datetime(df.index)

df.head()
#add open and change columns

df['Open'] = df['Close'].shift(1)

df['Change'] = df['Close'] - df['Open']

df['Sent_Open'] = df['Compound_Score'].shift(1)

df['Sent_change'] = df['Compound_Score'] - df['Sent_Open']

df['Compound_Score_t-24'] = df['Compound_Score'].shift(24)
# These are the "Tableau 20" colors as RGB.  

tableau20 = [(252,79,48),(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),

             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),

             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),

#              (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),

             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]



# Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.  

for i in range(len(tableau20)):

    r, g, b = tableau20[i]

    tableau20[i] = (r / 255., g / 255., b / 255.)
#Lets take a look at the movement of price through our dataset



plt.figure(figsize = (13,10))

plt.plot(df[df.index <= '2017-12-17 11:00:00']['Close'], label='Mercado Alcista', linewidth=3)

plt.plot(df[df.index > '2017-12-17 11:00:00']['Close'], label='Mercado Bajista' , linewidth=3)

plt.title('Precio histórico de Bitcoin', fontsize = 'xx-large')

plt.ylabel('Precio')

plt.xlabel('Tiempo')

plt.legend(fontsize= 'medium')
#Lets take a look at the distribution of the sentiment durning both markets

plt.figure(figsize = (13,10))

plt.hist([df[df.index <= '2017-12-17 11:00:00']['Compound_Score'],

          df[df.index > '2017-12-17 11:00:00']['Compound_Score']], 

         bins=100, label=['Sentimimiento en el Mercado Alcista', 'Sentimiento en el Mercado Bajista'], histtype ='stepfilled', stacked= True)

plt.vlines(df[df.index <= '2017-12-17 11:00:00']['Compound_Score'].mean(), 0, 1000, linestyle='dotted', linewidth=2, label = 'Media del sentimiento en el Mercado Alcista')

plt.vlines(df[df.index > '2017-12-17 11:00:00']['Compound_Score'].mean(), 0, 1000, linestyle='dashed', linewidth=2, label = 'Media del sentimiento en el Mercado Bajista')



plt.title('Histograma del sentimiento', fontsize = 'xx-large')

plt.ylabel('Cantidad de valores')

plt.xlabel('Sentimiento')



plt.legend(loc='upper right')

plt.show()

df_12h_MA_Sent = df[(df.index >= '2018-08-01 01:00:00') & (df.index <= '2018-08-10 01:00:00')]['Compound_Score'].fillna(method = 'ffill').rolling(window=24).mean()

df_12h_MA_Change = df[(df.index >= '2018-08-01 01:00:00') & (df.index <= '2018-08-10 01:00:00')]['Change'].fillna(method = 'ffill').rolling(window=24).mean()



fig, ax1 = plt.subplots(figsize = (13,10))

lsb1 = ax1.plot(df_12h_MA_Change, label='Cambio en el precio de Bitcoin', linewidth=3, linestyle='dashed', marker='o', markerfacecolor = tableau20[2], markeredgecolor = tableau20[2], c=tableau20[1])

ax1.set_title('Cambio del precio de Bitcoin comparado con el sentimiento de tweets', fontsize = 'xx-large')

ax1.set_ylabel('Cambio del precio')

ax1.set_xlabel('Tiempo')

ax1.grid(alpha=.4)



ax2 = ax1.twinx()

lsb2 = ax2.plot(df_12h_MA_Sent, label='Sentimiento de los Tweets',marker = 's', markerfacecolor = tableau20[8], markeredgecolor= tableau20[8], linestyle='dashed', linewidth=2, c=tableau20[7])

ax2.set_ylabel('Sentimiento')



lsb = lsb1 + lsb2

labs = [l.get_label() for l in lsb]

ax1.legend(lsb, labs,fontsize= 'medium', loc = 'best')

plt.show()

df_12h_MA_Sent = df[(df.index >= '2018-08-01 01:00:00') & (df.index <= '2018-08-10 01:00:00')]['Compound_Score']

df_12h_MA_Change = df[(df.index >= '2018-08-01 01:00:00') & (df.index <= '2018-08-10 01:00:00')]['Change']



fig, ax1 = plt.subplots(figsize = (13,10))

lsb1 = ax1.plot(df_12h_MA_Change, label='Cambio en el precio de Bitcoin', linewidth=3, linestyle='dashed', marker='o', markerfacecolor = tableau20[2], markeredgecolor = tableau20[2], c=tableau20[1])

ax1.set_title('Cambio del precio de Bitcoin comparado con el sentimiento de tweets', fontsize = 'xx-large')

ax1.set_ylabel('Cambio del precio')

ax1.set_xlabel('Tiempo')

ax1.grid(alpha=.4)

ax1.set_ylim(bottom = -220, top = 220)







ax2 = ax1.twinx()

lsb2 = ax2.plot(df_12h_MA_Sent, label='Sentimiento de los Tweets',marker = 's', markerfacecolor = tableau20[8], markeredgecolor= tableau20[8], linestyle='dashed', linewidth=2, c=tableau20[7])

ax2.set_ylabel('Sentimiento')

ax2.set_ylim(bottom = -0.35, top = 0.35)

lsb = lsb1 + lsb2

labs = [l.get_label() for l in lsb]

ax1.legend(lsb, labs,fontsize= 'medium', loc = 'best')

plt.hlines(y=0, xmin = '2018-08-01', xmax = '2018-08-10', linewidth=2, alpha = 0.5)

plt.show()

#Lets make a moving average

df_1d_MA_Tweets = df['Total Volume of Tweets'].fillna(method = 'ffill').rolling(window=168).mean()



#Lets take a look at the movement of price through our dataset

fig, ax1 = plt.subplots(figsize = (13,10))

lsb1 = ax1.plot(df['Close'], label='Precio Bitcoin', linewidth=3)

ax1.set_title('Precio histórico de Bitcoin comparado con el volumen de tweets', fontsize = 'xx-large')

ax1.set_ylabel('Precio')

ax1.set_xlabel('Tiempo')



#Let's take a look at Volumen de Tweets

ax2 = ax1.twinx()

lsb2 = ax2.plot(df_1d_MA_Tweets, label='Media movil semanal del volumen de Tweets' , linewidth=3, c=tableau20[0])

ax2.set_ylabel('Cantidad de Tweets')



lsb = lsb1 + lsb2

labs = [l.get_label() for l in lsb]

ax1.legend(lsb, labs,fontsize= 'medium', loc = 'best')

plt.show()

plt.figure(figsize=(13,10))

plt.scatter(df['Change'], df['Compound_Score'], label="Relación del sentimiento respecto a Bitcoin")

plt.legend()

plt.ylabel('Sentimiento')

plt.xlabel('Cambio en el precio')

plt.title("Gráfica de dispersión del sentimiento respecto a Bitcoin")
plt.figure(figsize=(13,10))

plt.scatter(df['Change'], df['Compound_Score_t-24'], label="Relación del sentimiento respecto a Bitcoin")

plt.legend()

plt.ylabel('Sentimiento')

plt.xlabel('Cambio en el precio')

plt.title("Gráfica de dispersión del sentimiento respecto a Bitcoin en t-24 ")
#Correlation. Basically I want to see how the volume of today affects the value of the future. 

#Apparently it affects positively until 100 days later

Autocorr = []

n = 0

for a in range(1500):

    corr_df = df[['Change']]

    corr_df[['Compound_Score']] = df[['Compound_Score']].shift(a)

    cor = corr_df.corr()['Change'][1]

    if str(cor) == 'nan':

        break

    Autocorr.append(cor)

plt.figure(figsize=(16,13))

plt.plot(Autocorr, linewidth = 2, label = 'Correlación' )

plt.title('Correlación entre el sentimiento y el cambio del precio con nLags en el futuro')

plt.hlines(0.05, 0, len(Autocorr), linestyles = 'dashed', colors = tableau20[0], linewidth = 2, alpha = 0.5)

plt.hlines(0, 0, len(Autocorr), linestyles = 'solid', colors = 'black')

plt.hlines(-0.05, 0, len(Autocorr), linestyles = 'dashed', colors = tableau20[0], linewidth = 2, alpha = 0.5)

plt.ylabel('Correlación');

plt.legend()

plt.xlabel('Lags temporales (1 Lag = 1 Hora)')

plt.grid(alpha = 0.5)



plt.show();
Autocorr = []

n = 0

for a in range(1500):

    corr_df = df[['Close']]

    corr_df[['Total Volume of Tweets']] = df[['Total Volume of Tweets']].shift(a)

    cor = corr_df.corr()['Close'][1]

    if str(cor) == 'nan':

        break

    Autocorr.append(cor)
plt.figure(figsize=(16,13))

plt.plot(Autocorr, linewidth = 2, label = 'Correlación' )

plt.title('Correlación entre el volumen y el precio con nLags en el futuro')

plt.hlines(0.05, 0, len(Autocorr), linestyles = 'dashed', colors = tableau20[0], linewidth = 2, alpha = 0.5)

plt.hlines(0, 0, len(Autocorr), linestyles = 'solid', colors = 'black')

plt.hlines(-0.05, 0, len(Autocorr), linestyles = 'dashed', colors = tableau20[0], linewidth = 2, alpha = 0.5)

plt.ylabel('Correlación');

plt.legend()

plt.xlabel('Lags temporales (1 Lag = 1 Hora)')

plt.grid(alpha = 0.5)



plt.show();

