import os # accessing directory structure

import itertools

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import statsmodels.api as sm

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import matplotlib.ticker as ticker

from pandas.plotting import andrews_curves

import plotly.plotly as py

import math

import datetime as dt

from datetime import datetime

from dateutil.parser import parse

#df.to_csv('pandas_dataframe_importing_csv/example.csv') #exporting to csv



# Any results you write to the current directory are saved as output.
print(os.listdir('../input'))
df = pd.read_csv('../input/spotifys-worldwide-daily-song-ranking/data.csv')

df2 = pd.read_csv('../input/top-spotify-tracks-of-2018/top2018.csv')

df3 = pd.read_csv('../input/top-tracks-of-2017/featuresdf.csv')

df.columns = (df.columns.str.lower()

                .str.replace(' ', '_'))

df.columns
df.head()


df['date'] = df['date'].astype('datetime64[ns]')

plt.rcParams["figure.figsize"] = [16,9]

plt.plot(df['date'], df['streams'],'*')

plt.xticks(rotation='vertical')

plt.ylabel('Streams')

plt.xlabel('Dates')

plt.title('Scatterplot of Streams vs Time')
df3.columns
df3 = df3.rename(columns={'name': 'track_name'})

df3 = df3.drop(columns=['id'])

df3.head()
df2 = df2.rename(columns={'name': 'track_name'})

df2 = df2.drop(columns=['id'])

df3.head()
df.drop('url', axis=1, inplace=True)

# Validate all artists and track names are the same missing

(df['artist'].isna() == df['track_name'].isna()).all()


# drop null rows

df = df.dropna(axis=0)

df = df[df['region'].str.contains("us")] #include only a region, US
df.describe(include='all')
df.artist.describe()
df.position.describe()
df.track_name.describe()
df.streams.describe()
#histogram

sns.distplot(df['position'])
x = np.log10(df.loc[:,'streams'])

sns.distplot(x, hist=False)
ax1 = df.plot.scatter(x='position', y='streams', c='DarkBlue')
sns.jointplot(x='position', y='streams', data=df);
df['date'].min() #First date entry
df['date'].max() #Last date entry
dftest = df[(df['date'] > '2017-12-31')]

df=df[(df['date'] < '2018-01-01')]
df.tail()
dftest.head()
dftest.tail()
dfnew = pd.merge(df, df3, on='track_name')

dfnew = dfnew.drop(columns=['artists'])

dfnew.sort_values(by='date')

dfnew.head()
dftest = pd.merge(dftest, df2, on='track_name')

dftest = dftest.drop(columns=['artists'])

dftest.head()
dfnew.describe()
testtn = dftest.track_name.unique()
dftesttn = pd.DataFrame({'track_name': testtn})

print(dftesttn)
dftrain = pd.merge(dfnew, dftesttn, on='track_name')



dftrain.head()


fig,ax= plt.subplots()

for n, group in dftrain.groupby('track_name'):

    group.plot(x='date',y='streams', ax=ax,label=n)

    

plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',

           ncol=4, mode="expand", borderaxespad=1.)

plt.title('Scatterplot of Streams vs Time for Training Data Track Names')


fig,ax= plt.subplots()

for n, group in dftrain.groupby('track_name'):

    group.plot(x='date',y='position', ax=ax,label=n)

    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',

           ncol=4, mode="expand", borderaxespad=1.)

plt.gca().invert_yaxis()

plt.title('Scatterplot of Position vs Time for Training Data Track Names')
x=dftrain['time_signature']

plt.hist(x, bins=10)

plt.gca().set(title='Time Signature in Ranking Data', ylabel='Frequency');


x=dftrain['instrumentalness']

plt.hist(x, bins=10)

plt.gca().set(title='Instrumentalness in Ranking Data', ylabel='Frequency');


x=dftrain['speechiness']

plt.hist(x, bins=10)

plt.gca().set(title='Speechiness in Ranking Data', ylabel='Frequency');
corr = dftrain.corr()

corr.style.background_gradient(cmap='cool').set_precision(2)
dfin = dfnew.set_index('date')

dfin['streams'].plot(linewidth=0.5);
ax = dfin.loc['2017-08':'2017-10', 'streams'].plot(marker='o', linestyle='-')

from IPython.display import Image

Image("../input/images2/positioncolormap.PNG")


Image("../input/images/img/img/artistvsstreams.png")


Image("../input/images2/tracknamevsstreams.png")


Image("../input/images/img/img/DanceabilityvsStreams.png")


Image("../input/images/img/img/DurationvsStreams.png")


Image("../input/images/img/img/EnergyvsStreams.png")


Image("../input/images/img/img/InstrumentalnessvsStreams.png")


Image("../input/images/img/img/KeyvsStreams.png")


Image("../input/images/img/img/LivenessvsStreams.png")


Image("../input/images/img/img/LoudnessvsStreams.png")


Image("../input/images/img/img/ModevsStreams.png")


Image("../input/images/img/img/SpeechinessvsStreams.png")


Image("../input/images/img/img/TempovsStreams.png")


Image("../input/images/img/img/TimeSignaturevsStreams.png")


Image("../input/images/img/img/ValencevsStreams.png")


Image("../input/images/img/img/AcousticnessvsStreams.png")
dftrainin = dftrain.set_index('date')

y = dftrainin['streams'].resample('D').mean()

p = d = q = range(0, 2)

pdq = list(itertools.product(p, d, q))

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

for param in pdq:

    for param_seasonal in seasonal_pdq:

        try:

            mod = sm.tsa.statespace.SARIMAX(y,

                                            order=param,

                                            seasonal_order=param_seasonal,

                                            enforce_stationarity=False,

                                            enforce_invertibility=False)

            results = mod.fit()

            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

        except:

            continue

mod = sm.tsa.statespace.SARIMAX(y,

                                order=(0, 1, 0),

                                seasonal_order=(1, 1, 1, 12),

                                enforce_stationarity=False,

                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])
results.plot_diagnostics(figsize=(16, 8))

plt.show()
pred = results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)

pred_ci = pred.conf_int()

ax = y['2017':].plot(label='observed')

pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 7))

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')

ax.set_ylabel('streams')

plt.legend()

plt.show()