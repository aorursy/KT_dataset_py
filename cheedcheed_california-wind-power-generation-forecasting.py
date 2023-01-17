import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from fbprophet import Prophet





plt.rcParams["figure.figsize"] = (12, 9)



df = pd.DataFrame.from_csv('../input/all_breakdown.csv')

df.head()
x = df.index

y = df['Wind Total'.upper()]

hour = df['Hour'].astype(int)

quart_day = hour // 4
plt.subplot(2, 1, 1)



for h, marker in [(1, 'x'), (2, 'v'), (3, '^')]:

    mask = quart_day == h

    plt.scatter(x[mask], y[mask], marker=marker,  # c=hour[mask].norm(),

                label='{:2}00 to {:2}00'.format(h*6, (h+1)*6))

plt.legend(loc='upper left')

plt.xlim([pd.to_datetime('01/01/2016'), pd.to_datetime('01/01/2017')])

plt.title('Solar Power Production in 2016 (MegaWatts)')



# Add the season change lines

seasons = [(pd.to_datetime('June 20, 2016'), 'Summer Solstice'),

           (pd.to_datetime('December 21, 2016'), 'Winter Solstice'),

           (pd.to_datetime('September 22, 2016'), 'Fall Equinox'),

           (pd.to_datetime('March 20, 2016'), 'Spring Equinox'),

           ]



for s, name in seasons:

    plt.axvline(s)

    plt.text(s, 8000, name.replace(' ', '\n'))





plt.subplot(2, 1, 2)

sns.boxplot(hour, y)



plt.title('Solar Power Production grouped by Hour')

plt.show()
daily_solar = y.resample('48H').mean()

dd = pd.DataFrame(daily_solar)



dd.reset_index(inplace=True)

dd.columns = ['ds', 'y']



# dd['y'] = np.log(dd['y'])
m = Prophet(daily_seasonality=False)

m.fit(dd)



future = m.make_future_dataframe(periods=365*2)

forecast = m.predict(future)
m.plot(forecast, ylabel='MegaWatts of Wind Production')



plt.title('Forecasted Wind Production')

axes = plt.gca()

axes.set_xlim([pd.to_datetime('01/01/2010'), None])

print('Future Wind Production')
m.plot_components(forecast)
forecast.head()