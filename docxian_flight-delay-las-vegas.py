import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import matplotlib.pyplot as plt

import seaborn as sns



from statsmodels.graphics.mosaicplot import mosaic



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# import data

df = pd.read_csv('../input/las-flight-arrivals-2019/LAS_Arrivals_2019.csv')
df.head()
df.info()
# distribution by origin airport

df.origin_airport.value_counts()
df.origin_airport.value_counts()[0:10].plot(kind='bar')

plt.grid()

plt.title('Origin Airport - Top 10')

plt.show()
# store top 5 airports for later

top5_airport = list(df.origin_airport.value_counts()[0:5].index)

top5_airport
# distribution by carrier

df.carrier_code.value_counts().plot(kind='bar')

plt.grid()

plt.title('Carrier')

plt.show()
# bivariate distribution airport/carrier (use only top 5 airports)

df_top5_airport = df[df.origin_airport.isin(top5_airport)]

plt.rcParams["figure.figsize"]=(12,8)

mosaic(df_top5_airport, ['origin_airport', 'carrier_code'], title='Distribution Airport [top 5] / Carrier')

plt.show()
# corresponding figures

pd.crosstab(df_top5_airport.carrier_code, df_top5_airport.origin_airport)
# another visualization

pd.crosstab(df_top5_airport.origin_airport, df_top5_airport.carrier_code).plot(kind='bar', stacked=True)

plt.grid()

plt.show()
# distribution by date

df.date.value_counts()
plt.figure(figsize=(12,4))

plt.hist(df.arrival_delay_minutes,100)

plt.grid()

plt.title('Arrival Delay [mins]')

plt.show()
df_origin_airport = df.groupby('origin_airport', as_index=False).agg(

    mean_delay = pd.NamedAgg(column='arrival_delay_minutes', aggfunc=np.mean),

    min_delay = pd.NamedAgg(column='arrival_delay_minutes', aggfunc=min),

    max_delay = pd.NamedAgg(column='arrival_delay_minutes', aggfunc=max),

    q10_delay = pd.NamedAgg(column='arrival_delay_minutes', aggfunc=lambda x:np.quantile(x,.1)),

    q90_delay = pd.NamedAgg(column='arrival_delay_minutes', aggfunc=lambda x:np.quantile(x,.9)))



df_origin_airport
plt.figure(figsize=(18,4))

plt.plot(df_origin_airport.origin_airport, df_origin_airport.mean_delay)

plt.grid()

plt.xticks(rotation=90)

plt.title('Mean delay by origin airport')

plt.show()
# select peaks

df_origin_airport_high = df_origin_airport[df_origin_airport.mean_delay > 20]

df_origin_airport_high
df_date = df.groupby('date', as_index=False).agg(

    mean_delay = pd.NamedAgg(column='arrival_delay_minutes', aggfunc=np.mean),

    min_delay = pd.NamedAgg(column='arrival_delay_minutes', aggfunc=min),

    max_delay = pd.NamedAgg(column='arrival_delay_minutes', aggfunc=max),

    q10_delay = pd.NamedAgg(column='arrival_delay_minutes', aggfunc=lambda x:np.quantile(x,.1)),

    q90_delay = pd.NamedAgg(column='arrival_delay_minutes', aggfunc=lambda x:np.quantile(x,.9)))



df_date
plt.figure(figsize=(18,4))

plt.plot(df_date.date, df_date.mean_delay)

plt.xticks(rotation=90)

plt.grid()

plt.title('Mean delay by date')

plt.show()
# select peaks

df_date_high = df_date[df_date.mean_delay > 40]

df_date_high
# Highest mean delay was on 04/29/2019. Let's have a closer look.

df_date_max = df[df.date=='04/29/2019']
# plot + compare with all delays

fig, ax = plt.subplots(2, sharex=True, figsize=(12,4*2))

fig.suptitle('Compare delays on peak day vs. all delays', fontweight='bold')

ax[0].hist(df_date_max.arrival_delay_minutes,100)

ax[0].set_title('Delays on 04/29/2019')

ax[0].grid()

ax[1].hist(df.arrival_delay_minutes,100)

ax[1].set_title('All Delays 2019')

ax[1].grid()

plt.show()
df_carrier = df.groupby('carrier_code', as_index=False).agg(

    mean_delay = pd.NamedAgg(column='arrival_delay_minutes', aggfunc=np.mean),

    min_delay = pd.NamedAgg(column='arrival_delay_minutes', aggfunc=min),

    max_delay = pd.NamedAgg(column='arrival_delay_minutes', aggfunc=max),

    q10_delay = pd.NamedAgg(column='arrival_delay_minutes', aggfunc=lambda x:np.quantile(x,.1)),

    q90_delay = pd.NamedAgg(column='arrival_delay_minutes', aggfunc=lambda x:np.quantile(x,.9)))



df_carrier
plt.figure(figsize=(8,4))

plt.plot(df_carrier.carrier_code, df_carrier.mean_delay)

plt.plot(df_carrier.carrier_code, df_carrier.q90_delay, linestyle='dashed')

plt.plot(df_carrier.carrier_code, df_carrier.q10_delay, linestyle='dashed')

plt.grid()

plt.title('Mean delay + 10%/90% quantile by carrier')

plt.show()
# violinplot

plt.figure(figsize=(8,8))

sns.violinplot(x='carrier_code', y='arrival_delay_minutes', data=df)

plt.grid()

plt.show()
plt.figure(figsize=(8,8))

sns.violinplot(x='carrier_code', y=np.log10(90+df.arrival_delay_minutes), data=df)

plt.grid()

plt.show()