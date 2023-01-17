import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
url = "../input/t-gondii-incidence-country/tgondii_incidence_w_country.xlsx"

tdf = pd.read_excel(url, index_col=0)

tdf.head(3)
url = "../input/corona-virus-report/covid_19_clean_complete.csv"

cdf = pd.read_csv(url)



cdf = cdf[cdf['Province/State'].isna()]

cdf[cdf['Country/Region'] == 'France'].head(3)
latest_dates = cdf.groupby('Country/Region').Date.max().reset_index()

df = pd.merge(cdf, latest_dates, how='inner', on=['Country/Region', 'Date'])

df = df[['Country/Region', 'Confirmed', 'Deaths', 'Recovered']].rename({'Country/Region':'Country'}, axis=1).set_index('Country')

df['cfr'] = df.Deaths / df.Confirmed

df['cfr2'] = df.Deaths / df.Recovered

df['crr'] = df.Recovered / df.Confirmed
# df = df[df.Confirmed > 5000]

# df = df[df.cfr2 < 6]

combined = df[df.Confirmed > 5000].join(tdf, how='inner').dropna()[['cfr', 'cfr2', 'crr', 'Prevalence_pct']].rename({'Prevalence_pct': 'TGondii_Prevalence'}, axis=1)
fig, axs = plt.subplots(1,4, figsize=(20,8))



combined.plot(y='cfr', x='TGondii_Prevalence', style='ro', ax=axs[0])

combined.plot(y='cfr2', x='TGondii_Prevalence', style='yo', ax=axs[1])

combined.plot(y='cfr2', x='cfr', style='yo', ax=axs[2])

combined.plot(y='crr', x='TGondii_Prevalence', style='go', ax=axs[3])
c = ['Argentina', 'Australia', 'Austria', 'Belgium', 'Brazil', 'China', 'Colombia', 'Croatia', 'Czech Republic', 'Denmark', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'India', 'Indonesia', 'Ireland', 'Israel', 'Italy', 'Jamaica', 'Japan', 'Jordan', 'Malaysia', 'Mexico', 'Netherlands', 'New Zealand', 'Norway', 'Peru', 'Poland', 'Portugal', 'Romania', 'Singapore', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Taiwan', 'Thailand', 'Turkey', 'UK', 'United States of America', ]

t = [0.46209256, 0.22992856, 0.35310518, 0.38609525, 0.60372601, 0.16466979, 0.48599131, 0.34172129, 0.27196209, 0.21737301, 0.16189364, 0.44073627, 0.41873487, 0.23408792, 0.56081994, 0.29972293, 0.53489357, 0.2155291, 0.167, 0.21695008, 0.521, 0.123, 0.37407461, 0.3847282, 0.26302442, 0.2985, 0.2720488, 0.086127605, 0.329, 0.36318183, 0.17, 0.576, 0.12806508, 0.29895846, 0.23411083, 0.13040469, 0.26810686, 0.26009428, 0.10999422, 0.42583808, 0.087382118, 0.159, ]



tdf2 = pd.DataFrame({'country': c, 'TG incidence': t}).set_index('country')
v = tdf2.join(df[['cfr', 'cfr2', 'crr']], how='inner')



fig, ax = plt.subplots()

x, y = 'TG incidence', 'cfr2'

v.plot(x=x, y=y, kind='scatter', figsize=(18,10), grid=True, ax=ax)

for i in range(len(v[x])):

    ax.annotate(v.iloc[i].name, (v[x].iloc[i], v[y].iloc[i]))

v[v.cfr>.07].sort_values(by='cfr2', ascending=False)