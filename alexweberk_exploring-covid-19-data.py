import pandas as pd

import matplotlib.pyplot as plt

plt.style.use('seaborn')

data = pd.read_csv("/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv")

data.replace('China', "Mainland China", inplace=True)

data.Date = pd.to_datetime(data.Date)

metrics = ['Confirmed', 'Deaths', 'Recovered']

data['date'] = data.Date.apply(lambda x: x.strftime('%Y-%m-%d'))
tots = data.groupby('date')[metrics].sum()

tots.plot(style='o-')

plt.title("Covid-19: Totals")
tots = data.groupby('date')[metrics].sum()

recovered_to_confirmed_ratio = tots['Recovered'] / tots['Confirmed']

deaths_to_confirmed_ratio = tots['Deaths'] / tots['Confirmed']



plt.figure()

plt.plot(recovered_to_confirmed_ratio, 'o-', label='Recovered-to-Confirmed Ratio')

plt.plot(deaths_to_confirmed_ratio, 'o-', label='Deaths-to-Confirmed Ratio')

plt.title("Recovered-to-confirmed ratio growing, while deaths-to-confirmed ratio stable at around 2.5%")

plt.legend()

plt.xticks(rotation=90)

plt.show()
tots.pct_change().rolling(5).mean().plot(style='o-')

plt.title("Daily growth rate in confirmed cases is slowing")
tbl = data.pivot_table(index='date', columns='Country', values='Confirmed', aggfunc='sum') 

tbl = tbl.fillna(0).T.sort_values('2020-02-17', ascending=False)

tbl['2020-02-17'].plot(kind='bar', logy=False)

plt.title("Total Confirmed is mostly concentrated in Mainland China")

plt.show()
tbl['2020-02-17'][:20].plot(kind='bar', logy=True)

plt.title("Total Confirmed (Log Scale to get a better look)")

plt.show()
# Let's look at specific countries, China and Japan.

china = data[data.Country == 'Mainland China']

japan = data[data.Country == 'Japan']
china_table = china.pivot_table(index='date', columns='Province/State', values='Confirmed', aggfunc='sum').T.sort_values('2020-02-17', ascending=False)

china_table[:10].T.plot(style='o-', logy=False)

plt.suptitle("Top 10 Chinese Provinces by Number of Confirmeds")

plt.title("Hubei dominates cases")
china_table[1:11].T.plot(style='o-', logy=False)

plt.suptitle("Top 2~11 Chinese Provinces by Number of Confirmeds, Excluding Hubei Province")

plt.title("Growth rate appears to be slowing in all regions")
china_table[:10].T.pct_change().rolling(5).mean().plot(style='o-', logy=False)

plt.title("Growth rates of Confirmeds across top 10 provinces")

plt.ylim(-0.2, 1)
japan.groupby('date')[metrics].sum().plot(style='o-')

plt.suptitle("Japan: Num Confirmed, Deaths, and Recovereds")

plt.title("Confirmeds is increasing")
japan.set_index('date')[metrics].pct_change().rolling(3).mean().plot(style='o-')

plt.title("Daily growth rate, smoothed")
# Assuming Sick = Confirmed - Deaths - Recovered:

tots['Sick'] = tots['Confirmed'] - tots['Deaths'] - tots['Recovered']

tots[['Recovered', 'Deaths', 'Sick']].plot.bar(stacked=True)

tots['Confirmed'].plot(style='o-')

plt.title("Breakdown")

plt.xticks(rotation=90)
not_china = data[data['Country'] != 'Mainland China'].groupby('date')[metrics].sum()

not_hubei = data[(data['Province/State'] != "Hubei") & (data['Country'] == 'Mainland China')].groupby('date')[metrics].sum()
not_china.plot(style='o-')

plt.title("Still in its growth phase outside of China")
not_china.pct_change().plot(style='o-')
not_china['Recovered/Confirmed'] = not_china['Recovered'] / not_china['Confirmed']

not_china['Deaths/Confirmed'] = not_china['Deaths'] / not_china['Confirmed']

not_china[['Recovered/Confirmed', 'Deaths/Confirmed']].plot(style='o-')

plt.title("Deaths/Confirmed is a lot lower outside of China")
not_hubei.plot(style='o-')

plt.title("Cases are still growing...")
not_hubei.pct_change().plot(style='o-')

plt.title("But at a slower pace...")
not_hubei['Recovered/Confirmed'] = not_hubei['Recovered'] / not_hubei['Confirmed']

not_hubei['Deaths/Confirmed'] = not_hubei['Deaths'] / not_hubei['Confirmed']

not_hubei[['Recovered/Confirmed', 'Deaths/Confirmed']].plot(style='o-')

plt.title("Deaths/Confirmed is a lot lower when Hubei is excluded")
hubei = data[data['Province/State'] == 'Hubei'][metrics]

hubei
hubei.plot(style='o-')

plt.title("Hubei Numbers")
hubei['Recovered/Confirmed'] = hubei['Recovered'] / hubei['Confirmed']

hubei['Deaths/Confirmed'] = hubei['Deaths'] / hubei['Confirmed']

hubei[['Recovered/Confirmed', 'Deaths/Confirmed']].plot(style='o-')

plt.title("Hubei's Death/Confirmed Ratio is high")