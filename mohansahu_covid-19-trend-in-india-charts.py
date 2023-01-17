import pandas as pd

import matplotlib.pyplot as plt
df = pd.read_csv('https://api.covid19india.org/csv/latest/case_time_series.csv')
df.Date = df.Date + '2020'

df.Date = pd.to_datetime(df.Date)
dt = df['Date']

dc = df['Daily Confirmed']

dr = df['Daily Recovered']

dd = df['Daily Deceased']

Tc = df['Total Confirmed']

Tr = df['Total Recovered']

Td = df['Total Deceased']
plt.xkcd()

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, figsize=(20, 40), sharex=True)



ax1.plot(dt, dc / 10, 'b.-', label='Daily Confirmed 10x')

ax1.legend(fontsize=15)

ax1.plot(dt, dr / 10, 'g.-', label='Daily Recovered 10x')

ax1.legend(fontsize=15)

ax1.plot(dt, dd, 'r.-', label='Daily Deceased')

ax1.grid(True, linewidth=2)

ax1.set_title('Covid-19 Trend in India\nDaily Confirmed, Recovered & Deceased')

ax1.legend(fontsize=15)



ax2.plot(dt, dd, 'ro-', label='Daily Deceased')

ax2.grid(True, linewidth=1)

ax2.set_title('Daily Deceased')

ax2.legend(fontsize=15)



ax3.plot(dt, ((Tr/Tc) * 100) / 10, 'g.-', label='Recovery Rate 10x')

ax3.plot(dt, (Td/Tc) * 100, 'r.-', label='Mortality Rate')

ax3.grid(True, linewidth=2)

ax3.set_title('Recovery & Mortailty Rate (Cumulative)')

ax3.legend(fontsize=15)



ax4.plot(dt, ((dr/dc) * 100) / 10, 'g.-', label='Recovery Rate 10x')

ax4.plot(dt, (dd/dc) * 100, 'r.-', label='Mortality Rate')

ax4.grid(True, linewidth=2)

ax4.set_title('Recovery & Mortailty Rate (Daily)')

ax4.legend(fontsize=15)



ax5.plot(dt, Tc / 10, 'b.-', label='Confirmed 10x (Cumulative)')

ax5.plot(dt, Tr / 10, 'g.-', label='Recovered 10x (Cumulative)')

ax5.plot(dt, Td, 'r.-', label='Deceased (Cumulative)')

ax5.grid(linewidth=2)

ax5.set_title('Cumulative Trend')

ax5.legend(fontsize=15)



plt.savefig('covid_trend_in_india.png')

plt.show()