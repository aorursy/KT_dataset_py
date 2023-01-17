import matplotlib.pyplot as plt
import pandas as pd
plt.style.use("seaborn-colorblind")
covid_19_india = pd.read_csv("../input/corona-indianstatesunionterritories/covid_19_india_eachMonth.csv")

covid_19_india_monthly = covid_19_india.pivot_table(values='Confirmed', index='State/UnionTerritory', columns='Date',
                                                    fill_value=0)


fig, ax = plt.subplots()

ax.bar(covid_19_india_monthly.index, covid_19_india_monthly['6-Jun'], label='June')
ax.bar(covid_19_india_monthly.index, covid_19_india_monthly['8-Aug'], bottom=covid_19_india_monthly['6-Jun'],
       label='Aug')

ax.set_xticklabels(covid_19_india_monthly.index, rotation=90)
ax.set_ylabel('Covid-19 Confirmed Cases')
ax.set_title('Covid-19 cases across States/UnionTerritories in India')

ax.legend()

plt.show()