from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

x = np.linspace(0, 3, 60)
y = np.exp(x)
plt.style.use('ggplot')
f, ax = plt.subplots(figsize=(14,6))

plt.bar(x, y, align='center', width=0.03)

# Imports
import pandas as pd
import matplotlib.dates as mdates 
import matplotlib.ticker as tkr
from datetime import datetime
import datetime as dt

# Dados de Óbitos das Secretarias Estaduais de Saúde do dia 12 de Junho
covid2 = pd.read_csv('../input/corona-virus-brazil/brazil_covid19.csv')
covid2 = covid2.sort_values(['state', 'region', 'date'])

# Create a new column with new deaths/day.
covid2['new_deaths'] = np.where(covid2.state.eq(covid2.state.shift()), covid2.deaths.diff(), 1)

# Invert date format
covid2["date"] = pd.to_datetime(covid2["date"]).dt.strftime('%d/%m/%Y')

# Group by day and Aggregate by adding each days` deaths 
covid2 = covid2.groupby('date').agg({'new_deaths': 'sum'})
covid2 = covid2.groupby('date').sum().reset_index()
covid2.head()

# Transform dates into an array of tuples to include on y-axis
x = covid2['date']
xdates = [dt.datetime.strptime(i,'%d/%m/%Y') for i in x]
y = covid2['new_deaths'].values

# Grab last day and last 24hs deaths
now = datetime.now()
last_day = now.strftime('%d/%m')




# Plot
plt.style.use('ggplot')
f, ax = plt.subplots(figsize=(16,8), dpi=300)

# Major ticks every 50, minor ticks every 10
major_ticks = np.arange(0, 1500, 200)
minor_ticks = np.arange(0, 1500, 50)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

# And a corresponding grid
ax.grid(which='minor', alpha=0.5)
ax.grid(which='major', alpha=0.8)

plt.bar(xdates,y, width=0.7, label="Óbitos por data da Notificação")
plt.legend(loc="upper left")
plt.title("Mortos por COVID19 de acordo com a data da Notificação", fontweight='bold')
plt.ylabel('Óbitos', fontsize=10)
plt.setp(plt.gca().xaxis.get_majorticklabels(),rotation=90)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))

f.text(0.5, 0.8, f'TOTAL: {int(covid2.new_deaths.sum())}', horizontalalignment='center', bbox={'alpha': 0.1, 'pad': 6})
f.text(0.12, 0.01, "Fonte: Secretarias Estaduais de Saúde {last_day}", fontsize=10)
f.text(0.9, 0.01, "Código: www.kaggle.com/fabianofpena/covid-19-no-brasil/", ha="right", fontsize=10)

plt.savefig('filename.png')
plt.show()

""" Dia 27 de Junho """
# Fonte: https://data.brasil.io/dataset/covid19/_meta/list.html
# Dados de Óbitos dos Cartórios
obitos2 = pd.read_csv("../input/cartoriosjun/obito_cartorio27jun.csv", sep=",", error_bad_lines=False)

# Check dtypes
obitos2.dtypes



# Convert to Datetime
obitos2["date"] = pd.to_datetime(obitos2["date"]).dt.strftime('%d/%m/%Y')
obitos2['date'] = pd.to_datetime(obitos2.date, format='%d/%m/%Y')
obitos2.describe()
# Group by day and Aggregate by adding each days` deaths 
agg2 = obitos2.groupby('date').agg({'new_deaths_covid19': 'sum'})
agg2 = agg2.groupby('date').sum().reset_index()

# Drop days without death
agg2 = agg2[agg2.new_deaths_covid19 != 0.0]

# Filter by dates before 8th Mar
agg2 = agg2[agg2['date'] >= '2020-03-08']

print(agg2.isna().sum())
agg2.head()
# Total accumulated deaths
agg2['new_deaths_covid19'].sum()

""" Dia 26 de Junho """
# Dados de Óbitos dos Cartórios dia 26 de Junho
obitos = pd.read_csv("../input/cartoriosjun/obito_cartorio26jun.csv",
                   sep=",", error_bad_lines=False)


# Check dtypes
obitos.dtypes

# Convert to Datetime
obitos["date"] = pd.to_datetime(obitos["date"]).dt.strftime('%d/%m/%Y')
obitos['date'] = pd.to_datetime(obitos.date, format='%d/%m/%Y')


# Group by day and Aggregate by adding each days` deaths 
agg = obitos.groupby('date').agg({'new_deaths_covid19': 'sum'})
agg = agg.groupby('date').sum().reset_index()


#Drop days without death
agg = agg[agg.new_deaths_covid19 != 0.0]

# Filter by dates before 8th Mar
agg = agg[agg['date'] >= '2020-03-08']


# Extract last 24hs data
agg2['diff'] = agg2['new_deaths_covid19'].subtract(agg['new_deaths_covid19'], fill_value=0)

# Check deaths confirmed in the last 24hs
agg2['diff'].sum()
# Transform datetime Series into an object
agg2["date"] = pd.to_datetime(agg2["date"]).dt.strftime('%d/%m/%Y')

# Grab axis and transform dates into an array of tuples
x = agg2['date']
xdates = [dt.datetime.strptime(i,'%d/%m/%Y') for i in x]
y = agg2['new_deaths_covid19'].values
y = y.astype(int)
y2 = agg2['diff'].values
y2 = y2.astype(int)

# Grab last day and last 24hs deaths
now = datetime.now()
last_day = now.strftime('%d/%m')
last_deaths = int(agg2['diff'].sum())
prev_death = int(agg2['new_deaths_covid19'].sum()-agg2['diff'].sum())

# Add Labels on the top of the bars
def bar_label(bars):
    for bar in bars:
        height = bar.get_height()
        if height == 0:
            continue
        ax.annotate('{}'.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + height),
            xytext=(0, 3),  # 3 points vertical offset
            textcoords="offset points",
            ha='center', va='bottom', fontsize=8, rotation=90)
# Plot
plt.style.use('ggplot')
f, ax = plt.subplots(figsize=(16,8), dpi=300)

# Texts
f.text(0.5, 0.8, f'TOTAL: {int(agg2.new_deaths_covid19.sum())}', horizontalalignment='center', bbox={'alpha': 0.1, 'pad': 6})
f.text(0.125, 0.01, f"Fonte: Portal Transparência Cartórios {last_day}", fontsize=10)
f.text(0.9, 0.01, "Código: www.kaggle.com/fabianofpena/covid-19-no-brasil/", ha="right", fontsize=10)

# Major ticks every 200, minor ticks every 100
major_ticks = np.arange(0, 1001, 200)
minor_ticks = np.arange(0, 1001, 50)

ax.set_yticks(major_ticks)
ax.set_yticks(minor_ticks, minor=True)

# And a corresponding grid
ax.grid(which='minor', alpha=0.4)
ax.grid(which='major', alpha=0.9)

plt.title("Óbitos por COVID-19")

# ax.plot(y[:len(ma)], ma, color="#dd0000")
ax.bar(xdates,y, width=0.7, color='darkorange', alpha=0.9, label="Óbitos Anteriores")
#ax.bar(xdates,list(y2.astype(int)), width=0.7, color='steelblue', bottom=y, label="Óbitos confirmados nas últimas 24 horas")
top_bar = ax.bar(xdates,list(y2), width=0.7, color='steelblue', bottom=y, label="Óbitos confirmados nas últimas 24 horas")
bar_label(top_bar)

ax.legend((f'Óbitos anteriores: {prev_death}',
           f"Óbitos Reportados no dia {last_day}: {last_deaths}"), loc='upper left')

plt.ylim(0,agg2.new_deaths_covid19.max()*1.2)
plt.ylabel('Óbitos', fontsize=12)
plt.xlabel('Data do Óbito', fontsize=12, labelpad=10)
plt.setp(plt.gca().xaxis.get_majorticklabels(),rotation=90)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=2))

plt.show()