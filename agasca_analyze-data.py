import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import calendar
import matplotlib.ticker as tkr
from datetime import datetime, date
import seaborn as sns
dt = date(2018, 8, 1)
tickets = pd.read_csv('../input/Sales.csv', parse_dates=[1])
tickets.dataframeName = 'Sales tickets'
nRow, nCol = tickets.shape
tickets = tickets[(tickets['Hora'].dt.month == dt.month) & (tickets['Hora'].dt.year == dt.year)]
tickets = tickets.assign(
    datetime = tickets['Hora'].map(lambda x: x.replace(microsecond=0,second=0,minute=0))
)
per_hour = (tickets
            .groupby(tickets.datetime.dt.hour)
            .agg({'Folio': 'count', 
                  'Total': ['sum', 'mean']}))
per_hour.columns = per_hour.columns.droplevel(0)

per_hour = per_hour.assign(
    sales_mean = (tickets
                  .groupby([tickets.datetime.dt.day, 
                            tickets.datetime.dt.hour])
                  .Total.sum()
                  .unstack()
                  .mean()),
    tickets_mean = (tickets
                    .groupby([tickets.datetime.dt.day, 
                              tickets.datetime.dt.hour])
                    .Folio.count()
                    .unstack()
                    .mean()),    
    date = dt
)

per_hour = per_hour[['tickets_mean','sales_mean','mean','count','sum','date']]
per_day_hour = (tickets
                .groupby('datetime')
                .agg({'Total': 'sum', 
                      'Folio':'count'})
                .reset_index())

per_day_hour = per_day_hour.assign(
    weekday = per_day_hour['datetime'].dt.weekday,
    hour = per_day_hour['datetime'].dt.hour,
)
per_day_hour = (per_day_hour
                .groupby(['weekday', 'hour'])
                ['Folio', 'Total'].mean()
                .reset_index())
tickets['Total'].describe()
tickets['Total'].plot.box(showfliers=False)
normalize = lambda x, r_max, r_min, t_max = 1, t_min = 0.1: (x - r_min) / (r_max - r_min) * (t_max - t_min) +  t_min
normalized = per_day_hour['Folio'].map(lambda x: normalize(x, 
                                                           per_day_hour['Folio'].max(), 
                                                           per_day_hour['Folio'].min()))
ax = per_day_hour.plot.scatter('weekday', 'hour',
                               s=normalized*700,
                               alpha=0.5,
                               grid=True,
                               xticks=per_day_hour['weekday'].unique())
ax.xaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: list(calendar.day_name)[int(x)]))
df = (tickets
      .groupby([tickets.datetime.dt.day, tickets.datetime.dt.hour])
      .agg({'Total': 'sum', 
            'Folio':'count'})
      .reset_index(level=1))
df.plot.hexbin(x='datetime', y='Total', gridsize=20, grid=True, figsize=(17,10))
#df.plot.scatter(x='datetime', y='Folio', figsize=(25,10))
per_month = (per_hour
             .reset_index()
             .pivot_table(values=['tickets_mean','sales_mean','mean','count','sum'], 
                          index=['date', 'datetime']))
per_month = per_month[['tickets_mean','sales_mean','mean','count','sum']]
titles = ['Clientes promedio','Venta promedio', 'Ticket promedio', 'Total de Clientes', 'Total de Venta']
_, axes = plt.subplots(2,3)
axes = axes.flatten()
for date in per_month.index.levels[0]:
    for i, col in enumerate(per_month.columns):
        per_month.xs(date)[col].plot(figsize=(17,9),
                                     sharex=False, 
                                     legend=True,
                                     title=titles[i],
                                     grid=True,
                                     xticks=per_month.index.levels[1],
                                     ax=axes[i],
                                     rot=20)
for ax in axes:
    months_names = [month.strftime('%B %Y') for month in per_month.index.levels[0]]
    ax.legend(months_names)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: format(int(x), ',')))
    ax.set_xlabel('Hora')
plt.subplots_adjust(top = 0.95, bottom = 0.1, right=0.99,left=0.03, wspace=0.2, hspace=0.3)