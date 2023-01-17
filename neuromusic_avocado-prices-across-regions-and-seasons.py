import pandas as pd
avocados = pd.read_csv(
    '../input/avocado.csv',
    index_col=0,
)
avocados.head()
avocados['Date'] = pd.to_datetime(avocados['Date'])
avocados.head()
import seaborn as sns
sns.set_style('white')
mask = avocados['type']=='conventional'
g = sns.factorplot('AveragePrice','region',data=avocados[mask],
                   hue='year',
                   size=8,
                   aspect=0.6,
                   palette='Blues',
                   join=False,
              )
order = (
    avocados[mask & (avocados['year']==2018)]
    .groupby('region')['AveragePrice']
    .mean()
    .sort_values()
    .index
)
g = sns.factorplot('AveragePrice','region',data=avocados[mask],
                   hue='year',
                   size=8,
                   aspect=0.6,
                   palette='Blues',
                   order=order,
                   join=False,
              )
regions = ['PhoenixTucson', 'Chicago']
mask = (
    avocados['region'].isin(regions)
    & (avocados['type']=='conventional')
)
avocados['Month'] = avocados['Date'].dt.month
avocados[mask].head()
g = sns.factorplot('Month','AveragePrice',data=avocados[mask],
               hue='year',
               row='region',
               aspect=2,
               palette='Blues',
              )
