import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

x =  pd.read_csv('/kaggle/input/data-yield-curve/Data Yield Curve.csv')
x.head(5)
x.set_index('Date')

x.index = pd.to_datetime(x['Date'])

x.head(5)
x = x.groupby(pd.Grouper(freq='A')).mean()

x.head(5)
x.loc[x['SPREAD'] <= 0, 'Status'] = 'Bad' 

x.loc[x['SPREAD'] > 0, 'Status'] = 'Good'

x.head(5)
x.Status.value_counts().loc['Bad']
y = x.SPREAD.sort_values(ascending=True)

y.head(6)
x.SPREAD.idxmin()
x.SPREAD.plot(ylim = 0)