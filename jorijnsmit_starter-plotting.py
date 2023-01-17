import numpy as np

import pandas as pd



import matplotlib.ticker as mtick
df = pd.read_csv('../input/euribor_daily_rates.csv', index_col=0, parse_dates=[0])



ax = df.plot(figsize=(12,7), colormap='Spectral')



# making it look good

ax.set_title('EURIBOR Daily Rates')

ax.set_xlabel('Date')

ax.set_ylabel('Rate')

ax.axhline(y=0, color='black', linewidth=1)

ax.yaxis.set_major_formatter(mtick.PercentFormatter())

ax.spines['top'].set_color('none')

ax.spines['right'].set_color('none')

ax.spines['bottom'].set_color('none')

ax.grid(axis='y', linestyle=':')
df.describe()