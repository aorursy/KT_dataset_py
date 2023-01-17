import numpy as np

import pandas as pd

import seaborn as sns

from pandas import DataFrame

import matplotlib.pyplot as plt



all_data=pd.read_csv('../input/GlobalTemperatures.csv', parse_dates=['dt'])

all_data=all_data.replace([np.inf, -np.inf], np.nan).dropna()



all_data['maxAvgTemp']=all_data['LandAverageTemperature']+all_data['LandAverageTemperatureUncertainty']

all_data['minAvgTemp']=all_data['LandAverageTemperature']-all_data['LandAverageTemperatureUncertainty']



all_data=all_data.groupby(all_data['dt'].map(lambda x: x.year)).mean().reset_index()

min_year=all_data['dt'].min()

max_year=all_data['dt'].max()
sns.set(style="whitegrid")

sns.set_color_codes("pastel")

_, ax=plt.subplots(figsize=(10, 6))



plt.plot(all_data['dt'], all_data['LandAverageTemperature'], color='black')

ax.fill_between(all_data['dt'], all_data['minAvgTemp'], all_data['maxAvgTemp'], color='b')



plt.xlim(min_year, max_year)

ax.set_title('Average global temperature including uncertainty')



plt.show()