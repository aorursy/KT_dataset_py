# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input/database.csv"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
shooting_data = pd.read_csv('../input/database.csv')

shooting_data.head()
shooting_data.loc[shooting_data.flee != 'Not fleeing', 'fleeing'] = 'flee'

shooting_data.loc[shooting_data.flee == 'Not fleeing', 'fleeing'] = 'not flee'

shooting_data.loc[shooting_data.armed != 'unarmed', 'armed status'] = 'armed'

shooting_data.loc[shooting_data.armed == 'unarmed', 'armed status'] = 'unarmed'

shooting_data.loc[shooting_data.race != 'W', 'race'] = 'non-white'

shooting_data.loc[shooting_data.race == 'W', 'race'] = 'white'

shooting_data.loc[shooting_data.threat_level != 'attack', 'threat'] = 'not attack'

shooting_data.loc[shooting_data.threat_level == 'attack', 'threat'] = 'attack'

shooting_data.head()
dataframes = []

for param in ['armed status', 'threat', 'fleeing']:

    shooting_data_grouped = shooting_data.groupby(by = ['state', 'race', param])

    shooting_data_sum = shooting_data_grouped['id'].apply(lambda x: x.count())

    shooting_data_sum = shooting_data_sum.reset_index()

    shooting_data_sum.rename(columns={'id': 'count', param: 'status'}, inplace=True)

    dataframes.append(shooting_data_sum)

shooting_data_stacked = pd.concat(dataframes, ignore_index=True)

shooting_data_stacked.head()
shooting_data_groupedbyrace = shooting_data.groupby(by = ['state', 'race'])

shooting_data_race = shooting_data_groupedbyrace['id'].apply(lambda x: x.count())

shooting_data_race = shooting_data_race.reset_index()

shooting_data_race.rename(columns={'id': 'total'}, inplace=True)

shooting_data_race.head()
shooting_data2plot = pd.merge(shooting_data_stacked, shooting_data_race, on=['state', 'race'], 

                              how='left')

shooting_data2plot.loc[:, 'fraction'] = shooting_data2plot['count']/shooting_data2plot['total']

shooting_data2plot
shooting_data2plot[np.isnan(shooting_data2plot.fraction)]
seaborn.boxplot(x = 'status', y = 'fraction', hue = 'race', 

                data = shooting_data2plot[shooting_data2plot.status.isin(['unarmed', 'not attack', 'not flee'])],

                 )
shooting_data_state = shooting_data_race.pivot(index = 'state', 

                       columns = 'race', 

                       values = 'total').reset_index()

shooting_data_state = shooting_data_state.fillna(0)

for status in ['armed', 'attack', 'flee']:

    shooting_bias = shooting_data2plot[shooting_data2plot.status == status].pivot(index = 'state', 

                                                               columns = 'race', 

                                                               values = 'fraction')

    shooting_bias = shooting_bias.reset_index()

    shooting_bias.loc[:, 'bias_' + status] = shooting_bias['non-white'] - shooting_bias['white']

    shooting_data_state = pd.merge(shooting_data_state, 

                         shooting_bias.loc[:, ['state', 'bias_' + status]], 

                         on='state', 

                         how='outer')

shooting_data_state.loc[:, 'total'] = shooting_data_state['non-white'] + shooting_data_state['white']

shooting_data_state.loc[:, 'fraction_white'] = shooting_data_state['white']/shooting_data_state['total']

shooting_data_state.loc[:, 'fraction_non_white'] = shooting_data_state['non-white']/shooting_data_state['total']
seaborn.pairplot(shooting_data_state, 

                 hue="state", 

                 vars=['bias_armed', 'bias_attack', 'bias_flee',

                  'total', 'fraction_non_white'], 

                 size = 2.5,

                 aspect = 1.2,

                 plot_kws=dict(s = 30, marker = 's'))
seaborn.distplot(shooting_bias.bias.values[~np.isnan(shooting_bias.bias.values)], 

                 bins=20, kde=False, rug=True)