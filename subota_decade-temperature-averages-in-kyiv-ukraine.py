import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



w = pd.read_csv('../input/kyiv-ukraine.csv')



kyiv = w[ w.LocationCode == 33345 ]



decades = (kyiv.Year - 5).round(-1)

kyivByDecades = kyiv.groupby(decades)



min_temp  = kyivByDecades['t_min_c'].min()

mean_temp = kyivByDecades['t_mean_c'].mean()

max_temp  = kyivByDecades['t_max_c'].max()

yearly_mean_temp = kyiv.groupby(['Year'])['t_mean_c'].mean()
ax = pd.DataFrame( [min_temp, max_temp, mean_temp] ).transpose().plot()



yearly_mean_temp.plot(axes = ax)



ax.grid( True )

ax.get_figure().set_size_inches(20, 15)