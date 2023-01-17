# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



#import os

#print(os.listdir("../input"))



c = pd.read_csv('../input/Country.csv')

i = pd.read_csv('../input/Indicators.csv')
N_max = 500

N = 25

np.random.seed(634582193) # random value by random.org

rand_y = np.random.randint(1970,2010, N_max)

rand_c = np.random.choice(c.ShortName,N_max)

print(*enumerate(zip( rand_y[:N],rand_c[:N])), sep='\n')
world_gdp = i[(i.CountryCode == 'WLD') & (i.IndicatorName=='GDP at market prices (constant 2005 US$)')].set_index('Year').loc[:,('Value',)]

c_gdp = i[i.IndicatorName=='GDP at market prices (constant 2005 US$)'].set_index(['CountryName','Year']).loc[:,('Value',)]
d = pd.DataFrame(columns = ['i','country','y1','y5','experience'],

                 data = [(0,'Rwanda', 1994, 1999, 1000)

                         ,(1,'Finland',2000, 2005, 13461)

                         ,(2,'St. Lucia', 1982, 1987, 176)

                         ,(3,'Sierra Leone', 1985, 1990, 42200)

                         ,(4, 'Paraguay', 1993, 1998, 5000)

                ])

d = d.merge(

    world_gdp.rename(columns={'Value':'wgdp1'}), 'left', left_on='y1', right_index=True

).merge(

    world_gdp.rename(columns={'Value':'wgdp5'}), 'left', left_on='y5', right_index=True

).merge(

    c_gdp.rename(columns={'Value':'cgdp1'}), 'left', left_on=['country','y1'], right_index=True

).merge(

    c_gdp.rename(columns={'Value':'cgdp5'}), 'left', left_on=['country','y5'], right_index=True

)

d['country_change_gdp'] = d.cgdp5-d.cgdp1

d['world_change_gdp'] = d.wgdp5-d.wgdp1

d['norm_change_gdp'] = d.country_change_gdp / d.world_change_gdp * 1e4

d['log_experience'] = np.log(d.experience)

d
d.plot.scatter('log_experience','norm_change_gdp')

x = [min(d.log_experience), max(d.log_experience)]

plt.plot(x, np.poly1d(np.polyfit(d.log_experience,d.norm_change_gdp,1))(x), '-g');