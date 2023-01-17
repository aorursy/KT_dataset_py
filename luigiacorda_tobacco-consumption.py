# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
tobacco = pd.read_csv('../input/tobacco.csv')

tobacco.set_index('Year', inplace = True)

tobacco
tobacco.dtypes
tobacco['Smoke everyday'] = tobacco['Smoke everyday'].str[:-1].astype(float)

tobacco['Smoke some days'] = tobacco['Smoke some days'].str[:-1].astype(float)

tobacco['Former smoker'] = tobacco['Former smoker'].str[:-1].astype(float)

tobacco['Never smoked'] = tobacco['Never smoked'].str[:-1].astype(float)
tobacco.dtypes
tobacco
tobacco.corr()
tobacco[['Smoke everyday', 'Never smoked']].plot(kind = 'scatter', x = 0, y = 1)

plt.show()
avg_tobacco = tobacco.groupby('State')[['Smoke everyday', 'Smoke some days', 'Former smoker', 'Never smoked']].mean()

avg_tobacco
avg_tobacco[['Smoke everyday', 'Smoke some days']].sort_values(['Smoke everyday', 'Smoke some days'], ascending = True).tail(10).plot(kind = 'barh')

plt.show()
avg_tobacco[['Smoke everyday', 'Smoke some days']].sort_values(['Smoke everyday', 'Smoke some days'], ascending = False).tail(10).plot(kind = 'barh')

plt.show()
tobacco[(tobacco.State == 'Kentucky')].sort_index().plot()

plt.show()
tobacco[(tobacco.State == 'Guam')].sort_index().plot()

plt.show()
tobacco[(tobacco.State == 'West Virginia')].sort_index().plot()

plt.show()
tobacco[(tobacco.State == 'Tennessee')].sort_index().plot()

plt.show()
tobacco[(tobacco.State == 'Missouri')].sort_index().plot()

plt.show()
tobacco[tobacco.State == 'Virgin Islands'].sort_index().plot()

plt.show()
tobacco[tobacco.State == 'Puerto Rico'].sort_index().plot()

plt.show()
tobacco[tobacco.State == 'Utah'].sort_index().plot()

plt.show()
tobacco[tobacco.State == 'California'].sort_index().plot()

plt.show()
tobacco[tobacco.State == 'District of Columbia'].sort_index().plot()

plt.show()