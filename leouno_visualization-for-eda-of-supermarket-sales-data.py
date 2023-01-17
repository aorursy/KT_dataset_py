# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
df = pd.read_csv('/kaggle/input/supermarket-sales/supermarket_sales - Sheet1.csv')

df['formatted_date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')

df
dfA = df[df['Branch'] == 'A']

dfB = df[df['Branch'] == 'B']

dfC = df[df['Branch'] == 'C']
dfASum = dfA.groupby('formatted_date').sum()

dfASum['Branch'] = 'A'

dfBSum = dfB.groupby('formatted_date').sum()

dfBSum['Branch'] = 'B'

dfCSum = dfC.groupby('formatted_date').sum()

dfCSum['Branch'] = 'C'



dfSum = pd.concat([dfASum, dfBSum, dfCSum])

dfSum.reset_index(inplace=True)





g = sns.relplot(x='formatted_date', y='Total', hue='Branch', kind='line', data=dfSum)

g.fig.autofmt_xdate()

g.fig.set_figheight(10)

g.fig.set_figwidth(30)
import shutil

shutil.copyfile('/kaggle/input/supermarket-sales/supermarket_sales - Sheet1.csv', '/kaggle/working/supermatket-sales-data.csv')
import os

os.chdir(r'/kaggle/working')



from IPython.display import FileLink

FileLink(r'supermatket-sales-data.csv')