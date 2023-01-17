# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/college-basketball-dataset/cbb.csv')
data
data['cw'] = data['W']/data['G']
data.info()
data.columns =data.columns.str.lower()

data.columns
data['postseason'] = data['postseason'].fillna('Not playoff')
data.pivot_table(index='team',values = 'w',

                 aggfunc='max').reset_index().sort_values(by='w'

                    ,ascending=False).head(10).plot.bar('team',figsize=[10,5],color='red',alpha=0.35)
data['adjoe'].hist(bins=50,figsize=[10,7],color='red',alpha=0.35)
data[data['postseason']=='Not_playoff']
data.pivot_table(index='team',values = 'adjoe',aggfunc='max').reset_index().sort_values(by='adjoe',ascending=False).head(10)
data.pivot_table(index='team',values = 'adjde',aggfunc='max').reset_index().sort_values(by='adjde',ascending=False).head(10)


data.pivot_table(index='year',columns='conf',values = 'w'

                 ,aggfunc='sum').boxplot(figsize=[15,7]).set_title('Multiple average annual conference winnings.')

plt.ylim(50,300)
data['barthag'].hist(bins=80,figsize=[10,5],color='red',alpha=0.25)
data.pivot_table(index='year',values = 'cw').plot.line(figsize=[10,2],color='red',alpha=0.35)
data.query('year in [2016,2017]').pivot_table(index='year',values=['adjoe','adjde'],aggfunc=['mean','median']).reset_index()
data.query('year in [2016,2017]').pivot_table(index='year',values=['g','w','cw']).reset_index()
data.query('postseason=="Champions"').sort_values(by='year')
data.pivot_table(index = 'postseason'

        ,values = 'cw').sort_values('cw',

        ascending='postseason').plot.barh(fill=True,figsize=[10,2],alpha=0.35,color='red',grid=True)

data.pivot_table(index = 'postseason'

        ,values = 'adjoe').sort_values('adjoe',

        ascending='postseason').plot.barh(fill=True,figsize=[10,2],alpha=0.35,color='red',grid=True)

data.pivot_table(index = 'postseason'

        ,values = 'adjde').sort_values('adjde',

        ascending='postseason').plot.barh(fill=True,figsize=[10,2],alpha=0.35,color='red',grid=True)
data.boxplot(['barthag','cw'])