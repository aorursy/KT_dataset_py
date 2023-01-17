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
import matplotlib.pyplot as plt

import ipywidgets as widgets





df = pd.read_csv('/kaggle/input/coronavirusdataset/TimeProvince.csv', index_col='date')





def pick_data(province):

    df_ggd = df[(df['province']==province)][:]

    df_ggd.drop(['time', 'province', 'released', 'deceased'], axis=1, inplace=True)



    df_ggd['difference'] = df_ggd.diff(axis=0).fillna(0)



    fig = plt.figure(figsize=(20,5)) # Create matplotlib figure



    ax = fig.add_subplot(111) # Create matplotlib axes

    ax2 = ax.twinx() # Create another axes that shares the same x-axis as ax.



    width = 0.4



    df_ggd.confirmed.plot(kind='bar', color='purple', ax=ax, width=width, position=1)

    df_ggd.difference.plot(kind='bar', color='red', ax=ax2, width=width, position=0)



    ax.set_ylabel('confirmed')

    ax2.set_ylabel('difference')



    ax.tick_params(axis='x', labelrotation=45)



    plt.show()



    #df_ggd.head(10)



#widget settings

province = widgets.Select(options=df['province'].unique().tolist(), rows=10, description='Province')

out = widgets.interactive_output(pick_data, {'province':province})

display(province, out)