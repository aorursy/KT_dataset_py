# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import warnings

plt.style.use('ggplot')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

warnings.filterwarnings('ignore')

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
confirmed=pd.read_csv("/kaggle/input/covid2019/confirmed.csv",header=0, parse_dates=[0], index_col=0, squeeze=True)

death=pd.read_csv("/kaggle/input/covid2019/death.csv",header=0, parse_dates=[0], index_col=0, squeeze=True)

recoveries=pd.read_csv("/kaggle/input/covid2019/recoveries.csv",header=0, parse_dates=[0], index_col=0, squeeze=True)
fig,ax=plt.subplots(1,3,figsize=(15,5))

fig.tight_layout()

c=death.columns[0]

ax[0].set_title("confirmed "+c)

confirmed[c].plot(ax=ax[0])

ax[1].set_title("death "+c)

death[c].plot(ax=ax[1])

ax[2].set_title("recoveries "+c)

recoveries[c].plot(ax=ax[2], color='green')

plt.show()
fig,ax=plt.subplots(1,3,figsize=(15,5))

fig.tight_layout()

c=death.columns[1]

ax[0].set_title("confirmed "+c)

confirmed[c].plot(ax=ax[0])

ax[1].set_title("death "+c)

death[c].plot(ax=ax[1])

ax[2].set_title("recoveries "+c)

recoveries[c].plot(ax=ax[2], color='green')

plt.show()
fig,ax=plt.subplots(1,3,figsize=(15,5))

fig.tight_layout()

c=death.columns[2]

ax[0].set_title("confirmed "+c)

confirmed[c].plot(ax=ax[0])

ax[1].set_title("death "+c)

death[c].plot(ax=ax[1])

ax[2].set_title("recoveries "+c)

recoveries[c].plot(ax=ax[2], color='green')

plt.show()
#
fig,ax=plt.subplots(1,3,figsize=(15,5))

fig.tight_layout()

c=death.columns[4]

ax[0].set_title("confirmed "+c)

confirmed[c].plot(ax=ax[0])

ax[1].set_title("death "+c)

death[c].plot(ax=ax[1])

ax[2].set_title("recoveries "+c)

recoveries[c].plot(ax=ax[2], color='green')

plt.show()
fig,ax=plt.subplots(1,3,figsize=(15,5))

fig.tight_layout()

c=death.columns[5]

ax[0].set_title("confirmed "+c)

confirmed[c].plot(ax=ax[0])

ax[1].set_title("death "+c)

death[c].plot(ax=ax[1])

ax[2].set_title("recoveries "+c)

recoveries[c].plot(ax=ax[2], color='green')

plt.show()
fig,ax=plt.subplots(1,3,figsize=(15,5))

fig.tight_layout()

c=death.columns[6]

ax[0].set_title("confirmed "+c)

confirmed[c].plot(ax=ax[0])

ax[1].set_title("death "+c)

death[c].plot(ax=ax[1])

ax[2].set_title("recoveries "+c)

recoveries[c].plot(ax=ax[2], color='green')

plt.show()
fig,ax=plt.subplots(1,3,figsize=(15,5))

fig.tight_layout()

c=death.columns[7]

ax[0].set_title("confirmed "+c)

confirmed[c].plot(ax=ax[0])

ax[1].set_title("death "+c)

death[c].plot(ax=ax[1])

ax[2].set_title("recoveries "+c)

recoveries[c].plot(ax=ax[2], color='green')

plt.show()
fig,ax=plt.subplots(1,3,figsize=(15,5))

fig.tight_layout()

c=death.columns[8]

ax[0].set_title("confirmed "+c)

confirmed[c].plot(ax=ax[0])

ax[1].set_title("death "+c)

death[c].plot(ax=ax[1])

ax[2].set_title("recoveries "+c)

recoveries[c].plot(ax=ax[2], color='green')

plt.show()
fig,ax=plt.subplots(1,3,figsize=(15,5))

fig.tight_layout()

c=death.columns[9]

ax[0].set_title("confirmed "+c)

confirmed[c].plot(ax=ax[0])

ax[1].set_title("death "+c)

death[c].plot(ax=ax[1])

ax[2].set_title("recoveries "+c)

recoveries[c].plot(ax=ax[2], color='green')

plt.show()
fig,ax=plt.subplots(1,3,figsize=(15,5))

fig.tight_layout()

c=death.columns[10]

ax[0].set_title("confirmed "+c)

confirmed[c].plot(ax=ax[0])

ax[1].set_title("death "+c)

death[c].plot(ax=ax[1])

ax[2].set_title("recoveries "+c)

recoveries[c].plot(ax=ax[2], color='green')

plt.show()
fig,ax=plt.subplots(1,3,figsize=(15,5))

fig.tight_layout()

c=death.columns[11]

ax[0].set_title("confirmed "+c)

confirmed[c].plot(ax=ax[0])

ax[1].set_title("death "+c)

death[c].plot(ax=ax[1])

ax[2].set_title("recoveries "+c)

recoveries[c].plot(ax=ax[2], color='green')

plt.show()
fig,ax=plt.subplots(1,3,figsize=(15,5))

fig.tight_layout()

c=death.columns[12]

ax[0].set_title("confirmed "+c)

confirmed[c].plot(ax=ax[0])

ax[1].set_title("death "+c)

death[c].plot(ax=ax[1])

ax[2].set_title("recoveries "+c)

recoveries[c].plot(ax=ax[2], color='green')

plt.show()
fig,ax=plt.subplots(1,3,figsize=(15,5))

fig.tight_layout()

c=death.columns[13]

ax[0].set_title("confirmed "+c)

confirmed[c].plot(ax=ax[0])

ax[1].set_title("death "+c)

death[c].plot(ax=ax[1])

ax[2].set_title("recoveries "+c)

recoveries[c].plot(ax=ax[2], color='green')

plt.show()
fig,ax=plt.subplots(1,3,figsize=(15,5))

fig.tight_layout()

c=death.columns[14]

ax[0].set_title("confirmed "+c)

confirmed[c].plot(ax=ax[0])

ax[1].set_title("death "+c)

death[c].plot(ax=ax[1])

ax[2].set_title("recoveries "+c)

recoveries[c].plot(ax=ax[2], color='green')

plt.show()
fig,ax=plt.subplots(1,3,figsize=(15,5))

fig.tight_layout()

c=death.columns[15]

ax[0].set_title("confirmed "+c)

confirmed[c].plot(ax=ax[0])

ax[1].set_title("death "+c)

death[c].plot(ax=ax[1])

ax[2].set_title("recoveries "+c)

recoveries[c].plot(ax=ax[2], color='green')

plt.show()
fig,ax=plt.subplots(1,3,figsize=(15,5))

fig.tight_layout()

c=death.columns[16]

ax[0].set_title("confirmed "+c)

confirmed[c].plot(ax=ax[0])

ax[1].set_title("death "+c)

death[c].plot(ax=ax[1])

ax[2].set_title("recoveries "+c)

recoveries[c].plot(ax=ax[2], color='green')

plt.show()
fig,ax=plt.subplots(1,3,figsize=(15,5))

fig.tight_layout()

c=death.columns[17]

ax[0].set_title("confirmed "+c)

confirmed[c].plot(ax=ax[0])

ax[1].set_title("death "+c)

death[c].plot(ax=ax[1])

ax[2].set_title("recoveries "+c)

recoveries[c].plot(ax=ax[2], color='green')

plt.show()
fig,ax=plt.subplots(1,3,figsize=(15,5))

fig.tight_layout()

c=death.columns[18]

ax[0].set_title("confirmed "+c)

confirmed[c].plot(ax=ax[0])

ax[1].set_title("death "+c)

death[c].plot(ax=ax[1])

ax[2].set_title("recoveries "+c)

recoveries[c].plot(ax=ax[2], color='green')

plt.show()
fig,ax=plt.subplots(1,3,figsize=(15,5))

fig.tight_layout()

c=death.columns[19]

ax[0].set_title("confirmed "+c)

confirmed[c].plot(ax=ax[0])

ax[1].set_title("death "+c)

death[c].plot(ax=ax[1])

ax[2].set_title("recoveries "+c)

recoveries[c].plot(ax=ax[2], color='green')

plt.show()