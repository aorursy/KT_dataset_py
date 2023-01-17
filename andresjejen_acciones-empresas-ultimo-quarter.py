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
data  = pd.read_csv('/kaggle/input/investdatatest/INVEST.csv',delimiter=";")
data
data.info()
data["PRECIO_CIERRE"] = [float(x.replace(',','.')) for x in data["PRECIO_CIERRE"]]
data.info()
data['FECHA'] =  pd.to_datetime(data['FECHA'])

datagrouped = data.groupby(['COMPAÑIA'])
data
datagrouped.head()
datagrouped.first().info()
import matplotlib.pyplot as plt
cierres = []

for name, group in datagrouped:

    group.index = group['FECHA']

    datagraph = group['2019-08-01':'2019-10-31']

    datagraph = datagraph[['PRECIO_CIERRE']]

    dif = datagraph['PRECIO_CIERRE']['2019-10-31']-datagraph['PRECIO_CIERRE']['2019-08-01']

    cierre = dif*100/datagraph['PRECIO_CIERRE']['2019-08-01']

    maxima = "Maxima",datagraph['PRECIO_CIERRE'].max

    cierres.append({'name':name,'value':cierre})

    fig= plt.figure(figsize=(10,10))

    axes = fig.add_axes([0.1,0.1,0.8,0.8])

    axes.plot(datagraph['PRECIO_CIERRE'])

    

    # axes.text('2019-08-30', maxima, 'Test', color='red')

    axes.text(0.95, 0.01, '% Crecimiento: {0:3f}%'.format(cierre),

        verticalalignment='bottom', horizontalalignment='right',

        transform=axes.transAxes,

        color='blue', fontsize=15)

    

    plt.title(name)

    plt.grid(True)

    plt.show()
mayores = {'company': [], 'value': []}

for i in cierres:

    mayores['company'].append(i['name'])

    mayores['value'].append(i['value'])

datacierres = pd.DataFrame(mayores)

datacierres.head()
datacierres = datacierres.sort_values(['value'], ascending=False, axis =0)

datacierres.index = datacierres['company']

datacierres
fig= plt.figure(figsize=(10,10))

axes = fig.add_axes([0.1,0.1,0.8,0.8])

import matplotlib.pyplot as plt

for name, group in datagrouped:

    if (name in datacierres['company'].iloc[0:5].values):

        # print(name)

        group.index = group['FECHA']

        datagraph = group['2019-08-01':'2019-10-31']

        #datagraph = datagraph[['PRECIO_CIERRE']]

        datagraph = datagraph[['PRECIO_CIERRE']]/datagraph[['PRECIO_CIERRE']].values.max()

        axes.plot(datagraph['PRECIO_CIERRE'], label= name )

#plt.yticks([i*100 for i in range(12)]) 

axes.legend()

plt.grid(True)
fig= plt.figure(figsize=(10,10))

axes = fig.add_axes([0.1,0.1,0.8,0.8])

import matplotlib.pyplot as plt

for name, group in datagrouped:

    if (name in datacierres['company'].iloc[0:5].values):

        # print(name)

        group.index = group['FECHA']

        datagraph = group['2019-08-01':'2019-10-31']

        datagraph = datagraph[['PRECIO_CIERRE']]

        #datagraph = datagraph[['PRECIO_CIERRE']]/datagraph[['PRECIO_CIERRE']].values.max()

        axes.plot(datagraph['PRECIO_CIERRE'], label= name )

plt.yticks([i*100 for i in range(12)]) 

axes.legend()

plt.grid(True)