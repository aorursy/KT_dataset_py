# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/TRY_USD.csv')
data['Tarih'] = pd.to_datetime(data.Tarih)

data.info()
data.plot(x = 'Tarih', y = 'Düşük')
data.plot(x = 'Tarih', y = 'Fark', kind = 'Line', alpha = .5)
data.plot(subplots = True, figsize = (12,12))

plt.show()
x = data['Açılış'] <= 1.15

data[x]
threshold = sum(data.Açılış) / len(data.Yüksek)

print('threshold' ,threshold)



data['Açılış_Seviyesi'] = ["Yüksek" if i > threshold else "Düşük" for i in data.Açılış]

data.loc[:10, ['Açılış_Seviyesi', 'Açılış']]

melted = pd.melt(frame = data.head(), id_vars = 'Tarih', value_vars = ['Yüksek', 'Düşük'])

melted