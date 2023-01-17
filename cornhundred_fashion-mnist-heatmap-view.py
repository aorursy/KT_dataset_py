# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
from clustergrammer2 import net

df = {}
# defined labels

labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',

         'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
df['test'] = pd.read_csv('../input/fashion-mnist_test.csv')

df['test'].shape

rows = df['test'].index.tolist()

ser_label = df['test']['label']

new_rows = [('F-' + str(x), 'Label: ' + str(labels[ser_label[x]])) for x in rows]

df['test'] = df['test'].drop(['label'], axis=1)

df['test'].index = new_rows

df['test'] = df['test'].transpose()
net.load_df(df['test'])

net.random_sample(axis='col', num_samples=2000, random_state=99)

net.load_df(net.export_df().round(2))

net.widget()