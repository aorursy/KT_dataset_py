# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import pylab



df = pd.read_csv('../input/mock_kaggle.csv', parse_dates=['data'])



pylab.figure(figsize=(14,6))

for column in ['venda', 'estoque', 'preco']:

	pylab.plot(df['data'], df[column]/max(df[column]), label=column)

pylab.legend(loc='upper right')

pylab.show()