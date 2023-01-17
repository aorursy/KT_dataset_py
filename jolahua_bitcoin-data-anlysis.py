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
coin, bit = [os.path.join('../input', e) for e in os.listdir('../input')]

coin, bit = pd.read_csv(coin), pd.read_csv(bit)

print('Coin data:')

print(coin.head())

print('Bit data:')

print(bit.head())