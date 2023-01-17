# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

import matplotlib.pyplot as plt

import seaborn as sns

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df = pd.read_csv('/kaggle/input/foreign-exchange-rates-per-dollar-20002019/Foreign_Exchange_Rates.csv')

df.head()
df.info()

yuan_usd = df.groupby(['CHINA - YUAN/US$'])['Time Serie'].count()

plt.barh(yuan_usd.index,yuan_usd)

plt.show()
hongkong_usd = df.groupby(['HONG KONG - HONG KONG DOLLAR/US$'])['Time Serie'].count()

plt.barh(hongkong_usd.index,hongkong_usd)

plt.show()
sing_usd = df.groupby(['SINGAPORE - SINGAPORE DOLLAR/US$'])['Time Serie'].count()

plt.barh(sing_usd.index,sing_usd)

plt.show()