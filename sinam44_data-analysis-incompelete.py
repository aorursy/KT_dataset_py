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
calendar = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')
sell_train = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
sale_price = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')

calendar.head()
import matplotlib.pyplot as plt
import seaborn as sns
calendar.dtypes
calendar.plot()
plt.show()
calendar['date'].unique()
fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=(20,5))
sns.kdeplot(calendar['year'],ax=ax1)
sns.kdeplot(calendar['month'],ax=ax2)
sns.kdeplot(calendar['wday'],ax=ax3)
