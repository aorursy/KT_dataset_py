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
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (10,6) # define figure size of pyplot

pd.set_option("display.max_columns", 100) # set max columns when displaying pandas DataFrame

pd.set_option("display.max_rows", 200) # set max rows when displaying pandas DataFrame
df_product = pd.read_csv('/kaggle/input/uisummerschool/Product.csv')

df_online = pd.read_csv('/kaggle/input/uisummerschool/Online_sales.csv')

df_offline = pd.read_csv('/kaggle/input/uisummerschool/Offline_sales.csv')

df_marketing = pd.read_csv('/kaggle/input/uisummerschool/Marketing.csv')
df_marketing_spend=pd.read_csv('/kaggle/input/uisummerschool/Marketing.csv')

df_marketing_spend
df_marketing_spend.describe()
df_marketing_spend=pd.read_csv('/kaggle/input/uisummerschool/Marketing.csv')

df_marketing_spend.set_index('Date', inplace=True)

df_marketing_spend.index = pd.to_datetime(df_marketing_spend.index)

df_marketing_spend.resample('1M').sum().plot(kind='line')