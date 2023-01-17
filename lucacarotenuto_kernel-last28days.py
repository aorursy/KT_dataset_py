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
cal = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/calendar.csv")

sales = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv")

prices = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sell_prices.csv")

subm_ex = pd.read_csv("/kaggle/input/m5-forecasting-accuracy/sample_submission.csv")

subm = subm_ex.copy()

subm.drop(subm.tail(30490).index,inplace=True) # drop last n rows
subm[subm.columns[-28:]] = sales[sales.columns[-28:]]
for day in range(28):

    subm["F" + str(day + 1)] = sales["d_" + str(day + 1886)]
subm_2 = subm.copy()

subm_2['id'] = subm_2['id'].apply(lambda x: x.replace('validation',

                                                     'evaluation'))
subm = subm.append(subm_2).reset_index(drop=True)
subm.to_csv("submission.csv", index=False)
sales