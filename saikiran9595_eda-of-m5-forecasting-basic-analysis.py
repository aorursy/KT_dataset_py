# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input/m5-forecasting-accuracy'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

calendar = pd.read_csv("../input/m5-forecasting-accuracy/calendar.csv")

sales_train_validation = pd.read_csv("../input/m5-forecasting-accuracy/sales_train_validation.csv")

sample_submission = pd.read_csv("../input/m5-forecasting-accuracy/sample_submission.csv")

sell_prices = pd.read_csv("../input/m5-forecasting-accuracy/sell_prices.csv")
sales_train_validation.head()
sales_train_validation.columns
sell_prices.head()
calendar_df.head()
calendar_df.columns