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
import numpy as np   # importing numpy for linear algebra operations
import pandas as pd  # importing pandas for data processing
import os            # importing os for using operating system functionality
#Reading the news sample data from the news_sample.csv file using pandas
news_data=pd.read_csv("../input/news_sample.csv")
#Reading the market data from the marketdata_sample.csv file using pandas
market_data=pd.read_csv("../input/marketdata_sample.csv")

# Getting the descriptive statistics at coulmn level for news sample data
news_data.describe()
# Getting the descriptive statistics at coulmn level for news sample data
market_data.describe()
#Finding the Null Values in market data
market_data.isna().sum()
#Finding the Null Values in market data
news_data.isna().sum()
