# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
full_dataframe = pd.read_csv("/kaggle/input/global-covid-data/public/data/owid-covid-data.csv")

full_dataframe.head(10)
df_austria = full_dataframe[full_dataframe.location == "Austria"]

df_austria.head(10)
df_sweden = full_dataframe[full_dataframe.location == "Sweden"]

df_sweden.head(10)
data_mobility = pd.read_csv("/kaggle/input/covid19-mobility-data/Google_Mobility_Data.csv", low_memory=False)

data_mobility.head(10)
mb_austria =  data_mobility[(data_mobility.country_region =="Austria") & (data_mobility.sub_region_1.isnull())]

mb_austria.head(10)
mb_sweden =  data_mobility[(data_mobility.country_region =="Sweden") & (data_mobility.sub_region_1.isnull())]

mb_sweden.head(10)
