# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sus
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
corona_dataset_csv=pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')
corona_dataset_csv.head()
df=corona_dataset_csv.drop(['Lat','Long'],axis=1,inplace=True)
corona_dataset_csv.shape
corona_dataset_csv.head()

data_set=corona_dataset_csv.groupby("Country/Region").sum()
data_set.head(10)
data_set.shape
data_set.loc["India"].plot()
data_set.loc["Russia"].plot()
data_set.loc["Spain"].plot()
plt.legend()