# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mpl
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#importing the data
filepath = "../input/us-counties-covid-19-dataset/us-counties.csv"
data = pd.read_csv(filepath, index_col = 'date')
print(data.state.unique())
%matplotlib inline 
#The .diff() command counts the daily increases, by subtracting the previous
# day's cumulative totals
lacounty = data.loc[data.county == 'Los Angeles']
lacounty['cases'].diff().plot(rot = 90)
plt.title('Daily COVID-19 Cases in LA County')
plt.show()
orangecounty = data.loc[(data.county == 'Orange') & (data.state == 'California')]
orangecounty['cases'].diff().plot(rot = 90)
plt.title('Daily COVID-19 Cases in Orange County')
plt.show()
