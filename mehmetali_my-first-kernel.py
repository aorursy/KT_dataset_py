# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # data visualization
import seaborn as sns #data visualization

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("/kaggle/input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv")
df.head()
df.rename(columns={"Volume_(BTC)": "Volume_BTC", "Volume_(Currency)" : "Volume_Currency"}, 
                 inplace=True)
print(df.columns)
df.Weighted_Price.plot(kind = 'line', color = 'g', label = 'Weighted_Price',linewidth = 1,figsize = (16,6), alpha = 0.5, grid = True, linestyle = ':')
plt.ylabel("Weighted Price($)")
plt.show()
df.corr()
#set the width and heigth of the figure
plt.subplots(figsize=(11,11))
plt.title("Bitcoin Dataset Correlation Heatmap")

sns.heatmap(df.corr(), annot = True, linewidths = 1, fmt = '.1f')
plt.show()

df.plot(kind='scatter', x='Open', y='Close',alpha = 0.5,color = 'purple')
plt.xlabel("Open")  
plt.ylabel("Close")
plt.show()
df.Weighted_Price.plot(kind = 'hist', bins = 90, figsize = (20, 20), color = 'grey')
plt.show()