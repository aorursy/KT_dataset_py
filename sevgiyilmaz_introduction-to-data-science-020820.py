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
df = pd.read_csv("../input/bitcoin-historical-data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv")
df.head()
df.info()
df.columns
#Timestamp is the id of datas
df.describe()
#We learned that the features involve numeric values thanks to info().With describe(), we can learn numerical relations features
df.corr()
# We can check the correlation information to find relations between features
# 1 shows that there is a positive relation between features and -1 shows that there is a negative relation between features
import matplotlib.pyplot as plt
import seaborn as sns

f,ax=plt.subplots(figsize=(20,20))
sns.heatmap(df.corr(),annot=True,linewidths=.3,fmt='.1f',ax=ax)
# With seaborn,we can saw the correlation in graph
df.plot(kind='line',x='Timestamp',y='Open',alpha=.5,color='blue')
plt.xlabel('Timestamp')
plt.ylabel('Open')
plt.title('Changing Opening Values in Time')
plt.show()
plt.scatter(df.Open,df.Close,color="green",alpha=0.5)
plt.xlabel('Open')
plt.ylabel('Close')
plt.title('Opening-Closing Values')
plt.show()
#When we checked the graph,it shows that openning and closing values are directly interaction between them.Also,we saw this relation in correlation table
df = df.rename(columns={'Volume_(Currency)': 'Volume_Currency'})
plt.hist(df.Volume_Currency,bins=10)
plt.xlabel('Volume_Currency')
plt.ylabel('frequency')
plt.title('Frequency of Volume Currency')
plt.show()

#Graph shows that the volume(currency) is not normal distribution.