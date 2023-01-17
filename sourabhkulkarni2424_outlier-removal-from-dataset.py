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
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df = pd.read_csv('../input/national-stock-exchange-time-series/nifty_it_index.csv')

df.head()
sns.pairplot(df,palette='coolwarm')
plt.figure(figsize=(10,5))

plt.hist(df['Turnover'],bins=40,rwidth=0.8)

plt.show
df.Turnover.describe()
df['Zscore']= (df.Turnover - df.Turnover.mean())/df.Turnover.std()

df.head()
df[(df.Zscore>2)|(df.Zscore<-2)]
final_df= df[(df.Zscore<2)&(df.Zscore>-2)]

print(df.shape,final_df.shape)
plt.figure(figsize=(10,5))

plt.hist(final_df['Turnover'],bins=40,rwidth=0.8)

plt.show
final_df.Turnover.describe()
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)

plt.hist(df['Turnover'],bins=40,rwidth=0.8,color='g')

plt.title('Data with Outliers')





plt.subplot(1,2,2)

plt.hist(final_df['Turnover'],bins=40,rwidth=0.8,color='b')

plt.title('Data after processing')




