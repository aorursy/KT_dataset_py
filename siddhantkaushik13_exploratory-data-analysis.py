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
import matplotlib.pyplot as plt

import seaborn as sns
df004=pd.read_csv('../input/lok-sabha-election-candidate-list-2004-to-2019/LokSabha2004.csv')

df009=pd.read_csv('../input/lok-sabha-election-candidate-list-2004-to-2019/LokSabha2009.csv')

df014=pd.read_csv('../input/lok-sabha-election-candidate-list-2004-to-2019/LokSabha2014.csv')

df019=pd.read_csv('../input/lok-sabha-election-candidate-list-2004-to-2019/LokSabha2019.csv')
df004.columns
plt.figure(figsize=(18, 8))



plt.subplot(221)

df004['Gender'].hist()



plt.subplot(222)

df014['Gender'].hist()
fig, ax2 = plt.subplots(2, 2, figsize=(25, 12))





sns.countplot(y='Education', data=df004, ax=ax2[0][0])

sns.countplot(y='Education', data=df009, ax=ax2[0][1])

sns.countplot(y='Education', data=df019, ax=ax2[1][0])

sns.countplot(y='Education', data=df014, ax=ax2[1][1])

plt.figure(figsize=(20, 12))



plt.subplot(221)

sns.distplot(df004['Age'], kde=False, bins=10)



plt.subplot(222)

sns.distplot(df009['Age'], kde=False, bins=10)



plt.subplot(223)

sns.distplot(df014['Age'], kde=False, bins=10)



plt.subplot(224)

sns.distplot(df019['Age'], kde=False, bins=10)
df004['Year']=2004

df009['Year']=2009

df014['Year']=2014

df019['Year']=2019

y= [df004, df009, df014, df019]
df = pd.concat(y)

df.head(10)
plt.plot(df['Year']. value_counts(), 'bo')