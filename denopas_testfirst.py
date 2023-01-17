# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



df = pd.read_csv('../input/50_Startups.csv')

X = df.iloc[:,:-1].values

y = df.iloc[:, 4].values

df
df.describe()
df.corr()
df.plot(x='R&D Spend', y='Profit', style='o')  

plt.title('R&D Spend - Profit')  

plt.xlabel('R&D Spend')  

plt.ylabel('Profit')  

plt.show() 