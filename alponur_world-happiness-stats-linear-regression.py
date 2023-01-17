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

import matplotlib.pyplot as plt

import numpy as np

import seaborn as sns



df = pd.read_csv('/kaggle/input/world-happiness/2019.csv')
df.head()
hm = df.corr()

sns.heatmap(hm)

plt.show();
df.corr()
x = np.array(df.loc[:,'GDP per capita']).reshape(-1,1)

y = np.array(df.loc[:,'Healthy life expectancy']).reshape(-1,1)
plt.figure(figsize=[8,8])

plt.scatter(x=x,y=y)

plt.xlabel('GDP')

plt.ylabel('Healthy life expectancy')

plt.show();
from sklearn.linear_model import LinearRegression

reg = LinearRegression()

reg.fit(x,y)

predict1 = np.linspace(min(x), max(x)).reshape(-1,1)



predict2 = reg.predict(predict1)



print('R^2 score: ',reg.score(x, y))





plt.figure(figsize=[8,8])

plt.plot(predict1, predict2, color='black', linewidth=3)

plt.scatter(x=x,y=y)

plt.xlabel('GDP')

plt.ylabel('Health Exp')

plt.show()
