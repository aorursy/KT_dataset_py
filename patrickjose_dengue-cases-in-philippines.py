# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns



%matplotlib inline

sns.set()
import pandas as pd

df= pd.read_csv("../input/dengue-cases-in-the-philippines/denguecases.csv")

df.head()
a = df['Dengue_Cases'].sum()

mx = df['Year'].max()

mn = df['Year'].min()

print('The total number of dengue cases from {} to {} is {}'.format(mn, mx, round(int(a))))
df['percentage'] = df['Dengue_Cases']/a

df.head()
plt.figure(figsize=[20,5])

sns.kdeplot(df['percentage'], shade=True)

plt.title('Distribution of Dengue')
df.boxplot(column='percentage', by = 'Region', figsize=[20,8], grid=False, patch_artist='True')

plt.title("Dispersion per Region")
cases_agg = pd.pivot_table(index=['Year'], 

                           values='Dengue_Cases', 

                           data=df, 

                           aggfunc=[np.mean, np.median, np.std, max, min])

cases_agg
cases_agg.plot(kind='area', figsize=[20,5])
cases_agg = pd.pivot_table(index=['Region'], 

                           values='Dengue_Cases', 

                           data=df, 

                           aggfunc=[np.mean, np.median, np.std, max, min])

cases_agg
cases_agg.plot(kind='area', figsize=[20,5])
sns.catplot(x='Month',

            y='Dengue_Cases',

            hue='Year',

            data=df,

            kind='swarm',

            height=8,

            aspect=2,

            s=10

           )