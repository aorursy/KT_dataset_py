# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

data=pd.read_csv('../input/cleandata.csv')

data.head()
data.isnull().sum() #checking for total null values

f,ax=plt.subplots(1,2,figsize=(18,8))

sns.countplot('Dist',data=data,ax=ax[0])

ax[0].set_title('District-Wise')

ax[0].set_xlim(0,3)

sns.countplot('State',data=data,ax=ax[1])

ax[1].set_title('State-Wise')

f.show()
data.groupby(['Gender','Dist'])['Dist'].count()
df_agg = data.groupby(['Gender','Dist'])['Dist'].count().reset_index(name='counts')

df_agg.sort_values(['counts','Dist','Gender'],ascending=False).groupby('Gender').head(3)