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
data = pd.read_csv("/kaggle/input/customer-behaviour/Customer_Behaviour.csv")
data.head()
data.describe()
data.shape
import seaborn as sns
sns.distplot(data.Age)
iris= sns.load_dataset("iris")
iris.shape
iris.describe()
iris.head()
hp = pd.read_csv("/kaggle/input/others/House_Price.csv")
hp.head()
hp.describe()
hp.shape
sns.jointplot(x='n_hot_rooms', y='price', data=hp)
sns.jointplot(x='rainfall', y='price', data=hp)
hp.head()
sns.countplot(x="airport", data=hp)
sns.countplot(x="waterbody", data=hp)
sns.countplot(x="bus_ter", data=hp)
hp.info()
np.percentile(hp.n_hot_rooms,[99])
np.percentile(hp.n_hot_rooms,[99])[0]
uv = np.percentile(hp.n_hot_rooms,[99])[0]
hp[(hp.n_hot_rooms>uv)]
hp.n_hot_rooms[(hp.n_hot_rooms>3*uv)] = 3*uv
np.percentile(hp.rainfall,[1])[0]
lv = np.percentile(hp.rainfall,[1])[0]
hp[(hp.rainfall < lv)]
hp.rainfall[(hp.rainfall< 0.3*lv)]= 0.3*lv
sns.jointplot(x="crime_rate", y="price", data=hp)
hp.describe()
hp.n_hos_beds = hp.n_hos_beds.fillna(hp.n_hos_beds.mean)
hp.info()
#hp = hp.fillna(df.mean())

#it will work column wise 
sns.jointplot(x="crime_rate", y="price", data=hp)
hp.crime_rate = np.log(1+hp.crime_rate)
sns.jointplot(x="crime_rate", y="price", data=hp)
import math
#math.exp(2)
hp['avg_dist'] = (hp.dist1+hp.dist2+hp.dist3+hp.dist4)/4
hp.describe()
del hp['dist1']
del hp['dist2']

del hp['dist3']
del hp['dist4']
hp.describe()
del hp['bus_ter']
hp.info()
#hp = pd.get_dummies(hp, columns = ['airport', 'waterbody'], drop_first=True)

hp = pd.get_dummies(hp, columns = ['airport', 'waterbody'])
hp.head()
del hp['airport_NO']
del hp['waterbody_None']
hp.corr()
del hp['parks']
hp.info()