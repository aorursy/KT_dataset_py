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
# import necessery libraries
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
%matplotlib inline
# import data
airbnb = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
airbnb.tail()
airbnb.info()
airbnb.isnull().sum()
# drop irrelevant columns
airbnb = airbnb[['latitude', 'longitude', 'price']]
airbnb.sample(10)
# replace all NaN values in 'reviews_per_month' with 0
#airbnb.fillna({'reviews_per_month': 0}, inplace=True)
#airbnb.isnull().sum()
airbnb.hist(bins=100, figsize=(10,8))
plt.tight_layout()
plt.show()
# drop random sample for reduce compute workload

np.random.seed(42)
remove_n = 45000
drop_indices = np.random.choice(airbnb.index, remove_n, replace=False)
airbnb_subset = airbnb.drop(drop_indices)
len(airbnb_subset)
from sklearn.preprocessing import normalize
airbnb_scaled = normalize(airbnb_subset)
airbnb_scaled = pd.DataFrame(airbnb_scaled, columns=airbnb.columns)
airbnb_scaled.head()
import scipy.cluster.hierarchy as sch

plt.figure(figsize=(15,8))
plt.title("Dendrograms")
plt.xlabel('Place')
plt.ylabel('Euclidean distance')
dend = sch.dendrogram(sch.linkage(airbnb_scaled, method='ward'))
plt.show()
sub = airbnb[airbnb.price < 500]
viz=sub.plot(kind='scatter', x='longitude', y='latitude', label='airbnb', c='price',
                  cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, figsize=(20,16))
viz.legend()
