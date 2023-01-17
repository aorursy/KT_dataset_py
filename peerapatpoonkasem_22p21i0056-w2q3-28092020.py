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
import seaborn as sns
ds = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
ds.sample(10) ##ดึงข้อมูลมา (x) ตัว
ols = ['id','host_id','latitude','longitude','price','minimum_nights','number_of_reviews'      
       ,'calculated_host_listings_count','availability_365']
ds[ols]
ds[ols].info()

sortdataset = ds[ols].sort_values('id')
print(sortdataset)
sortdataset.head()

# data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/การบ้าน/Week 2/Q3/Dataset/AB_NYC_2019.csv',
#                    parse_dates = {'last_reviews': ['last_review']}
#                    ,infer_datetime_format=True
#                    ,dayfirst=True
#                    )
# ds[ols]
ds[ols].drop(ds[ols].index[2])
sortdataset.drop(ds[ols].index[100:48894])
print(ds[ols].head())
ds[ols].info()
sortdataset.head()
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch #เอาไว้วาด
import sklearn.datasets
ds[ols].sample(10)
ds[ols].drop(ds[ols].index[2])
# len(ds[ols])
# len(sortdataset)
ds[ols]
dendrogram = sch.dendrogram(sch.linkage(sortdataset[0:1000], method='ward'))
plt.title('Dendrogram')
plt.xlabel('Price')
plt.ylabel('Availability')
plt.show()