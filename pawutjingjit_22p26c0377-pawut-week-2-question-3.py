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
data = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
################### EDA ######################################################################
data.head()
data.describe()
############# latitude  longitude price  minimum_nights number_of_reviews  reviews_per_month calculated_host_listings_count availability_365 Is Numerical
# Drop ID (Unique Numerical), Hostid  (Unique Numerical ), HostName(Unique stinrg) , last_review(Date)  , neighbourhood (Spare Category)
# room_type is Category
# drop unique column
data = data.drop(['id' , 'host_id' , 'host_name' ,'neighbourhood_group' ,'neighbourhood','last_review', 'name'] ,axis=1 )

# roomtype to one hot encoder 
data = pd.get_dummies(data , ['room_type'])
# if reviews = Null -> it's meaning no reviews for row
data['reviews_per_month'] = data['reviews_per_month'].fillna(0)
data.columns
################# Define dendrogram plot###########
################ We can't create dendrogram for 50k row , we need to sampling (in case we sampling 30)
def plot_dendrogram():
    link = hierarchy.linkage(data.iloc[:30])
    plt.figure(figsize = (25,15))
    dend = hierarchy.dendrogram(link)
    plt.show()
plot_dendrogram()