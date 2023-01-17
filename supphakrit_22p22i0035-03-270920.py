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

import scipy.cluster.hierarchy as sch

from sklearn.cluster import AgglomerativeClustering

from sklearn.preprocessing import MinMaxScaler



data = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

data.head()
data['neighbourhood'].value_counts()
fit_data = data[['neighbourhood','price','availability_365','number_of_reviews','latitude','longitude']] 

fit_data = fit_data.groupby('neighbourhood').agg({'price':'mean','availability_365':'mean','number_of_reviews':'mean','latitude':'mean','longitude': 'mean'})

fit_data
fig = plt.figure(figsize=(20,20))

dendogram = sch.dendrogram(sch.linkage(fit_data,method='ward'),leaf_rotation=90, leaf_font_size=12,labels=fit_data.index) 

plt.title("Neighbourhood")  

plt.show()