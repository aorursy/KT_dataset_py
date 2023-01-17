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
# reading tips.csv to dataframe named data

data_path = '../input/tipping/tips.csv'

data = pd.read_csv(data_path)
from sklearn.cluster import KMeans
new_data = data.loc[:,['tip','size']]
kmeans = KMeans(n_clusters=2).fit(new_data)
new_data['cluster_labels'] = kmeans.labels_


import seaborn as sns

import matplotlib.pyplot as plt

sns.scatterplot(x=new_data.loc[:,'size'],y=new_data.loc[:,'tip'],hue=new_data.loc[:,'cluster_label'])

plt.show()

# clusters centers:

kmeans.cluster_centers_