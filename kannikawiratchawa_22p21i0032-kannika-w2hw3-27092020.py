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

import pandas as pd

%matplotlib inline

import numpy as np
AB_NYC_2019 = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
AB_NYC_2019.head()
AB_NYC_2019.shape
AB_NYC_2019.columns
AB_NYC_2019['neighbourhood'].value_counts()
AB_NYC_2019.isnull().any()
AB_NYC_2019_Drop = AB_NYC_2019.dropna()
AB_NYC_2019_Drop.head()
df_AB_NYC_2019 = AB_NYC_2019_Drop[['id' ,'room_type' ,'neighbourhood_group' ,'price','number_of_reviews']]
df_AB_NYC_2019.head()
df_AB_NYC_2019.shape
df_AB_NYC_2019.isnull().any()
df_AB_NYC_2019['room_type'].value_counts()
labels = 'Entire_home/apt ', 'Private_room ', 'Shared_room'

sizes = [20321, 17654,  846]



p = plt.pie(sizes, labels=labels, explode=(0.07, 0, 0),

            autopct='%1.1f%%', startangle=130, shadow=True)

plt.axis('equal')



for i, (Entire_home_apt , Private_room , Shared_room) in enumerate(p):

    if i > 0:

        Entire_home_apt.set_fontsize(12)

        Private_room.set_fontsize(12)

        Shared_room.set_fontsize(12)

       

    plt.show()
df_AB_NYC_2019['neighbourhood_group'].value_counts().plot.bar()
sam = df_AB_NYC_2019['price'] 



plt.scatter( range(len(sam)) , sam )

plt.show()
x = df_AB_NYC_2019['number_of_reviews'].values
plt.hist(x, 50, density=True, facecolor='r', alpha=0.75);
import scipy.cluster.hierarchy as shc
data_NYC = df_AB_NYC_2019.iloc[:,3:4].values
data_NYC 
plt.figure(figsize=(10, 7))

plt.title("Airbnb Dendograms")

dendrogram_Airbnb = shc.dendrogram(shc.linkage(data_NYC, method='ward'))
from sklearn.cluster import AgglomerativeClustering



cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')

cluster.fit_predict(data_NYC)
plt.scatter(range(len(data_NYC)) ,data_NYC, c=cluster.labels_, cmap='rainbow' )

plt.show()