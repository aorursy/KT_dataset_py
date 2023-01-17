# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import StandardScaler
#from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df = pd.read_csv("../input/music-frequency-dataset/music.csv")

test_data = pd.read_csv("../input/music-frequency-dataset/test_music.csv")
df.columns
x =  df[['LVar', 'LAve', 'LMax', 'LFEner', 'LFreq']]
my_col  = list(x.columns)
print(my_col)
scaler = StandardScaler()
x = scaler.fit_transform(x)
x
x  = pd.DataFrame(x)
x.columns =  my_col
print(x)
#### Scree Plot #########

## we are applying K-means and trying to find the cost(intra cluster distance)

'''for k in range (1, 21):
    # Create a kmeans model on our data, using k clusters.  random_state helps ensure that the algorithm returns the same results each time.
    kmeans_model = KMeans(n_clusters=k, random_state=1)
    kmeans_model.fit(x)

    # These are our fitted labels for clusters -- the first cluster has label 0, and the second has label 1.
    
    labels = kmeans_model.labels_

    # Sum of distances of samples to their closest cluster center
    interia = kmeans_model.inertia_
    print("k:",k, " cost:", interia)'''
model = DBSCAN(eps=0.8, min_samples=5)
model.fit(x)

#model = KMeans(n_clusters=3, random_state=1)
#model.fit(x)
np.unique(model.labels_)
model.labels_
# create a dataframe of model cluster numbers
model_labels_df = pd.DataFrame(model.labels_)
model_labels_df.columns = ["Cluster"]
model_labels_df
music_output = pd.concat([df,model_labels_df],axis =1)
music_output
test_data
x_test  = test_data.drop(['Song','Artist'],axis =1)
cols =  x_test.columns

x_test = scaler.transform(x_test)
x_test
x_test = pd.DataFrame(x_test)
x_test
x_test.columns = cols
x_test
model.predict(x_test)