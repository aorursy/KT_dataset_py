# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.preprocessing import StandardScaler

from sklearn.cluster import DBSCAN



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv("../input/iris/Iris.csv")
data
#import pandas as pd

 

# The kmeans algorithm is implemented in the scikits-learn library

#from sklearn.cluster import KMeans





x = data.drop(['Species','Id'],axis =1)

 



    
x
my_col  = list(x.columns)

print(my_col)

scaler = StandardScaler()

x = scaler.fit_transform(x)
x
x  = pd.DataFrame(x)

x.columns =  my_col

x
#### Scree Plot #########

"""

## we are applying K-means and trying to find the cost(intra cluster distance)



for k in range (1, 21):

    # Create a kmeans model on our data, using k clusters.  random_state helps ensure that the algorithm returns the same results each time.

    kmeans_model = KMeans(n_clusters=k, random_state=1)

    kmeans_model.fit(x)



    # These are our fitted labels for clusters -- the first cluster has label 0, and the second has label 1.

    

    labels = kmeans_model.labels_



    # Sum of distances of samples to their closest cluster center

    interia = kmeans_model.inertia_

    print("k:",k, " cost:", interia)

"""
# K Means Cluster

model = DBSCAN(eps=0.45,min_samples=4)

model.fit(x)



# Cluster lables as outcome of model

print(model.labels_)



x
np.unique(model.labels_)
len(model.labels_)
x.shape
# create a dataframe of model cluster numbers

model_labels_df = pd.DataFrame(model.labels_)

model_labels_df.columns = ["Cluster"]

model_labels_df
# We join actual data and model output(cluster number)

iris = pd.concat([data,model_labels_df],axis =1)
iris
iris['Cluster'].value_counts()

my_data_for_knn = iris[iris['Cluster'] != -1]
my_data_for_knn['Cluster'].value_counts()
iris.to_csv("iris_output.csv")
##### eps = 0.5 and min points = 5



"""

1    71

 0    44

-1    35



"""



## min points  = 2



"""

 2    73

 0    44

-1    14

 7     3

 6     3

 5     3

 4     3

 3     3

 1     2

 8     2



"""



### eps = 0.8 and min pints = 5



"""

1    97

 0    49

-1     4

"""



### eps = 0.2 and min pints = 5



"""





"""
