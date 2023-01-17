from IPython.display import Image

from IPython.core.display import HTML 

Image(url="https://cdn-images-1.medium.com/max/800/1*pr2lbvx1jHw9aU1WsP73LA.gif")
# import numpy library as np

import numpy as np



import matplotlib.pyplot as plt
arr = np.array([1,3,5,np.nan,6,8])

arr
arr[1]
# import pandas library as pd

import pandas as pd

# Create Pandas Series 

s = pd.Series([1,3,5,np.nan,6,8])

print(s)
series_index = pd.Series(np.array([10,20,30,40,50,60]), index=['a', 'b', 'c', 'd', 'e', 'f'])

series_index
#Generate dates

dates = pd.date_range('20130101', periods=6)

print(dates)
temp_data = pd.DataFrame(np.random.randint(low=-20, high=40, size=(12, 2)))

temp_data
df = pd.DataFrame(np.random.randn(6,4), index=[1,2,3,4,5,6], columns=list('ABCD'))

print(df)
# Generate new list of dates

dates = pd.date_range('20190101', periods=12)



# Generate dataframe with random temperatures

temp_data = pd.DataFrame(np.random.randint(low=-20, high=40, size=(12, 2)), index=dates, 

                             columns=['Forecast Avg', 'Actual Average'])

# Print temp data

temp_data
df2 = pd.DataFrame({ 'A' : 1.,

                     'B' : pd.Timestamp('20130102'),

                     'C' : pd.Series(1,index=list(range(4)),dtype='float32'),

                     'D' : np.array([3] * 4,dtype='int32'),

                     'E' : pd.Categorical(["test","train","test","train"]),

                     'F' : 'foo' })



print(df2)

print("\n")

print(df2.dtypes)
temp_data.head()
temp_data.tail(3)
temp_data.index
temp_data.iloc[0]['Forecast Avg']
import pandas as pd

Data = {'x': [25,34,22,27,33,33,31,22,35,34,67,54,57,43,50,57,59,52,65,47,49,48,35,33,44,45,38,43,51,46],

        'y': [79,51,53,78,59,74,73,57,69,75,51,32,40,47,53,36,35,58,59,50,25,20,14,12,20,5,29,27,8,7]

       }

df = pd.DataFrame(Data,columns=['x','y'])
# vectors = array_function( dataframe )

twoDvectors = np.array(df)
from sklearn.cluster import KMeans



# model_name = Kmeans(n_Of_clusters).fit(vectors)

kmeans_model = KMeans(n_clusters = 3).fit(twoDvectors)



# get cluster centroids

# cluster_centroids = model_name.cluster_centers_

cluster_centroids = kmeans_model.cluster_centers_
import matplotlib.pyplot as plt



# plot feature points and cluster lables

plt.scatter(twoDvectors[:,0], twoDvectors[:,1], c= kmeans_model.labels_)



# plot cluster centroids 

plt.scatter(cluster_centroids[:,0], cluster_centroids[:,1], c='red', s=50)