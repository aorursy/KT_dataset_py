# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

%matplotlib inline 

pd.options.mode.chained_assignment = None  # default='warn'
train=pd.read_json("../input/train.json")

test=pd.read_json("../input/test.json")

train["Source"]='train'

test["Source"]='test'

data=pd.concat([train, test]) 
plt.scatter(data["longitude"], data["latitude"], s=5)

plt.title("Geographical positions of the listings")

plt.show()
plt.scatter(data.loc[data["longitude"]<-60,"longitude"], data.loc[data["latitude"]>20,"latitude"], s=5)

plt.title("Geographical positions of the listings")

plt.show()
plt.scatter(data.loc[(data["longitude"]<-73.75)&(data["longitude"]>-74.05)&(data["latitude"]>40.4)&(data["latitude"]<40.9),"longitude"],

                      data.loc[(data["latitude"]>40.4)&(data["latitude"]<40.9)&(data["longitude"]<-73.75)&(data["longitude"]>-74.05),"latitude"], s=5)

plt.title("Geographical positions of the listings")

plt.show()




#I use Birch because of how fast it is. 

from sklearn.cluster import Birch

def cluster_latlon(n_clusters, data):  

    #split the data between "around NYC" and "other locations" basically our first two clusters 

    data_c=data[(data.longitude>-74.05)&(data.longitude<-73.75)&(data.latitude>40.4)&(data.latitude<40.9)]

    data_e=data[~(data.longitude>-74.05)&(data.longitude<-73.75)&(data.latitude>40.4)&(data.latitude<40.9)]

    #put it in matrix form

    coords=data_c.as_matrix(columns=['latitude', "longitude"])

    

    brc = Birch(branching_factor=100, n_clusters=n_clusters, threshold=0.01,compute_labels=True)



    brc.fit(coords)

    clusters=brc.predict(coords)

    data_c["cluster_"+str(n_clusters)]=clusters

    data_e["cluster_"+str(n_clusters)]=-1 #assign cluster label -1 for the non NYC listings 

    data=pd.concat([data_c,data_e])

    plt.scatter(data_c["longitude"], data_c["latitude"], c=data_c["cluster_"+str(n_clusters)], s=10, linewidth=0.1)

    plt.title(str(n_clusters)+" Neighbourhoods from clustering")

    plt.show()

    return data 
data["created"]=pd.to_datetime(data["created"])

data["created_month"]=data["created"].dt.month

data["created_day"]=data["created"].dt.day

data["created_hour"]=data["created"].dt.hour
data["num_photos"]=data["photos"].apply(len)

data["num_features"]=data["features"].apply(len)

data["num_description_words"] = data["description"].apply(lambda x: len(x.split(" ")))
compute_logloss(3, data)
data=cluster_latlon(100, data)



clusters_price_map=dict(data.groupby(by="cluster_100")["price"].median())

data["price_comparison"]=data['price']-data["cluster_100"].map(clusters_price_map)
result=data[['price_comparison', 'Source', 'listing_id']]

result.to_pickle("cluster.pkl")
data