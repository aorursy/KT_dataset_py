import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data = pd.read_csv('../input/data.csv')
data.sample(5)
def mapping(data,feature):
    featureMap=dict()
    count=0
    for i in sorted(data[feature].unique(),reverse=True):
        featureMap[i]=count
        count=count+1
    data[feature]=data[feature].map(featureMap)
    return data
data=mapping(data,"diagnosis")
data=data.drop(["id","Unnamed: 32"],axis=1)
data.sample(5)
data["concavity_mean"]=((data["concavity_mean"]-data["concavity_mean"].min())/(data["concavity_mean"].max()-data["concavity_mean"].min()))*20
data.sample(5)
dataf=((data-data.min())/(data.max()-data.min()))*20
dataf.sample(5)
def normalize(dataset):
    dataNorm=((dataset-dataset.min())/(dataset.max()-dataset.min()))*20
    dataNorm["diagnosis"]=dataset["diagnosis"]
    return dataNorm
data=normalize(data)
data.sample(5)