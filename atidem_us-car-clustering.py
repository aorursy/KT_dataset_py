#cluster for expert system homework
from sklearn.cluster import KMeans 
import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv("/kaggle/input/usa-cers-dataset/USA_cars_datasets.csv",index_col=0)
df.drop("vin",axis=1,inplace=True)
df.drop("country",axis=1,inplace=True)
df.drop("condition",axis=1,inplace=True)
df.drop("lot",axis=1,inplace=True)
df.drop("title_status",axis=1,inplace=True)
df.describe().T
#bind values for numeric column
priceBind = [10200,16900,25555.5]
milBind = [21466.5,35365,63472.5]

for i in range(len(df)):
    if(df.loc[i,'price']<priceBind[0]):
        df.loc[i,'price']=0
    else:
        if(df.loc[i,'price']<priceBind[1]):
            df.loc[i,'price']=1
        else:
            if(df.loc[i,'price']<priceBind[2]):
                df.loc[i,'price']=2
            else:
                df.loc[i,'price']=3

    if(df.loc[i,'mileage']<milBind[0]):
        df.loc[i,'mileage']=0
    else:
        if(df.loc[i,'mileage']<milBind[1]):
            df.loc[i,'mileage']=1
        else:
            if(df.loc[i,'mileage']<milBind[2]):
                df.loc[i,'mileage']=2
            else:
                df.loc[i,'mileage']=3

#categorical to numerical
brand = list(df.brand.unique())
model = list(df.model.unique())
color = list(df.color.unique())
state = list(df.state.unique())

for i in range(len(df)):
    df.loc[i,'brand'] = brand.index(df.loc[i,'brand'])
    df.loc[i,'model'] = model.index(df.loc[i,'model']) 
    df.loc[i,'color'] = color.index(df.loc[i,'color']) 
    df.loc[i,'state'] = state.index(df.loc[i,'state']) 


df.head(10)
km = KMeans(n_clusters=10).fit(df)
cluster = km.predict(df)

df["cluster"] = cluster
for i in range(len(df)):
    df.loc[i,'brand'] = brand[df.loc[i,'brand']]
    df.loc[i,'model'] = model[df.loc[i,'model']]
    df.loc[i,'color'] = color[df.loc[i,'color']]
    df.loc[i,'state'] = state[df.loc[i,'state']]
df.head(100)