import numpy as np

import pandas as pd
df=pd.read_csv('../input/imdb-extensive-dataset/IMDb movies.csv')
df.head()
data=pd.DataFrame()
c=df['avg_vote'].mean()
c
m=df['votes'].quantile(0.90)
m
data=df[df['votes']>=9819.600000000006]
data.shape
df.shape
score=[]

v=data['votes'].values

r=data['avg_vote'].values

def weighted_average(v,r,c,m):

    s=((v*r)/(v+m))+((m*c)/(v+m))

    return s

for i in range(len(v)):

    score.append(weighted_average(v[i],r[i],c,m))
data['weighted_score']=score
data=data.sort_values('weighted_score',ascending=False)
data.head()