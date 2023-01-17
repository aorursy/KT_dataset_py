import pandas as pd

import numpy as np

import random

from datetime import datetime
wcet=pd.DataFrame(data=[[1,14,16,19],[2,13,19,18],[3,11,13,19],[4,13,8,17],[5,12,13,10],[6,13,10,9],[7,7,15,11],[8,5,11,14],[9,18,12,20],[10,21,7,15]],

             columns=["Task","P1","P2","P3"])
wcct=pd.DataFrame(data=[[1,2],[1,3],[1,4],[1,5],[1,6],[2,8],[2,9],[3,7],[4,8],[4,9],[5,9],[6,8],[7,10],[8,10],[9,10]],

             columns=["from","to"])
wcet["mean"]=wcet.drop(["Task"],axis=1).mean(axis=1)
wcet["std"]=wcet.drop(["Task"],axis=1).std(axis=1)
wcet
df=wcet

df1=pd.DataFrame(columns=["1","2","3","4","5"],index=range(0,10))

for z in df1.columns:

    now = datetime.now()

    random.seed(now)

    for x in range(0,df.shape[0]):

        m=int(df["mean"][x])

        s=int(df["std"][x])

        df1[z][x]=random.randint(m-s,m+s)

for w in df1.columns:

    df[w]=df1[w]

df    