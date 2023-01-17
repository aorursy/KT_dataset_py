import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
data=pd.read_csv('../input/BreadBasket_DMS.csv')
data
data.Item.unique()
data=data.drop(data[(data.Item=='NONE')].index)
data.Item.unique()
len(data)
data1=data.iloc[:,2:4].values
data1
len(data1)
data1=np.append(data1,[[100000,'xxxx']], axis=0)
data1
transactions=[]
dum=[]
f=0
for i in range(0,20507):
    dum.append(data1[i][1])
    if data1[i][0] != data1[i+1][0]:
        f=1
    if f==1:
        transactions.append(dum)
        dum=[]
    f=0
    
        
len(transactions)
transactions[9462]
from apyori import apriori
rules=apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2)
result=list(rules)
for k in result:
    print(k)