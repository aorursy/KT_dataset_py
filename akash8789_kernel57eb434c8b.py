import  matplotlib.pyplot as plt

import pandas as pd

import numpy as np
data=pd.read_csv('../input/train.csv')

data.head(7)
data.describe()
data.info()
plt.boxplot(data['x'])
plt.boxplot(data['y'])
b1=0

b2=0

l=0.01

#o=b1+b2*x
lx=list(data.x)

lx[216]

ly=list(data.y)
for i in range(260):

    o=b1+b2*lx[i]

    er=o-ly[i]

    b1=round(b1-l*er,2)

    b2=b2-l*er*lx[i]

    print(b1,b2)

    
data2=pd.read_csv('../input/test.csv')

data2.head()
testx=list(data2.x)

result=[]

index=[]

testy=list(data2.y)
for i in range(len(testx)-1):

    o=b1+b2*testx[i]

    result.append(o)

    index.append(i)

predict=list(zip(result,testy))
predictFrame=pd.DataFrame(predict,columns=['Y','y'])

predictFrame.head()
plt.scatter(x=predictFrame['Y'],y=predictFrame['y'],data=predictFrame)