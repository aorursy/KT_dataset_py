# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# Any results you write to the current directory are saved as output.
filesList=os.listdir('../input/')
dataSet=[]
for f in filesList:
    df=pd.read_csv('../input/'+f)
    dataSet.append(df)
print("Done!")
mainDF=pd.concat([x for x in dataSet],axis=0)
mainDF.columns
id_1_data=mainDF[mainDF.id==1]
fig2 = plt.figure(figsize=(20,10))
ax2 = fig2.add_subplot(121, projection='3d')
ax2.scatter(xs=id_1_data.x,ys=id_1_data.y,zs=id_1_data.z)
grp=mainDF.groupby(by=['x','y','z'])
meanDF=grp.aggregate('mean')
meanDF.columns
fig3 = plt.figure(figsize=(20,10))
ax3 = fig3.add_subplot(121, projection='3d')
ax3.scatter(xs=meanDF.vx[:1000],ys=meanDF.vy[:1000],zs=meanDF.vz[:1000])
from sklearn.cluster import KMeans
trainDFSet1=meanDF[['vx','vy','vz']].iloc[:1000]
inertiaList=[]
for i in range(1,70):
    km=KMeans(n_clusters=i)
    km.fit(trainDFSet1)
    inertiaList.append(km.inertia_)
plt.plot(range(1,70),inertiaList)
plt.show()
finalKM=KMeans(n_clusters=19)
finalKM.fit(trainDFSet1)
preds=finalKM.predict(trainDFSet1)
len(preds)
fig4 = plt.figure(figsize=(20,10))
ax4 = fig4.add_subplot(121, projection='3d')
ax4.scatter(xs=trainDFSet1.vx[:1000],ys=trainDFSet1.vy[:1000],zs=trainDFSet1.vz[:1000],c=preds[:1000])

