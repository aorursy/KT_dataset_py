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
df = pd.read_csv('../input/Iris.csv')

df.head()
target = df['Species'].values

target.shape
df = df.drop(['Id','Species'],axis=1)

df.head()
X_train = df.as_matrix()
def cen_init(k=3):

    np.random.seed(0)

    c = np.random.random((k,4))

    return c
c_init = cen_init()*5

print(c_init)
from math import sqrt

def distance(t,c):

    return sqrt(np.sum((t-c)*(t-c)))
def kmeans(X_train,nIter):

    m = X_train.shape[0]

    cen = cen_init(3)*5

    k = cen.shape[0]

    dis = np.zeros([m,k])

    cen_ass = np.zeros([m,])

    cen_his = cen

    temp = np.zeros([1,4])

    count = 0

    for t in range(nIter):

        #print(cen)

        for r in range(0,m):

            for c in range(0,k):

                dis[r][c] = distance(X_train[r],cen[c])

        cen_ass = (np.argmin(dis,axis=1)).reshape((-1,))

        cen_his = np.concatenate((cen_his,cen))

        print(cen)

        for c in range(0,k):

            temp = np.zeros([1,4])

            count = 0

            for r in range(0,m):

                temp = temp + (0.98)*(cen_ass[r]==c)*X_train[r]+(0.02)*(cen_ass[r]!=c)*X_train[r]

                count = count + (0.98)*(cen_ass[r]==c)+(0.02)*(cen_ass[r]!=c)

            cen[c] = (temp.reshape((-1,)))/(count)

    return cen,cen_ass,cen_his
cen,cen_ass,cen_his = kmeans(X_train,nIter=10)
print(cen_ass)
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

pca = PCA(n_components=2,random_state=1)

pca.fit(X_train)

X_new = pca.transform(X_train)

cen_new = pca.transform(cen)

c_init_new = pca.transform(c_init)
color = ['r','g','b']

for c,t in enumerate(np.unique(target)):

    print(c,t)

    target[target==t] = color[c]

    

cen_col = ['a' for i in range(150)]

for c,i in enumerate(cen_ass):

    if i==0:

        cen_col[c] = 'r'

    elif i==1:

        cen_col[c] = 'g'

    elif i==2:

        cen_col[c] = 'b'
out_index = target==cen_col

outliers = np.array([x for c,x in enumerate(X_new) if out_index[c]==False])

outliers.shape
plt.figure(figsize=(21,7))

plt.subplot(131)

plt.scatter(X_new[:,0],X_new[:,1],color=target)

yellow = plt.scatter(c_init_new[:,0],c_init_new[:,1],color='black',s=75,label='init cen')

black = plt.scatter(cen_new[:,0],cen_new[:,1],color='orange',s=100,label='final cen')

plt.legend(handles=[yellow,black])

plt.title('Actual labels and final centroids')



plt.subplot(132)

plt.scatter(X_new[:,0],X_new[:,1],color=cen_col)

plt.scatter(c_init_new[:,0],c_init_new[:,1],color='black',s=75,label='init cen')

plt.scatter(cen_new[:,0],cen_new[:,1],color='orange',s=100,label='final cen')

plt.legend(handles=[yellow,black])

plt.title('Clustered labels and final centroids')



plt.subplot(133)

axes = plt.gca()

axes.set_xlim([-3.5,4])

axes.set_ylim([-2.5,1.5])

plt.scatter(outliers[:,0],outliers[:,1])

plt.scatter(cen_new[:,0],cen_new[:,1],color='orange',s=100,label='final cen')

plt.title('Outlier points')



plt.show()
cen_his_col = []

for  i in range(11):

    cen_his_col.append('r')

    cen_his_col.append('g')

    cen_his_col.append('b')
axes = plt.gca()

axes.set_xlim([-3.5,4])

axes.set_ylim([-2.5,1.5])



cen_his_new = pca.transform(cen_his)

plt.scatter(cen_his_new[::3,0],cen_his_new[::3,1],s=[i*10 for i in range(11)],color=cen_his_col[::3])

plt.plot(cen_his_new[::3,0],cen_his_new[::3,1],'r')



plt.scatter(cen_his_new[1::3,0],cen_his_new[1::3,1],s=[i*10 for i in range(11)],color=cen_his_col[1::3])

plt.plot(cen_his_new[1::3,0],cen_his_new[1::3,1],'g')



plt.scatter(cen_his_new[2::3,0],cen_his_new[2::3,1],s=[i*10 for i in range(11)],color=cen_his_col[2::3])

plt.plot(cen_his_new[2::3,0],cen_his_new[2::3,1],'b')



plt.scatter(cen_new[:,0],cen_new[:,1],color=['r','g','b'],s=100,label='final cen')

plt.show()