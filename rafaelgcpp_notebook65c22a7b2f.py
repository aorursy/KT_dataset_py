# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as pp 



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

# print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.



trainset=pd.read_csv('../input/train.csv')
cols=['X%d'%i for i in range(9) if i!=7]

a=trainset[cols+['ID']].groupby(cols).count().reset_index() 

a.sort_values('ID')
a[a.X0==a.X1].head()
a=np.array(trainset[cols])

a=pd.DataFrame(a.ravel())

a=a[0].sort_values().unique()



la=len(a)

dict1={a[i]:2*int(i-np.floor(la/2))/la for i in range(la) }

dict2={int(i-np.floor(la/2)):a[i] for i in range(la) }



for c in cols:

    trainset[c]=trainset[c].map(dict1)

   

trainset.head()
X=np.array(trainset.iloc[:,2:])

y=np.array(trainset.iloc[:,1],ndmin=2).T

my=max(y)

y=y/my
R=np.dot(X.T,X)
pp.matshow(R)
U,S,V=np.linalg.svd(R)

print(np.sum(S>0.01))

del U, S, V, R
import sklearn.decomposition as skd

pca=skd.PCA(n_components=260, whiten=True)
Xpca=pca.fit(X)
Xnew=Xpca.transform(X)

Xnew.shape
pp.matshow(np.dot(Xnew.T,Xnew))
import sklearn.neural_network as nn



mlp=nn.MLPRegressor(hidden_layer_sizes=(600,300) )
mlp=mlp.fit(Xnew,y.ravel())
mlp.score(Xnew,y.ravel())
testdata=pd.read_csv('../input/test.csv')

for c in cols:

    testdata[c]=testdata[c].map(dict1)



testdata.head()

Xtest=Xpca.transform(np.array(testdata.iloc[:,1:]))
