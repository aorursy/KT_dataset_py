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
from sklearn.decomposition import PCA, FastICA,KernelPCA,TruncatedSVD

import numpy as np

from numpy.testing import assert_array_almost_equal

import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train=train[train.y<250]

y_train = train['y']

train=train.sort_values(['ID'])

test=test.sort_values(['ID'])

train=train.drop(['y'],axis=1)



df=pd.concat([train,test])



pd.get_dummies(df).shape







# process columns, apply LabelEncoder to categorical features

for c in train.columns:

    if train[c].dtype == 'object':

        lbl = LabelEncoder() 

        lbl.fit(list(train[c].values) + list(test[c].values)) 

        train[c] = lbl.transform(list(train[c].values))

        test[c] = lbl.transform(list(test[c].values))



# shape        

print('Shape train: {}\nShape test: {}'.format(train.shape, test.shape))





new_train=train.drop(['ID'],axis=1)

new_test=test.drop(['ID'],axis=1)





data = pd.concat([new_train,new_test])



data.shape

#VIF  analysis:

X=data .values



C=np.linalg.pinv(np.dot(X.T,X))



factor=[]

for i in range(0,C.shape[0]):

    factor.append(C[i][i])



plt.clf()

plt.plot(factor)

plt.show()
from numpy import linalg as LA

max_comp=100

start=20

error_record=[]

for i in range(start,max_comp):

    pca = PCA(n_components=i, random_state=42)

    pca2_results = pca.fit_transform(data)

    pca2_proj_back=pca.inverse_transform(pca2_results)

    total_loss=LA.norm((data-pca2_proj_back),None)

    error_record.append(total_loss)



plt.clf()

plt.figure(figsize=(15,15))

plt.title("reconstruct error of pca")

plt.plot(error_record,'r')

plt.xticks(range(len(error_record)), range(start,max_comp), rotation='vertical')

plt.xlim([-1, len(error_record)])

plt.show()
