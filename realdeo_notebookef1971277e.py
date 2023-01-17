# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from scipy import stats

from sklearn.neighbors import KernelDensity



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv",index_col='Id')

train.head()
train.columns.values
train[['SalePrice','LotArea']].head()
plt.plot(train['LotArea'],train['SalePrice'],'ro')

plt.xlabel('Lot Area')

plt.ylabel('SalePrice')

plt.show()
'''

plt.plot(train['Condition1'],train['SalePrice'],'ro')

plt.xlabel('Distance')

plt.ylabel('SalePrice')

plt.show()

'''

train[['SalePrice','BedroomAbvGr']][train['BedroomAbvGr']==2].head()
neighborhood_list = (list(set(i for i in train['Neighborhood'].values)))

for i in neighborhood_list:

    print ("%s %d" % (i,len(train[train['Neighborhood']==i])))
train['PricePerArea'] = np.array(train['SalePrice']) // np.array(train['LotArea'])

train[['Neighborhood','PricePerArea']].head()
def mean_stdDev(array):

    '''

    return:

    return[0] -> mean

    return[1] -> stddev

    '''

    mean = sum(array)/len(array)

    stdDev = 0.0

    for i in array:

        stdDev += (i-mean)**2

    stdDev /= len(array)

    return mean, stdDev
for i in neighborhood_list:

    wow = (train.SalePrice[train['Neighborhood']==i].values)//1000

    mean, stdDev = mean_stdDev(wow)

    print("%d %d %s" %(mean,stdDev,i))
xs = np.linspace(50,500,900)

for i in neighborhood_list:

    wow = (train.SalePrice[train['Neighborhood']==i].values)//1000

    kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(wow)

    plt.plot(xs,kde(xs),label=i)

plt.show()

    