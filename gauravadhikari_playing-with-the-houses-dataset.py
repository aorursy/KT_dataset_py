# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

import seaborn as sns 

%matplotlib inline
testDf=pd.read_csv('../input/test.csv')

trainDf=pd.read_csv('../input/train.csv')
trainDf.info()
trainDf.shape
testDf.shape
trainDf.head()
testDf.head()
trainDf.loc[:,'MSSubClass':'SaleCondition'].shape
allDf=pd.concat((trainDf.drop(['Id','SalePrice'],axis=1),

           testDf.drop(['Id'],axis=1)))
trainDf['SalePrice'].hist()
np.log1p(trainDf['SalePrice']).hist()
trainDf['SalePrice']=np.log1p(trainDf['SalePrice'])