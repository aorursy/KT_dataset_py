
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
train_df = pd.read_csv("../input/train.csv")
test_df=   pd.read_csv("../input/test.csv")
train_df=train_df.dropna()
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
import statsmodels.api as sm 

X=train_df.drop(['ID', 'target'],axis=1)
Y=train_df['target']
X = StandardScaler().fit_transform(X)
variables=list(X)
pca = PCA(n_components=1000)   
principalComponents = pca.fit_transform(X)
principalDf = pd.DataFrame(data = principalComponents)
principalDf
    

model = sm.OLS(Y,principalDf ).fit()
model.summary()