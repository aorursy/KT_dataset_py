# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/IPL2013.csv")
data.dtypes
column_head=list(data.head(0))
data.loc[0:5,column_head[0]:column_head[9]]
data=data.drop([column_head[0],column_head[1]],axis=1,inplace=False)
data
column_head=list(data.columns)

column_head
data_encoded=pd.get_dummies(data)
data_encoded
column_heads=list(data_encoded.columns)
from statsmodels.stats.outliers_influence import variance_inflation_factor
def get_vif_factors(X):

    x_matrix=X.as_matrix()

    vif=[variance_inflation_factor(x_matrix,i)for i in range(x_matrix.shape[1])]

    vif_factors=pd.DataFrame()

    vif_factors["column"]=X.columns

    vif_factors['vif']=vif

    return vif_factors

vif_factors=get_vif_factors(data_encoded.drop(column_heads[column_heads.index("SOLD PRICE")],axis=1,inplace=False))
vif_factors
columns_with_large_vif=vif_factors[vif_factors.vif>4].column