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
ipl16_df = pd.read_csv("../input/IPL2013.csv")
print(ipl16_df.iloc[0:4, 0:9])
ipl16_df.drop(["Sl.NO."], axis = 1, inplace = False) 

  


categorical_feature_mask = ipl16_df.dtypes==object



categorical_cols = ipl16_df.columns[categorical_feature_mask].tolist()
import pandas as pd

from sklearn import preprocessing

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
ipl16_df[categorical_cols] = ipl16_df[categorical_cols].apply(lambda col: le.fit_transform(col))

ipl16_df[categorical_cols].head(10)
from statsmodels.stats.outliers_influence import variance_inflation_factor

def get_vif_factors(ipl16_df):

    ipl16_df_matrix=ipl16_df.as_matrix()

    vif=[variance_inflation_factor(ipl16_df_matrix,i)for i in range(ipl16_df_matrix.shape[1])]

    vif_factors=pd.DataFrame()

    vif_factors['column']=ipl16_df.columns

    vif_factors['vif']=vif

    return vif_factors





vif_factors=get_vif_factors(predictors)

vif_factors

columns_with_large_vif=vif_factors[vif_factors.vif>4].column