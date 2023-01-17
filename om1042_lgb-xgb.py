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
lgb_df = pd.read_csv('../input/ieee-lgb-bayesian-opt-pca/LGB_Bayesian_PCA_0.9676682843135961.csv')

xgb_df = pd.read_csv('../input/ieee-fraud-xgboost-with-gpu-pca/xgboost_v1_0.7807836916516994.csv')
submission_df  = lgb_df.copy()
submission_df['isFraud'] = 0.8*lgb_df['isFraud'] + 0.20*xgb_df['isFraud']

submission_df.to_csv('lgb_xgb.csv', index=False )