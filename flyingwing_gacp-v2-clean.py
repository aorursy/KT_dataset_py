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
import pandas as pd
import numpy as np
PATH = "../input/"
train_data = pd.read_pickle(PATH+"train_v2_clean.pkl")
test_data  = pd.read_pickle(PATH+"test_v2_clean.pkl")
test_data.columns
train_data.shape, test_data.shape
train_data.date.max(), train_data.date.min(), test_data.date.max(), test_data.date.min()
result = test_data[['fullVisitorId','totals_transactionRevenue']].fillna(0)
result['totals_transactionRevenue'] = result['totals_transactionRevenue'].astype(float)
result = result.groupby('fullVisitorId').agg({"totals_transactionRevenue":"sum"})

result = result.reset_index().rename(columns={"totals_transactionRevenue":"PredictedLogRevenue"})
result['PredictedLogRevenue'] = np.log1p(result['PredictedLogRevenue'])
result.to_csv("predict.csv",index=False)
result.shape
