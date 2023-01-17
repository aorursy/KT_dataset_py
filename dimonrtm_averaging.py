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
ssub=pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv")
import pandas as pd

xgboost_lagged_features = pd.read_csv("../input/xgboost/xgboost_lagged_features.csv")
import pandas as pd

linear_regression_lagged_features_6 = pd.read_csv("../input/linearregression/linear_regression_lagged_features_6.csv")
import pandas as pd

random_forest_lagged_features = pd.read_csv("../input/randfor/random_forest_lagged_features.csv")
ssub.head()
xgboost_lagged_features.head()
linear_regression_lagged_features_6.head()
ssub["item_cnt_month"]=(xgboost_lagged_features["item_cnt_month"]+random_forest_lagged_features["item_cnt_month"])/2
ssub.head()
ssub.to_csv("weighted_averaging.csv",index=False)