# ----- CREATED UPON CREATION OF KAGGLE PYTHON KERNEL -----
# ----- KEPT AS IS -----
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
# Load data using BigQueryHelper:
import bq_helper
print("Loading Training Data Set...")
ga_bq_train = bq_helper.BigQueryHelper(active_project= "kaggle-public-datasets", dataset_name = "ga_train_set")
print("Loading Test Data Set... ")
ga_bq_test = bq_helper.BigQueryHelper(active_project= "kaggle-public-datasets", dataset_name = "ga_test_set")
print("Data Loaded: done")
# Test query estimate
queryy = \
    """
    SELECT  fullVisitorId
    FROM `kaggle-public-datasets.ga_test_set.ga_sessions_*` 
    GROUP BY fullVisitorId
    ORDER BY fullVisitorId
    """
print("size = " + str( ga_bq_test.estimate_query_size(queryy) * 1000 ) + " MB" )
# Test query final
print('starting queryy...')
test_as_pandas_data = ga_bq_test.query_to_pandas_safe(queryy)
print('done')
test_as_pandas_data.head(5)
# The predicted output needs to be the natural log of the sum + 1
# which is all zeros as ln(1) = 0
test_as_pandas_data['PredictedLogRevenue'] = np.log(0+1)
test_as_pandas_data.head(5)
test_as_pandas_data.describe()
test_as_pandas_data.info()
# Let's save it:
print('saving csv file...')
test_as_pandas_data.to_csv('submission_SteveBlack_v002_date_2018_1005_0959.csv', index = False)
print('done')