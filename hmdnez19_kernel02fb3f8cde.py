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

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence,partial_dependence
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA 
from sklearn.preprocessing import Imputer
from matplotlib import pyplot as plt
import numpy as np
import math
import lightgbm as lgb
nRowsRead = 100
df = pd.read_csv('../input/vgsales.csv', delimiter=',' ,nrows = nRowsRead) #Data setimizi pandas ile okumamız için yolu belirtiyoruz.
df.dataframeName = 'vgsales.csv' #Shape ve print kullanarak row ve columnları yazdırıyoruz.
[nRow, nCol] = df.shape
print(f'Bakılan {nRow} satır ve {nCol} sütun')
df = pd.read_csv('../input/vgsales.csv')
df.head()
df.corr()
df.isnull().sum()
selected_features = ['NA_Sales','Global_Sales','EU_Sales','JP_Sales']
defining_columns = df[selected_features]
prediction_column = df.Other_Sales
prediction_column.describe()
defining_columns_train,defining_columns_test,prediction_column_train,prediction_column_test = train_test_split(defining_columns,prediction_column,test_size = 0.2, random_state = 0)
forest_model = RandomForestRegressor(random_state = 0)
forest_model.fit(defining_columns_train,prediction_column_train)
prediction_column_pred = forest_model.predict(defining_columns_test)
other_sales_prediction = forest_model.predict(defining_columns_test)
mean_absolute_error(prediction_column_test,other_sales_prediction)
from sklearn.metrics import *
accuracy_score(prediction_column_test,forest_model.predict(defining_columns_test))