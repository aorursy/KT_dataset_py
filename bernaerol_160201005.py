# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from statistics import mode
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix
df = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv") 
df1 = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv") 
df1

train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv', index_col = 'date', parse_dates = True )
train.head()

test= pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
test.head()
train.info()
test=test.dropna()
train=train.dropna()
test.head()
sb.pairplot(train)

df.info()
df.isnull()
df.tail(10)
print(df.describe())
