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
from sklearn import preprocessing, metrics
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="ticks", color_codes=True)
import statsmodels.api as sm
from scipy import stats
data = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
data.head()
data.shape
data.isnull().sum()
data.dtypes
sns.heatmap(data.corr(),annot=True)
sns.pairplot(data)
data.fillna({'reviews_per_month':0}, inplace=True)
data.fillna({'name':"NoName"},inplace=True)
data.fillna({'host_name':"NoName"},inplace=True)
data.fillna({'last_review':"NotReviewed"},inplace=True)
data.isnull().sum()
data["price"].describe()
data_price = data["price"].hist()
hist_price1=data["price"][data["price"]<1000].hist()

data[data["price"]>1000]
data = data[data["price"]<250]
data["price"].describe()
data['neighbourhood'].value_counts()
