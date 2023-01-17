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
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression,BayesianRidge,LogisticRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from sklearn.ensemble import GradientBoostingRegressor,GradientBoostingClassifier
from sklearn.svm import SVR,SVC
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_log_error,mean_absolute_error
import matplotlib.pyplot as py
import seaborn as sb
sb.set(rc={"figure.figsize":(10,5)})
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop = stopwords.words('english')
from sklearn.feature_extraction.text import CountVectorizer
cvec=CountVectorizer()
from nltk.stem import PorterStemmer,WordNetLemmatizer
s=PorterStemmer()
l=WordNetLemmatizer()
train = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv")
train.head()
train.groupby(["shop_id","item_id"])["item_cnt_day"].sum()
k = pd.DataFrame({'item_cnt_month' : train.groupby(["shop_id","item_id"])["item_cnt_day"].sum()}).reset_index()
k.head(7)
test = pd.read_csv("/kaggle/input/competitive-data-science-predict-future-sales/test.csv")
test.head()
res = test.merge(k, on=["shop_id","item_id"],how="left")
test.shape
res.shape
res = res.fillna(0)
res.isna().sum()
res.dtypes
res[["ID", "item_cnt_month"]].to_csv('submission.csv',index=False)
