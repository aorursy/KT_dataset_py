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
usdf= pd.read_csv("../input/USvideos.csv")
dedf= pd.read_csv("../input/DEvideos.csv")
frdf= pd.read_csv("../input/FRvideos.csv")
cadf= pd.read_csv("../input/CAvideos.csv")
gbdf= pd.read_csv("../input/GBvideos.csv")

import pandas as pd # package for high-performance, easy-to-use data structures and data analysis
import numpy as np # fundamental package for scientific computing with Python
import matplotlib
import matplotlib.pyplot as plt # for plotting
import seaborn as sns # for making plots with seaborn
color = sns.color_palette()
# import plotly.offline as py
# py.init_notebook_mode(connected=True)
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.offline as offline
offline.init_notebook_mode()
from plotly import tools
import plotly.tools as tls
import squarify
from mpl_toolkits.basemap import Basemap
from numpy import array
from matplotlib import cm

# import cufflinks and offline mode
import cufflinks as cf
cf.go_offline()

import json 

#temp1 = gun[['state', 'n_killed']].groupby(['state'], as_index=False).sum().sort_values(by='n_killed', ascending=False).head(20)
usdf["region"] = "US"
cadf["region"] = "CA"
gbdf["region"] = "GB"
dedf["region"] = "DE"
frdf["region"] = "FR"

totaldf = pd.concat([usdf,cadf,gbdf, dedf,frdf])
totaldf.reset_index(inplace = True)
totaldf.head()
totaldf['trending_date'] = pd.to_datetime(totaldf['trending_date'], format='%y.%d.%m')
totaldf['publish_time'] = pd.to_datetime(totaldf['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
totaldf['publish_date'] = totaldf['publish_time'].dt.date
totaldf['publish_time'] = totaldf['publish_time'].dt.time
# creates a dictionary that maps `category_id` to `category`
us_id_to_category = {}

with open('../input/US_category_id.json', 'r') as f:
    data = json.load(f)
    for category in data['items']:
        us_id_to_category[category['id']] = category['snippet']['title']

ca_id_to_category = {}

with open('../input/CA_category_id.json', 'r') as f:
    data = json.load(f)
    for category in data['items']:
        ca_id_to_category[category['id']] = category['snippet']['title']
        
de_id_to_category = {}

with open('../input/DE_category_id.json', 'r') as f:
    data = json.load(f)
    for category in data['items']:
        de_id_to_category[category['id']] = category['snippet']['title']
        
gb_id_to_category = {}

with open('../input/GB_category_id.json', 'r') as f:
    data = json.load(f)
    for category in data['items']:
        gb_id_to_category[category['id']] = category['snippet']['title']
        
fr_id_to_category = {}

with open('../input/FR_category_id.json', 'r') as f:
    data = json.load(f)
    for category in data['items']:
        fr_id_to_category[category['id']] = category['snippet']['title']
        
us1 = pd.Series(us_id_to_category)
uscatdf = pd.DataFrame(us1, columns = ["USCat"])
ca1 = pd.Series(ca_id_to_category)
cacatdf = pd.DataFrame(ca1, columns = ["CACat"])
de1 = pd.Series(de_id_to_category)
decatdf = pd.DataFrame(de1, columns = ["DECat"])
gb1 = pd.Series(gb_id_to_category)
gbcatdf = pd.DataFrame(gb1, columns = ["GBCat"])
fr1 = pd.Series(fr_id_to_category)
frcatdf = pd.DataFrame(fr1, columns = ["FRCat"])

totalcatdf = pd.concat([uscatdf, cacatdf, decatdf, gbcatdf, frcatdf], axis =1 ,sort= False)
totalcatdf
type_int_list = ['views', 'likes', 'dislikes', 'comment_count']
for column in type_int_list:
    totaldf[column] = totaldf[column].astype(int)

type_str_list = ['category_id']
for column in type_str_list:
    totaldf[column] = totaldf[column].astype(str)

totaldf1 = totaldf.join(uscatdf,on = ['category_id'])
totaldf1.rename(columns = {"USCat" : "Category"}, inplace = True)
totaldf1.head()
temp1 = totaldf1.pivot_table(values = ['views', "likes"],  index = "Category").sort_values(by = "views", ascending = False)
temp1.iplot(kind = "bar", xTitle = "Category", yTitle = "Count", title = "Avg Views & Likes by Category" )
temp2 = totaldf1.pivot_table(values = ['views', "likes"],  index = "Category", aggfunc = np.sum ).sort_values(by = "views", ascending = False)
temp2.iplot(kind = "bar", xTitle = "Category", yTitle = "Count", title = "Total Views & Likes by Category" )
temp3 = totaldf1.pivot_table(values = ['views'],  index = "region", columns ="Category", aggfunc = np.mean )
temp3.iplot(kind = "bar", xTitle = "region", yTitle = "Count", title = "Avg Views by Category by Region" )
temp4 = totaldf1.pivot_table(values = ['views'],  index = "Category", columns ="region", aggfunc = np.mean )
temp4.iplot(kind = "bar", xTitle = "Category", yTitle = "Count", title = "Avg Views by Region by Category" )
correl_table = totaldf1[['views', 'likes', 'dislikes', 'comment_count']].corr()
correl_table
plt.figure(figsize = (8,8))
sns.heatmap(data = correl_table, annot = True)

totaldf1["publish_date"] = pd.to_datetime(totaldf1['publish_date'])
totaldf1["time to trend"] = totaldf1["trending_date"] - totaldf1["publish_date"]

totaldf1["time to trend (num)"] = totaldf1["time to trend"].dt.days
totaldf1.head()
totaldf1.head()
temp6 = totaldf1.pivot_table(values = ['time to trend (num)'],  index = "Category", columns ="region", aggfunc = np.mean )
temp6.iplot(kind = "bar", xTitle = "Category", yTitle = "Count", title = "Avg days to trend" )
temp7 = totaldf1.pivot_table(values = ['time to trend (num)'],  index = "region", columns ="Category", aggfunc = np.mean )
temp7.iplot(kind = "bar", xTitle = "region", yTitle = "Count", title = "Avg days to trend" )
temp8 = totaldf1.pivot_table(values = ['views'],  index = ["channel_title", 'Category'] , columns ="region", aggfunc = np.sum)
temp9= temp8.reset_index().fillna(0)
temp9['total views'] = temp9['views']['US'] + temp9['views']['CA'] + temp9['views']['GB'] + temp9['views']['FR'] + temp9['views']['DE']
temp9.sort_values(by = "total views", ascending = False, inplace = True)
temp10 = temp9.corr()
plt.figure(figsize = (10,10))
sns.heatmap( data = temp10, annot = True)
temp11 = totaldf1.pivot_table(values = ['views', 'time to trend (num)'],  index = 'Category' , columns ="region", aggfunc = np.mean)
temp12 = temp11.corr()
plt.figure(figsize = (10,10))
sns.heatmap( data = temp12, annot = True)
temp13 = totaldf1.pivot_table(values = ['views'],  index = ['publish_date'], columns = "Category", aggfunc = np.sum).tail(25)
temp13.iplot(kind = "bar", xTitle = "Date", yTitle = "View Count", title = "Total Views per Category in last 25 days")
temp13 = totaldf1.pivot_table(values = ['views'],  index = ['publish_date'], columns = "region", aggfunc = np.sum).tail(25)
temp13.iplot(kind = "bar", xTitle = "Date", yTitle = "View Count", title = "Total Views per Region in last 25 days")
from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC
from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
