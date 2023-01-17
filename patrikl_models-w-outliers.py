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
from kaggle.competitions import twosigmanews

# You can only call make_env() once, so don't lose it!

env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls



from wordcloud import WordCloud

from nltk.corpus import stopwords

stop = set(stopwords.words('english'))



import datetime

import lightgbm as lgb

from scipy import stats

from scipy.sparse import hstack, csr_matrix

from sklearn.model_selection import train_test_split

from nltk.util import ngrams

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import StandardScaler
# fill with 0s, but knowing it won't be good

# and still having outliers

market_train_df['returnsClosePrevMktres1'].fillna(0, inplace=True)



market_train_df.head()
sns.boxplot(x=market_train_df['returnsOpenNextMktres10'])
# take all continuous; maybe remove those returns not Savitzky-Golay-ed, or use it on them too.

X = market_train_df[['volume', 'close', 'open', 'returnsClosePrevRaw1','returnsOpenPrevRaw1', 'returnsClosePrevRaw10', 'returnsOpenPrevRaw10']]

y = market_train_df['returnsOpenNextMktres10']



Z = market_train_df[['close', 'open', 'volume', 'returnsOpenPrevMktres10']] # for correlation
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
from sklearn.linear_model import LinearRegression



slr = LinearRegression()

slr.fit(X_train, y_train)

print('Intercept: %.3f' % slr.intercept_)

print('Beta 1:  %.3f' % slr.coef_[0])

print('Beta 2:  %.3f' % slr.coef_[1])

print('Beta 3:  %.3f' % slr.coef_[2])



y_pred = slr.predict(X_test)

print(y_pred[:5])



from sklearn.metrics import mean_absolute_error

print(mean_absolute_error(y_test, y_pred))
from sklearn.ensemble import RandomForestRegressor



forest = RandomForestRegressor(n_estimators=500, 

                               random_state=123,

                               max_depth=3,

                               n_jobs=-1)

forest.fit(X_train, y_train)

y_pred = forest.predict(X_test)



from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, y_pred)
import xgboost as xgb



#Fitting XGB regressor 

model = xgb.XGBRegressor(n_estimators=500, 

                         n_jobs=-1,

                         max_depth=3,

                         random_state=123)

model.fit(X_train, y_train)

preds = model.predict(X_test)



from sklearn.metrics import mean_absolute_error

mean_absolute_error(y_test, preds)
model.score(X_test, y_test) # explained variance?
from sklearn.metrics import explained_variance_score # why

print(explained_variance_score(y_test, preds))