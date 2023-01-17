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
print('Done!')
import pandas as pd
import numpy as np
import time
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
(market_train_df, news_train_df) = env.get_training_data()
market_train, news_train = market_train_df.copy(), news_train_df.copy()
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
%matplotlib inline
import seaborn as sns
import numpy as np
import plotly.figure_factory as ff
import plotly.plotly as py
import plotly.graph_objs as go
%%time
def data_prep(market_train,news_train):
    market_train.time = market_train.time.dt.date
    news_train.time = news_train.time.dt.hour
    news_train.sourceTimestamp= news_train.sourceTimestamp.dt.hour
    news_train.firstCreated = news_train.firstCreated.dt.date
    news_train['assetCodesLen'] = news_train['assetCodes'].map(lambda x: len(eval(x)))
    news_train['assetCodes'] = news_train['assetCodes'].map(lambda x: list(eval(x))[0])
    kcol = ['firstCreated', 'assetCodes']
    news_train = news_train.groupby(kcol, as_index=False).mean()
    market_train = pd.merge(market_train, news_train, how='left', left_on=['time', 'assetCode'], 
                            right_on=['firstCreated', 'assetCodes'])
    lbl = {k: v for v, k in enumerate(market_train['assetCode'].unique())}
    market_train['assetCodeT'] = market_train['assetCode'].map(lbl)
    
    
    market_train = market_train.dropna(axis=0)
    
    return market_train

market_train = data_prep(market_train_df, news_train_df)
market_train.shape
%%time
from datetime import datetime, date
# The target is binary
market_train = market_train.loc[market_train['time_x']>=date(2009, 1, 1)]
up = market_train.returnsOpenNextMktres10 >= 0
fcol = [c for c in market_train if c not in ['assetCode', 'assetCodes', 'assetCodesLen', 'assetName', 'audiences', 
                                             'firstCreated', 'headline', 'headlineTag', 'marketCommentary', 'provider', 
                                             'returnsOpenNextMktres10', 'sourceId', 'subjects', 'time', 'time_x', 'universe','sourceTimestamp']]
%%time
# We still need the returns for model tuning
X = market_train[fcol].values
up = up.values
r = market_train.returnsOpenNextMktres10.values

# Scaling of X values
# It is good to keep these scaling values for later
mins = np.min(X, axis=0)
maxs = np.max(X, axis=0)
rng = maxs - mins
X = 1 - ((maxs - X) / rng)

# Sanity check
assert X.shape[0] == up.shape[0] == r.shape[0]
%%time
from xgboost import XGBClassifier
from sklearn import model_selection
from sklearn.metrics import accuracy_score
import time

X_train, X_test, up_train, up_test, r_train, r_test = model_selection.train_test_split(X, up, r, test_size=0.25, random_state=99)
xgb_up = XGBClassifier(n_jobs=4,n_estimators=250,max_depth=9,eta=0.08)
t = time.time()
print('Fitting Up')
xgb_up.fit(X_train,up_train)
print(f'Done, time = {time.time() - t}')
from sklearn.metrics import accuracy_score
accuracy_score(xgb_up.predict(X_test),up_test)
