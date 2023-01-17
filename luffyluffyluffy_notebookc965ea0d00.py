import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))


## importing library

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from IPython.display import FileLinks

import warnings

warnings.filterwarnings('ignore')

from sklearn.pipeline import *

from sklearn.preprocessing import *

from tpot import TPOTRegressor

from tpot.builtins import StackingEstimator

from sklearn.linear_model import *

from sklearn.neighbors import *

from sklearn.model_selection import *

from sklearn.metrics import *

from sklearn.ensemble import *

from catboost import CatBoostRegressor

from lightgbm import LGBMRegressor
##cross validation to ensure that model doen't overfit

def cross(model, data, y, n_folds=10):

    y_pred = np.array([0]*len(df))

    kf = KFold(n_splits=n_folds)

    ii=0

    for train_idx, test_idx in kf.split(data):

        train = data.iloc[train_idx, :]

        test = data.iloc[test_idx, :]

        ytn = y[train_idx]

        yts = y[test_idx]

        model.fit(train, ytn)

        y_pred[test_idx] = model.predict(test).astype(int)

#         print(ii)

        ii+=1

    return y_pred

##cross validation to ensure that model doen't overfit

def cross1(model, data, y, n_folds=10):

    y_pred = np.array([0]*len(df))

    kf = KFold(n_splits=n_folds)

    ii=0

    for train_idx, test_idx in kf.split(data):

        train = data.iloc[train_idx, :]

        test = data.iloc[test_idx, :]

        ytn = y[train_idx]

        yts = y[test_idx]

        model.fit(train, ytn)

        y_pred[test_idx] = model.predict(test).astype(int)

        print(ii)

        ii+=1

    return y_pred

def rmse(a, b):

    return np.sqrt(mean_squared_error(a, b))

def neg_rmse(a, b):

    return -1*np.sqrt(mean_squared_error(a, b))

train = pd.read_csv('../input/kdag-ms/train.csv')

test = pd.read_csv('../input/kdag-ms/test.csv')

sam = pd.read_csv('../input/kdag-ms/sample.csv')

sam = pd.DataFrame(columns=sam.columns, index=test.index)

sam.iloc[:, 0] = test.iloc[:, 0]

train.drop('v.id', axis=1, inplace=True)

test.drop('v.id', axis=1, inplace=True)



df = train

cata = []

data = df

df.head()

drop = ['current price', ]

y = df['current price']

df = df.drop(drop, axis=1)

# gbr = GradientBoostingRegressor(loss ='ls', max_depth=3)

# rmse(y, cross(gbr, df, y))
# df['aa']  = (df['current price'] - df['on road now'])/-1

# df['use'] = (1/df['km'])*(df['rating'])*df['condition']*1000
for i in df.columns:

    print(i, df[i].unique().shape)
cata = ['years', 'rating', 'condition', 'economy']
df.corr()
# import seaborn as sns

# i=6

# sns.distplot(dat[df.columns[i]])
# sns.distplot(df[df.columns[i]])
# cc = ['on road old', 'on road now', 'years', 'km']

# data = df[cc]

# tt = test[cc]
cata

for i  in df.columns:

    print(i, df[i].dtype)

    if df[i].dtype == 'object':

        cata.append(i)
cata
### processing categorical data

from sklearn.preprocessing import LabelEncoder



# for i in cata:

#     le = LabelEncoder()

#     df[i] = le.fit_transform(df[i])

    

df.head()
lg = LGBMRegressor()
lg = RandomForestRegressor(bootstrap=True, max_features=0.7500000000000001, min_samples_leaf=11, min_samples_split=9, n_estimators=100)
conti = list(df.columns)

for i in cata:

    conti.remove(i)
conti
def get_dum(df, cata):

    dum = pd.DataFrame(columns=[], index=df.index)

    for i in cata:

        dum = pd.concat([dum, pd.get_dummies(df[i])], axis=1)

    dum.columns = range(500, 500+dum.shape[1])

    dum.head()



    df = df.drop(cata, axis=1)

    df =  pd.concat([df, dum], axis=1)

    print(df.shape)

    return df



test = get_dum(test, cata)

df = get_dum(df, cata)

test.head()
df.head()
df.head()
from sklearn.preprocessing import StandardScaler

for i in conti:

    sc = StandardScaler()

    df[i] = sc.fit_transform(np.array(df[i]).reshape(-1, 1))

    test[i] = sc.transform(np.array(test[i]).reshape(-1, 1))



df.head()
from sklearn.model_selection import *

df.head()
from sklearn.linear_model import *

lr = LinearRegression()
train_copy =train.copy() 
lg
lr = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=-1, normalize=False)
# yc= sc.fit_transform(np.array(y).reshape(-1, 1))

# yc = yc[:, 0]
df.head()
model = lr



train = df.copy()

cols = train.columns

y_pred = cross(lr, train, y)

origina = rmse(y, cross(model, train, y))

for i in cols:

    col = train.columns

    for i in col:

        c = train[i]

        train = train.drop(i, axis=1)

        new_error = rmse(y, cross(model, train, y))

#         print("ori", origina, "new", new_error,)

        if(new_error < origina):

            origina = new_error

            column_to_remove = i

        train[i] = c

#         print("done", i)

    try:

        print("--> removing", column_to_remove)

        train = train.drop(column_to_remove, axis=1)

    except:

        break
rmse(y, cross(model, df, y))
# df = df[train.columns]

# test = test[train.columns]

# df.head()
# count_plot(df['km'])
test
lr.fit(df, y)

y_pred = lr.predict(test)
test
sam
len(test)
sam.iloc[:, 1] = y_pred

sam.to_csv('subb.csv', index = False)

FileLinks('.')