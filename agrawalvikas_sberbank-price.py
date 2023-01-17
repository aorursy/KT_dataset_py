# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib as mpl

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import model_selection, preprocessing

import xgboost as xgb

color = sns.color_palette()



%matplotlib inline



pd.options.mode.chained_assignment = None  # default='warn'

pd.set_option('display.max_columns', 500)



# Any results you write to the current directory are saved as output.
train_df = pd.read_csv("../input/train.csv", parse_dates=['timestamp'])

test_df = pd.read_csv("../input/test.csv", parse_dates=['timestamp'])

macro_df = pd.read_csv("../input/macro.csv", parse_dates=['timestamp'])

#train_df = pd.merge(train_df, macro_df, how='left', on='timestamp')

#test_df = pd.merge(test_df, macro_df, how='left', on='timestamp')

print(train_df.shape, test_df.shape)
for f in train_df.columns:

    if train_df[f].dtype=='object':

        print(f)

        lbl = preprocessing.LabelEncoder()

        lbl.fit(list(train_df[f].values.astype('str')) + list(test_df[f].values.astype('str')))

        train_df[f] = lbl.transform(list(train_df[f].values.astype('str')))

        test_df[f] = lbl.transform(list(test_df[f].values.astype('str')))
# year and month #

train_df["yearmonth"] = train_df["timestamp"].dt.year*100 + train_df["timestamp"].dt.month

test_df["yearmonth"] = test_df["timestamp"].dt.year*100 + test_df["timestamp"].dt.month



# year and week #

train_df["yearweek"] = train_df["timestamp"].dt.year*100 + train_df["timestamp"].dt.weekofyear

test_df["yearweek"] = test_df["timestamp"].dt.year*100 + test_df["timestamp"].dt.weekofyear



# year #

train_df["year"] = train_df["timestamp"].dt.year

test_df["year"] = test_df["timestamp"].dt.year



# month of year #

train_df["month_of_year"] = train_df["timestamp"].dt.month

test_df["month_of_year"] = test_df["timestamp"].dt.month



# week of year #

train_df["week_of_year"] = train_df["timestamp"].dt.weekofyear

test_df["week_of_year"] = test_df["timestamp"].dt.weekofyear



# day of week #

train_df["day_of_week"] = train_df["timestamp"].dt.weekday

test_df["day_of_week"] = test_df["timestamp"].dt.weekday
train_df=train_df.drop(['timestamp'],axis=1)

test_df=test_df.drop(['timestamp'],axis=1)
train_id=train_df['id']

test_id=test_df['id']
train_df=train_df.drop(['id'],axis=1)

test_df=test_df.drop(['id'],axis=1)
train_df=train_df.fillna(0)

test_df=test_df.fillna(0)
from sklearn.decomposition import PCA, FastICA

n_comp = 10



# PCA

pca = PCA(n_components=n_comp, random_state=42)

pca2_results_train = pca.fit_transform(train_df.drop(["price_doc"], axis=1))

pca2_results_test = pca.transform(test_df)



# ICA

ica = FastICA(n_components=n_comp, random_state=42)

ica2_results_train = ica.fit_transform(train_df.drop(["price_doc"], axis=1))

ica2_results_test = ica.transform(test_df)



# Append decomposition components to datasets

for i in range(1, n_comp+1):

    train_df['pca_' + str(i)] = pca2_results_train[:,i-1]

    test_df['pca_' + str(i)] = pca2_results_test[:, i-1]

    

    train_df['ica_' + str(i)] = ica2_results_train[:,i-1]

    test_df['ica_' + str(i)] = ica2_results_test[:, i-1]
train_df.head(4)
test_df.head(3)
y_train = train_df["price_doc"]
train_df.shape,test_df.shape
train_df = train_df.drop(["price_doc"], axis=1)
train_df.shape,test_df.shape
xgb_params = {

    'eta': 0.05,

    'max_depth': 5,

    'subsample': 0.7,

    'colsample_bytree': 0.7,

    'objective': 'reg:linear',

    'eval_metric': 'rmse',

    'silent': 1

}
dtrain = xgb.DMatrix(train_df, y_train)

dtest = xgb.DMatrix(test_df)
cv_result = xgb.cv(xgb_params, 

                   dtrain, 

                   num_boost_round=900, # increase to have better results (~700)

                   early_stopping_rounds=50,

                   verbose_eval=10, 

                   show_stdv=False

                  )
num_boost_rounds = len(cv_result)

print(num_boost_rounds)



# train model

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=num_boost_rounds)
# plot the important features #

fig, ax = plt.subplots(figsize=(12,25))

xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

plt.show()
y_pred = model.predict(dtest)
# plot the important features #

fig, ax = plt.subplots(figsize=(12,25))

xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

plt.show()
output= pd.DataFrame({'id' : test_id, 'price_doc' : y_pred})

output.to_csv('preds_pca.csv', index=False)