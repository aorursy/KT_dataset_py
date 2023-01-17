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
%reload_ext autoreload

%autoreload 2

%matplotlib inline



!pip install fastai==1.0.52

import fastai



from fastai import *

from fastai.vision import *

from fastai.tabular import *



# from torchvision.models import *

# import pretrainedmodels



from utils import *

import sys



from fastai.callbacks.hooks import *



from fastai.callbacks.tracker import EarlyStoppingCallback

from fastai.callbacks.tracker import SaveModelCallback
from scipy.special import erfinv

import matplotlib.pyplot as plt

import torch

from torch.utils.data import *

from torch.optim import *

from fastai.tabular import *

import torch.utils.data as Data

from fastai.basics import *

from fastai.callbacks.hooks import *

from tqdm import tqdm_notebook as tqdm
def to_gauss(x): return np.sqrt(2)*erfinv(x)  #from scipy



def normalize(data, exclude=None):

    # if not binary, normalize it

    norm_cols = [n for n, c in data.drop(exclude, 1).items() if len(np.unique(c)) > 2]

    n = data.shape[0]

    for col in norm_cols:

        sorted_idx = data[col].sort_values().index.tolist()# list of sorted index

        uniform = np.linspace(start=-0.99, stop=0.99, num=n) # linsapce

        normal = to_gauss(uniform) # apply gauss to linspace

        normalized_col = pd.Series(index=sorted_idx, data=normal) # sorted idx and normalized space

        data[col] = normalized_col # column receives its corresponding rank

    return data
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")
df_train.head().T
df_test.head().T
df_train.info()
df_test.info()
add_datepart(df_train, "datetime", drop=False)

add_datepart(df_test, "datetime", drop=False)
df_train.head().T
df_train['season'] = df_train.season.map({1: 'spring', 2: 'summer', 3: 'fall', 4: 'winter'})

df_test['season'] = df_test.season.map({1: 'spring', 2: 'summer', 3: 'fall', 4: 'winter'})



df_train['holiday'] = df_train.holiday.map({0: 'non-holiday', 1: 'holiday'})

df_test['holiday'] = df_test.holiday.map({0: 'non-holiday', 1: 'holiday'})



df_train['workingday'] = df_train.workingday.map({0: 'holiday', 1: 'workingday'})

df_test['workingday'] = df_test.workingday.map({0: 'holiday', 1: 'workingday'})



df_train["weather"] = df_train.weather.map({1: " Clear + Few clouds + Partly cloudy + Partly cloudy",\

                                        2 : " Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist ", \

                                        3 : " Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds", \

                                        4 :" Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog " })



df_test["weather"] = df_test.weather.map({1: " Clear + Few clouds + Partly cloudy + Partly cloudy",\

                                        2 : " Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist ", \

                                        3 : " Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds", \

                                        4 :" Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog " })
df_train.head().T
df_train.info()
df_test.info()
procs=[FillMissing, Categorify]



cat_vars = ['season', 'holiday', 'workingday', 'weather', 'datetimeYear', 'datetimeMonth',

           'datetimeWeek', 'datetimeDay', 'datetimeDayofweek']



cont_vars = ['temp', 'atemp', 'humidity', 'windspeed', 'casual', 'registered']



dep_var = 'count'
df_train.head().T
df_train = normalize(df_train, exclude=['season', 'holiday', 'workingday', 'weather', 'datetimeYear', 'datetimeMonth',

           'datetimeWeek', 'datetimeDay', 'datetimeDayofweek', 'count', 'datetime'])
df_train.head().T
df = df_train[cat_vars + cont_vars + [dep_var,'datetime']].copy()

df.head().T
df_train['datetime'].min(), df_train['datetime'].max()
df_test['datetime'].min(), df_test['datetime'].max()
len(df_test), len(df_train)
path = Path("../input/")
np.random.seed(42)



data = (TabularList.from_df(df, path=path, cat_names=cat_vars, cont_names=cont_vars, procs=procs)

                .split_by_rand_pct(0.2, seed=42)

                .label_from_df(cols=dep_var, label_cls=FloatList)

                .databunch(bs=1024))
data.show_batch()
learn = tabular_learner(data, layers=[1000,500], metrics=mean_squared_error, model_dir="../temp/model",

                        ps=[0.1, 0.1], emb_drop=0.04)
learn.lr_find()

learn.recorder.plot(suggestion=True)
lr = 1e-1

learn.fit_one_cycle(5, max_lr=lr, wd=0.2, pct_start=0.3)
learn.lr_find()

learn.recorder.plot(suggestion=True)
lr = 1e-4

learn.fit_one_cycle(5, lr, wd=0.2, pct_start=0.3)
learn.lr_find()

learn.recorder.plot(suggestion=True)
lr=5e-6

learn.fit_one_cycle(5, max_lr=lr, wd=0.2)
learn.save('1')

learn.recorder.plot_losses()
class SaveFeatures():

    features=None

    def __init__(self, m): 

        self.hook = m.register_forward_hook(self.hook_fn)

        self.features = None

    def hook_fn(self, module, input, output): 

        out = output.detach().cpu().numpy()

        if isinstance(self.features, type(None)):

            self.features = out

        else:

            self.features = np.row_stack((self.features, out))

    def remove(self): 

        self.hook.remove()
learn.model
sf = SaveFeatures(learn.model.layers[4])
_= learn.get_preds(data.train_ds)
label = [x for x in (list(data.train_ds.y.items))]
len(label)
df_new = pd.DataFrame({'label': label})
df_new.head()
array = np.array(sf.features)
x=array.tolist()
df_new['img_repr'] = x
df_new.head()
d2 = pd.DataFrame(df_new.img_repr.values.tolist(), index = df_new.index).rename(columns = lambda x: 'img_repr{}'.format(x+1))
df_new_2 = df_new.join(d2)
df_new_2.head(10).T
df_new_2.shape

sf = SaveFeatures(learn.model.layers[4])
_=learn.get_preds(DatasetType.Valid)
label = [x for x in (list(data.valid_ds.y.items))]
df_new_valid = pd.DataFrame({'label': label})

array = np.array(sf.features)
x=array.tolist()

df_new_valid['img_repr'] = x

df_new_valid.head()
d2 = pd.DataFrame(df_new_valid.img_repr.values.tolist(), index = df_new_valid.index).rename(columns = lambda x: 'img_repr{}'.format(x+1))
df_new_valid_2 = df_new_valid.join(d2)

df_new_valid_2.head(10)
df_new_valid_2.shape
df_new_valid_2.drop(['img_repr'], axis=1, inplace=True)
df_new_valid_2.head()
df_new_2.drop(['img_repr'], axis=1, inplace=True)
df_new_2.shape
df_new_2.describe()
corr_matrix = df_new_2.corr()



corr_matrix["label"].sort_values(ascending = False)
X = df_new_2

y = df_new_2.label.copy()



from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
X_train = X_train.drop("label", axis =1)

y_train = y_train



X_test = X_test.drop("label", axis =1)

y_test = y_test
X_train.shape, y_train.shape, X_test.shape, y_test.shape
from sklearn.base import BaseEstimator, TransformerMixin



class DataFrameSelector(BaseEstimator, TransformerMixin):

    

    def __init__(self, attributes_names):

        self.attributes_names = attributes_names

        

    def fit(self, X, y=None):

        return self

    

    def transform(self, X):

        return X[self.attributes_names].values
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



# numerical pipeline



num_pipeline = Pipeline([

    

    ('select_data', DataFrameSelector(X_train.columns)),

    ('Std_Scaler', StandardScaler())

])



X_train_transformed = num_pipeline.fit_transform(X_train)

X_test_transformed = num_pipeline.fit_transform(X_test)
X_train_transformed.shape, X_test_transformed.shape
from sklearn.ensemble import RandomForestRegressor

import time



start = time.time()



rf_clf = RandomForestRegressor(bootstrap=True,

            criterion='mse', max_depth=15, max_features=0.5,

            max_leaf_nodes=None, min_impurity_decrease=0.0,

            min_impurity_split=None, min_samples_leaf=3,

            min_samples_split=8, min_weight_fraction_leaf=0.0,

            n_estimators=185, n_jobs=1, oob_score=False, random_state=42,

            verbose=0, warm_start=False)



rf_clf.fit(X_train_transformed, y_train)



end = time.time()



print("run_time:", (end-start)/(60*60))
# import scipy.stats as st

# from sklearn.model_selection import RandomizedSearchCV



# one_to_left = st.beta(10, 1)  

# from_zero_positive = st.expon(0, 50)



# params = {  

#     "n_estimators": st.randint(50, 300),

#     "max_depth": st.randint(3, 40),

#    "min_samples_leaf": st.randint(3, 40),

#     "min_samples_split": st.randint(3, 20),

#     "max_features": ['auto', 0.2, 0.3, 0.5]

# }



# gs = RandomizedSearchCV(rf_clf, params)
# gs.fit(X_train_transformed, y_train)  
# gs.best_params_
from sklearn.model_selection import cross_val_predict, cross_val_score



import time



start = time.time()



score_rf = cross_val_score(rf_clf, X_train_transformed, y_train, cv=5, scoring='neg_mean_squared_error', verbose=0)

print(score_rf.mean())



end = time.time()



print("run_time:", (end-start)/(60*60))
y_pred_test_rf = rf_clf.predict(X_test_transformed)
from sklearn.metrics import mean_squared_error

mean_squared_error(y_test, y_pred_test_rf)
X = df_new_valid_2

y = df_new_valid_2.label.copy()
X_val = X.drop("label", axis =1)

y_val = y
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



# numerical pipeline



num_pipeline = Pipeline([

    

    ('select_data', DataFrameSelector(X_val.columns)),

    ('Std_Scaler', StandardScaler())

])





X_val_transformed = num_pipeline.fit_transform(X_val)
y_pred_test_rf_val = rf_clf.predict(X_val_transformed)
from sklearn.metrics import mean_squared_error

mean_squared_error(y_val, y_pred_test_rf_val)