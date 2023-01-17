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
# Essentials

import numpy as np

import pandas as pd

import datetime

import random



# Plots

import seaborn as sns

import matplotlib.pyplot as plt



# Models

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor

from sklearn.kernel_ridge import KernelRidge

from sklearn.linear_model import Ridge, RidgeCV

from sklearn.linear_model import ElasticNet, ElasticNetCV

from sklearn.svm import SVR

from mlxtend.regressor import StackingCVRegressor

import lightgbm as lgb

from lightgbm import LGBMRegressor

from xgboost import XGBRegressor



# Stats

from scipy.stats import skew, norm

from scipy.special import boxcox1p

from scipy.stats import boxcox_normmax



# Misc

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import scale

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import RobustScaler

from sklearn.decomposition import PCA



pd.set_option('display.max_columns', None)



# Ignore useless warnings

import warnings

warnings.filterwarnings(action="ignore")
train = pd.read_csv('../input/restaurant-revenue-prediction/train.csv.zip')

test = pd.read_csv('../input/restaurant-revenue-prediction/test.csv.zip')

train.shape, test.shape
train.head()
train.info()
test.info()
#Distribution of revenue

sns.set()

f, ax = plt.subplots(figsize=(15, 5))

sns.distplot(train['revenue'], color="b");

ax.xaxis.grid(False)

ax.set(ylabel="Frequency")

ax.set(xlabel="revenue")

ax.set(title="revenue distribution")

sns.despine(trim=True, left=True)

plt.show()
combine = [train, test]



for dataset in combine:

    dataset["Open Date"] = pd.to_datetime(dataset["Open Date"])

    dataset["Standard_date"] = "2015-04-27"

    dataset["Standard_date"] = pd.to_datetime(dataset["Standard_date"])

    dataset["Business_Period"] = (dataset["Standard_date"]-dataset["Open Date"]).apply(lambda x: x.days)
train.head()
train["Type"].unique()
test["Type"].unique()
train["City"].unique()
test["City"].unique()
data2 = pd.DataFrame()

data2["City_name"] = train["City"].unique()

data3 = pd.DataFrame()

data3["City_name"] = test["City"].unique()



data_city = pd.concat([data2,data3])

data_city["City_name"].unique()
train["City Group"].unique()
#dammy



train["Type"] = train["Type"].map({"IL":0,"FC":1,"DT":2}).astype(int)

test["Type"] = test["Type"].map({"IL":0,"FC":1,"DT":2,"MB":3}).astype(int)



train["City Group"] = train["City Group"].map({"Big Cities":0, "Other":1}).astype(int)

test["City Group"] = test["City Group"].map({"Big Cities":0, "Other":1}).astype(int)



train.head()
train["City"] = train["City"].map({'İstanbul':0, 'Ankara':1, 'Diyarbakır':2, 'Tokat':3, 'Gaziantep':4,

       'Afyonkarahisar':5, 'Edirne':6, 'Kocaeli':7, 'Bursa':8, 'İzmir':9, 'Sakarya':10,

       'Elazığ':11, 'Kayseri':12, 'Eskişehir':13, 'Şanlıurfa':14, 'Samsun':15, 'Adana':16,

       'Antalya':17, 'Kastamonu':18, 'Uşak':19, 'Muğla':20, 'Kırklareli':21, 'Konya':22,

       'Karabük':23, 'Tekirdağ':24, 'Denizli':25, 'Balıkesir':26, 'Aydın':27, 'Amasya':28,

       'Kütahya':29, 'Bolu':30, 'Trabzon':31, 'Isparta':32, 'Osmaniye':33, 'Niğde':34,

       'Rize':35, 'Düzce':36, 'Hatay':37, 'Erzurum':38, 'Mersin':39, 'Zonguldak':40,

       'Malatya':41, 'Çanakkale':42, 'Kars':43, 'Batman':44, 'Bilecik':45, 'Giresun':46,

       'Sivas':47, 'Kırıkkale':48, 'Mardin':49, 'Erzincan':50, 'Manisa':51,

       'Kahramanmaraş':52, 'Yalova':53, 'Tanımsız':54, 'Kırşehir':55, 'Aksaray':56,

       'Nevşehir':57, 'Çorum':58, 'Ordu':59, 'Artvin':60, 'Siirt':61, 'Çankırı':62}).astype(int)



test["City"] = test["City"].map({'İstanbul':0, 'Ankara':1, 'Diyarbakır':2, 'Tokat':3, 'Gaziantep':4,

       'Afyonkarahisar':5, 'Edirne':6, 'Kocaeli':7, 'Bursa':8, 'İzmir':9, 'Sakarya':10,

       'Elazığ':11, 'Kayseri':12, 'Eskişehir':13, 'Şanlıurfa':14, 'Samsun':15, 'Adana':16,

       'Antalya':17, 'Kastamonu':18, 'Uşak':19, 'Muğla':20, 'Kırklareli':21, 'Konya':22,

       'Karabük':23, 'Tekirdağ':24, 'Denizli':25, 'Balıkesir':26, 'Aydın':27, 'Amasya':28,

       'Kütahya':29, 'Bolu':30, 'Trabzon':31, 'Isparta':32, 'Osmaniye':33, 'Niğde':34,

       'Rize':35, 'Düzce':36, 'Hatay':37, 'Erzurum':38, 'Mersin':39, 'Zonguldak':40,

       'Malatya':41, 'Çanakkale':42, 'Kars':43, 'Batman':44, 'Bilecik':45, 'Giresun':46,

       'Sivas':47, 'Kırıkkale':48, 'Mardin':49, 'Erzincan':50, 'Manisa':51,

       'Kahramanmaraş':52, 'Yalova':53, 'Tanımsız':54, 'Kırşehir':55, 'Aksaray':56,

       'Nevşehir':57, 'Çorum':58, 'Ordu':59, 'Artvin':60, 'Siirt':61, 'Çankırı':62}).astype(int)
train.head()
train["Business_Period"].hist()
train.describe()
#log



train["revenue"] = train.revenue.apply(lambda x:np.log1p(x))
train = train.drop(["Id"], axis=1)

train = train.drop(["Open Date"], axis=1)

train = train.drop(["Standard_date"], axis=1)





test = test.drop(["Id"], axis=1)

test = test.drop(["Open Date"], axis=1)

test = test.drop(["Standard_date"], axis=1)
y = train["revenue"]

train = train.drop(["revenue"], axis=1)
train.head()
test.head()
train.shape
test.shape
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

    train, y, random_state=0, train_size=0.7,shuffle=False)
# lightGBMによる予測

lgb_train = lgb.Dataset(X_train, y_train)

lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)



# LightGBM parameters

params = {

        'task' : 'train',

        'boosting_type' : 'gbdt',

        'objective' : 'regression',

        'metric' : {'l2'},

        'num_leaves' : 31,

        'learning_rate' : 0.1,

        'feature_fraction' : 0.9,

        'bagging_fraction' : 0.8,

        'bagging_freq': 5,

        'verbose' : 0,

        'n_jobs': 2

}



gbm = lgb.train(params,

            lgb_train,

            num_boost_round=100,

            valid_sets=lgb_eval,

            early_stopping_rounds=10)



prediction_lgb = np.exp(gbm.predict(test))
sample =pd.read_csv('/kaggle/input/restaurant-revenue-prediction/sampleSubmission.csv')
sample["Prediction"] = prediction_lgb
sample.to_csv('../working/submission.csv', index = False)