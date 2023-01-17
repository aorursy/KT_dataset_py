# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np # linear algebra

import pandas as pd
train = pd.read_csv("/kaggle/input/cte-ml-hack-2019/train_real.csv")

test = pd.read_csv("/kaggle/input/cte-ml-hack-2019/test_real.csv")
def feature(df):

  #df = df[(df[['Hillshade_9am','Hillshade_Noon','Hillshade_3pm']] != 0).all(axis=1)]

  df[['Hillshade_9am','Hillshade_Noon','Hillshade_3pm']]=df[['Hillshade_9am','Hillshade_Noon','Hillshade_3pm']].mask(df[['Hillshade_9am','Hillshade_Noon','Hillshade_3pm']]==0).fillna(df[['Hillshade_9am','Hillshade_Noon','Hillshade_3pm']].median())

  df['Hillshade_9am'] = np.log(df['Hillshade_9am'])

  df['Hillshade_Noon'] = np.log(df['Hillshade_Noon'])

  df['Hillshade_3pm'] = np.log(df['Hillshade_3pm'])

  df['Hillshade_mean'] = 0.33*(df['Hillshade_9am']+df['Hillshade_Noon']+df['Hillshade_3pm'])

  df['Euclidean_Distance_To_Hydrology'] = np.sqrt(np.power(df['H_dist_Hydro'],2) + np.power(df['V_dist_Hydro'],2))

  df['log_elevation'] = np.log(df['Altitude'])

  df['Hillshade_9am_sq'] = np.power(df['Hillshade_9am'],2)

  df['Hillshade_Noon_sq'] = np.power(df['Hillshade_Noon'],2)

  df['Hillshade_3pm_sq'] = np.power(df['Hillshade_3pm'],2)

  df['cosine_slope'] = np.cos(df['Incline'])

  df['interaction_9amnoon'] = df['Hillshade_9am']*df['Hillshade_Noon']

  df['interaction_noon3pm'] = df['Hillshade_3pm']*df['Hillshade_Noon']

  df['interaction_9am3pm'] = df['Hillshade_9am']*df['Hillshade_3pm']

  df['Elev_to_HD_Hyd'] = df['Altitude'] - (0.2*df['H_dist_Hydro'])

  df['Elev_to_VD_Hyd'] = df['Altitude'] - (df['V_dist_Hydro'])

  df['Elev_to_HD_Road'] = df['Altitude'] - (0.05*df['H_dist_Road'])

  df['HR1'] = abs(df['H_dist_Hydro']+df['H_dist_Road'])

  df['HR2'] = abs(df['H_dist_Hydro']-df['H_dist_Road'])

  df['FR1'] = abs(df['H_dist_Fire']+df['H_dist_Road'])

  df['FR2'] = abs(df['H_dist_Fire']-df['H_dist_Road'])

  df['HF1'] = abs(df['H_dist_Hydro']+df['H_dist_Fire'])

  df['HF2'] = abs(df['H_dist_Hydro']-df['H_dist_Fire'])

  return df
train = feature(train)

test = feature(test)
def split_train(df):

  X_train = df.drop(['Id', 'label','Soil'], axis=1)

  Y_train = df['label']

  

  return X_train, Y_train
X_train, Y_train = split_train(train)
def split_test(df):

  X_train = df.drop(['Id','Soil'], axis=1)

  

  return X_train
X_comp = split_test(test)
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import LassoLarsCV

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsRegressor

from sklearn.pipeline import make_pipeline, make_union

from sklearn.preprocessing import RobustScaler

from tpot.builtins import StackingEstimator, ZeroCount
exported_pipeline = make_pipeline(

    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.99, learning_rate=0.001, loss="huber", max_depth=5, max_features=0.15000000000000002, min_samples_leaf=19, min_samples_split=3, n_estimators=100, subsample=0.25)),

    StackingEstimator(estimator=LassoLarsCV(normalize=False)),

    RobustScaler(),

    StackingEstimator(estimator=GradientBoostingRegressor(alpha=0.99, learning_rate=0.01, loss="huber", max_depth=3, max_features=0.25, min_samples_leaf=2, min_samples_split=4, n_estimators=100, subsample=0.55)),

    ZeroCount(),

    KNeighborsRegressor(n_neighbors=96, p=1, weights="distance")

)
exported_pipeline.fit(X_train, Y_train)
pred = exported_pipeline.predict(X_comp)
sub = pd.DataFrame()

sub['Id'] = test['Id']

sub['Predicted'] = pred
from IPython.display import HTML

import pandas as pd

import numpy as np

import base64



# function that takes in a dataframe and creates a text link to  

# download it (will only work for files < 2MB or so)

def create_download_link(df, title = "Download CSV file", filename = "data.csv"):  

    csv = df.to_csv(index=False)

    b64 = base64.b64encode(csv.encode())

    payload = b64.decode()

    html = '<a download="{filename}" href="data:text/csv;base64,{payload}" target="_blank">{title}</a>'

    html = html.format(payload=payload,title=title,filename=filename)

    return HTML(html)
create_download_link(sub)