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

df=pd.read_csv('../input/airbnb-price-prediction/train.csv')

print(df.shape)

df.head()
df=df.fillna(0)
df.describe()
df[['property_type','log_price']].groupby(['property_type']).mean().sort_values(by='log_price',ascending=False)[0:10]
df[['city','log_price']].groupby(['city']).mean().sort_values(by='log_price',ascending=False)[0:10]
np.corrcoef(df['accommodates'],df['log_price'])
df[['accommodates','log_price']].groupby(['accommodates']).mean().sort_values(by='log_price',ascending=False)#[0:10]
def lat_center(row):

    if (row['city']=='NYC'):

        return 40.72

def long_center(row):

    if (row['city']=='NYC'):

        return -74

    

df['lat_center']=df.apply(lambda row:lat_center(row),axis=1)

df['long_center']=df.apply(lambda row:long_center(row),axis=1)
df['distance_to_center']=np.sqrt((df['lat_center']-df['latitude'])**2+(df['long_center']-df['longitude'])**2)
pd.options.mode.chained_assignment = None

ny=df[df['city']=='NYC']

lat_ny=40.72

long_ny=-74

ny['distance to center']=np.sqrt((lat_ny-ny['latitude'])**2+(long_ny-ny['longitude'])**2)
import seaborn as sns


soho_vs_price=sns.regplot(x=ny['distance to center'],y=ny['log_price'],fit_reg=True)

print (np.corrcoef(ny['distance to center'], ny['log_price']))
print (np.corrcoef(ny['beds'], ny['accommodates']))
categorical=['property_type','room_type','bed_type','cancellation_policy']

ny_model=pd.get_dummies(ny, columns=categorical)
numerics=['uint8','int16', 'int32', 'int64', 'float16', 'float32', 'float64']

ny_train_x=ny_model.select_dtypes(include=numerics).drop('log_price',axis=1).fillna(0).values

ny_train_y=ny_model['log_price'].values

from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestRegressor

cv_groups=KFold(n_splits=3)

regr=RandomForestRegressor(random_state=0,n_estimators=10)



for train_index,test_index in cv_groups.split(my_train_x):

    regr.fit(ny_train_x[train_index], ny_train_y[train_index])

    pred_rf=regr.predict(ny_train_x[test_index])

    rmse=str(np.sqrt(np.mean((ny_train_y[test_index]-pred_rf)**2)))

    print("RMSE for current split: " + rmse)
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

from sklearn.model_selection import cross_val_score

import matplotlib.pyplot as plt



def rmse_cv(model):

    rmse=np.sqrt(-cross_val_score(model,ny_train_x,ny_train_y,scoring='neg_mean_squared_error',cv=5))

    return rmse

alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]



cv_ridge=[rmse_cv(Ridge(alpha=alpha)).mean()

          for alpha in alphas]

cv_ridge = pd.Series(cv_ridge, index = alphas)

cv_ridge.plot(title = "Validation - Just Do It")

plt.xlabel("alpha")

plt.ylabel("rmse")