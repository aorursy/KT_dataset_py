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
from fastai.tabular import *

import math 
!pip install wang-ds-toolbox

from wang_ds_toolbox import *
train = pd.read_csv('/kaggle/input/big-vehicle-peasants/train.csv', parse_dates=['saledate'])
train.shape
train.columns
train.describe()
train.SalePrice.hist()
train.SalePrice.apply(lambda x: np.log(x)).hist()
train.corr()
def datatype(df):

    date_cols = df.select_dtypes(include=['datetime']).columns.tolist()

    num_cols = df.select_dtypes(include=['number','bool']).columns.tolist()

    cat_cols = df.select_dtypes(include=['object']).columns.tolist() # may contain other type of data

    print(f"Date columns: {date_cols} \n\nNumerical columns:{num_cols} \n\nString columns: {cat_cols}")

    return date_cols, num_cols, cat_cols

    

date_cols, num_cols, cat_cols = datatype(train)

cat_to_num = Categorify(cat_cols, num_cols)

cat_to_num(train)
train.dtypes
train.UsageBand.cat.categories
train.UsageBand.cat.codes.unique()
train.UsageBand.cat.set_categories(['High', 'Medium', 'Low'], ordered=True, inplace=True)
print(train.UsageBand.cat.categories)

print(train.UsageBand.cat.codes.unique())

train
def rmse(x,y): 

    return math.sqrt(((x-y)**2).mean())



def score_output(model):

    print(['train_rmse:',rmse(model.predict(X_train), y_train), 'validation_rmse:',rmse(model.predict(X_valid), y_valid)])

    

X, y, nas = proc_df(train, 'SalePrice')
import numpy as np

from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)
m = RandomForestRegressor(n_jobs=8,n_estimators=10)

%time m.fit(X_train, y_train)

score_output(m)
m.score(X_valid,y_valid)
from sklearn.ensemble import RandomForestRegressor



m = RandomForestRegressor(n_jobs=8)

%time m.fit(X_train, y_train)
a=np.c_[X_train.columns, m.feature_importances_]

pd.DataFrame(a).sort_values(ascending=False,by=1)
test = pd.read_csv('/kaggle/input/big-vehicle-peasants/test.csv',parse_dates=['saledate'])
X, nas = proc_df(test)
test_output = m.predict(X)
SalesID=X.SalesID.values

pd.DataFrame({"SalesID":SalesID,'SalePrice':test_output}).to_csv('output2.csv',index=False)