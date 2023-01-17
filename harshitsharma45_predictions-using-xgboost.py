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
import pandas as pd
filepath='/kaggle/input/kickstarter-projects/ks-projects-201801.csv'
data=pd.read_csv(filepath,parse_dates=['deadline','launched'])
data
data.groupby(["state"]).state.count()
data=data.query('state!="live"')
data=data.assign(outcome=(data.state=="successful").astype(int))
data.head()
data=data.assign(hour=data.launched.dt.hour,
                day=data.launched.dt.day,
                month=data.launched.dt.month,
                year=data.launched.dt.year)
data
from sklearn.preprocessing import LabelEncoder
cat_features=['category','currency','country']
encoder=LabelEncoder()
encoded=data[cat_features].apply(encoder.fit_transform)
encoded
data=data[['hour','day','month','year','outcome']].join(encoded)
data
valid_fraction=0.1
valid_size=int(len(data)*valid_fraction)

train=data[:-2*valid_size]
valid=data[-2*valid_size:-valid_size]
test=data[-valid_size:]

from sklearn.model_selection import train_test_split
from sklearn.ensemble   import RandomForestClassifier
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
import numpy as np

X_train=train.drop(columns=['outcome'])

y_train=train.outcome

X_test=test.drop(columns=['outcome'])

y_test=test.outcome
#y=train.outcome



model=XGBRegressor(n_estimators=800,n_jobs=4)
model.fit(X_train,y_train)
predictions=model.predict(X_test)
mae=mean_absolute_error(y_test,predictions)
print(mae)