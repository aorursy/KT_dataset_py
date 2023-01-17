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
train_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/train.csv")

test_data = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/test.csv")
train_data.head()
train_data.isnull().sum()

object

object_type = np.dtype("O")

train_data = train_data.apply(pd.to_numeric,errors = 'coerce')

train_data.dropna()



test_data = train_data.apply(lambda x: x.fillna("NA") if x.dtype == object_type

                                       else x.fillna(-1))
from catboost import CatBoostRegressor

from sklearn.model_selection import train_test_split
X = train_data.drop(["Id", "Fatalities"], axis=1)

y= train_data.Fatalities

X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7)

X.dtypes
catagorical_features = np.where(X.dtypes != np.float)[0]

model = CatBoostRegressor(iterations = 1000 ,learning_rate=0.1,depth = 3)

model.fit(X_train,y_train, cat_features=catagorical_features)
preds = model.predict(test_data.drop("Id",axis=1))
submission = pd.read_csv("/kaggle/input/covid19-global-forecasting-week-3/submission.csv")
preds.shape

submission.head
sub= pd.DataFrame({

    "ForecastId" : test_data.Id,

    "ConfirmedCases" : test_data.ConfirmedCases,

    "Fatalities" : preds

})

sub.head()
sub.to_csv("submission.csv",index=False)