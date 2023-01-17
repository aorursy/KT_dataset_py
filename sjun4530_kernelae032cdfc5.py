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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



train = pd.read_csv('../input/train.csv', parse_dates=['Dates'])

test = pd.read_csv('../input/test.csv', parse_dates=['Dates'], index_col='Id')



train['Date'] = pd.to_datetime(train['Dates'].dt.date)

train['n_days'] = (train['Date'] - train['Date'].min()).apply(lambda x: x.days)

train['Day'] = train['Dates'].dt.day

train['DayOfWeek'] = train['Dates'].dt.weekday

train['Month'] = train['Dates'].dt.month

train['Year'] = train['Dates'].dt.year

train['Hour'] = train['Dates'].dt.hour

train['Minute'] = train['Dates'].dt.minute

train['Block'] = train['Address'].str.contains('block', case=False).apply(lambda x: 1 if x == True else 0)

train['ST'] = train['Address'].str.contains('ST', case=False).apply(lambda x: 1 if x == True else 0)

train["X_Y"] = train["X"] - train["Y"]

train["XY"] = train["X"] + train["Y"]



test['Date'] = pd.to_datetime(test['Dates'].dt.date)

test['n_days'] = (test['Date'] - test['Date'].min()).apply(lambda x: x.days)

test['Day'] = test['Dates'].dt.day

test['DayOfWeek'] = test['Dates'].dt.weekday

test['Month'] = test['Dates'].dt.month

test['Year'] = test['Dates'].dt.year

test['Hour'] = test['Dates'].dt.hour

test['Minute'] = test['Dates'].dt.minute

test['Block'] = test['Address'].str.contains('block', case=False).apply(lambda x: 1 if x == True else 0)

test['ST'] = test['Address'].str.contains('ST', case=False).apply(lambda x: 1 if x == True else 0)

test["X_Y"] = test["X"] - test["Y"]

test["XY"] = test["X"] + test["Y"]



from sklearn.preprocessing import LabelEncoder

le1 = LabelEncoder()

train['PdDistrict'] = le1.fit_transform(train['PdDistrict'])

test['PdDistrict'] = le1.transform(test['PdDistrict'])



le2 = LabelEncoder()

y= le2.fit_transform(train['Category'])



le3 = LabelEncoder()

le3.fit(list(train['Address']) + list(test['Address']))

train['Address'] = le3.transform(train['Address'])

test['Address'] = le3.transform(test['Address'])



train.drop(['Dates','Date','Descript','Resolution', 'Category'], 1, inplace=True)

test.drop(['Dates','Date',], 1, inplace=True)





from lightgbm import LGBMClassifier

model = LGBMClassifier(objective="multiclass", num_class=39, max_bin = 465, max_delta_step = 0.9,

                      learning_rate=0.4, num_leaves = 42, n_estimators=100,)

model.fit(train, y, categorical_feature=["PdDistrict", "DayOfWeek"])

preds = model.predict_proba(test)

submission = pd.DataFrame(preds, columns=le2.inverse_transform(np.linspace(0, 38, 39, dtype='int16')), index=test.index)

submission.to_csv('LGBM_final.csv', index_label='Id')