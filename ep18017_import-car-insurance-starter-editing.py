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
train_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/train.csv', index_col=0)

test_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/test.csv', index_col=0)
train_df
test_df
train_df.dtypes
test_df.dtypes
train_df['fuel-type'] = train_df['fuel-type'].map({'diesel':0,'gas':1})
train_df['aspiration'] = train_df['aspiration'].map({'std':0,'turbo':1})
train_df['num-of-doors'] = train_df['num-of-doors'].map({'four':0,'two':1})
train_df['body-style'] = train_df['body-style'].map({'hardtop':0,'wagon':1,'sedan':2,'hatchback':3,'convertible':4})
train_df['drive-wheels'] = train_df['drive-wheels'].map({'4wd':0,'fwd':1,'rwd':2})
train_df['engine-location'] = train_df['engine-location'].map({'front':0,'rear':1})
train_df['engine-type'] = train_df['engine-type'].map({'dohc':0,'dohcv':1,'l':2,'ohc':3,'ohcf':4,'ohcv':5,'rotor':6})
train_df['num-of-cylinders'] = train_df['num-of-cylinders'].map({'eight':0,'five':1,'four':2,'six':3,'three':4,'twelve':5,'two':6})
train_df['fuel-system'] = train_df['fuel-system'].map({'1bbl':0,'2bbl':1,'4bbl':2,'idi':3,'mfi':4,'mpfi':5,'spdi':6,'spfi':7})
test_df['fuel-type'] = test_df['fuel-type'].map({'diesel':0,'gas':1})
test_df['aspiration'] = test_df['aspiration'].map({'std':0,'turbo':1})
test_df['num-of-doors'] = test_df['num-of-doors'].map({'four':0,'two':1})
test_df['body-style'] = test_df['body-style'].map({'hardtop':0,'wagon':1,'sedan':2,'hatchback':3,'convertible':4})
test_df['drive-wheels'] = test_df['drive-wheels'].map({'4wd':0,'fwd':1,'rwd':2})
test_df['engine-location'] = test_df['engine-location'].map({'front':0,'rear':1})
test_df['engine-type'] = test_df['engine-type'].map({'dohc':0,'dohcv':1,'l':2,'ohc':3,'ohcf':4,'ohcv':5,'rotor':6})
test_df['num-of-cylinders'] = test_df['num-of-cylinders'].map({'eight':0,'five':1,'four':2,'six':3,'three':4,'twelve':5,'two':6})
test_df['fuel-system'] = test_df['fuel-system'].map({'1bbl':0,'2bbl':1,'4bbl':2,'idi':3,'mfi':4,'mpfi':5,'spdi':6,'spfi':7})
train_df
test_df
train_df.dtypes
test_df.dtypes
train_df = train_df.replace('?',np.NaN)
train_df
test_df = test_df.replace('?',np.NaN)
test_df
columns = train_df.columns

for c in columns:

    train_df[c] = pd.to_numeric(train_df[c], errors='ignore')

train_df
columns = test_df.columns

for c in columns:

    test_df[c] = pd.to_numeric(test_df[c], errors='ignore')

train_df
train_df = train_df.fillna(train_df.mean())



train_df
test_df = test_df.fillna(test_df.mean())



test_df
train_df.dtypes
test_df.dtypes
numeric_columns = ['price', 'wheel-base', 'length', 'width', 'body-style', 'city-mpg', 'city-mpg', 'highway-mpg']



X_train = train_df[numeric_columns].to_numpy()

y_train = train_df['symboling'].to_numpy()

X_test = test_df[numeric_columns].to_numpy()
print(X_train)

print(y_train)

print(X_test)
from sklearn.linear_model import LinearRegression



model = LinearRegression()

model.fit(X_train, y_train)
p_test = model.predict(X_test)
submit_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/sampleSubmission.csv', index_col=0)

submit_df['symboling'] = p_test

submit_df
submit_df.to_csv('submission.csv')