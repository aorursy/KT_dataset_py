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


import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import warnings

warnings.filterwarnings('ignore')
df = pd.read_csv('/kaggle/input/bits-f464-l1/train.csv')

df.fillna(value=df.mean(),inplace=True)

df



from sklearn.model_selection import train_test_split

from sklearn.preprocessing import LabelEncoder





df = df.sample(frac=1).reset_index(drop=True)

#print(df)

#df = df.head(25000)

features = list(df.columns)

features.remove('id')

features.remove('label')

y = list(df['label'])

x = df[features]

#x = np.log(1+x)

x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.30,random_state=42)

from sklearn.preprocessing import RobustScaler



#scaler = RobustScaler()

#x_train[features] = scaler.fit_transform(x_train[features])

#x_val[features] = scaler.transform(x_val[features]) 

from sklearn import tree

from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import validation_curve

from sklearn.pipeline import make_pipeline

from sklearn.metrics import make_scorer

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import mean_squared_error

def root_mean_squared_error(y_true, y_pred):

    ''' Root mean squared error regression loss

    

    Parameters

    ----------

    y_true : array-like of shape = (n_samples) or (n_samples, n_outputs)

    Ground truth (correct) target values.



    y_pred : array-like of shape = (n_samples) or (n_samples, n_outputs)

    Estimated target values.

    '''

    return np.sqrt(mean_squared_error(y_true, y_pred))







from sklearn.ensemble import RandomForestRegressor

rregressor = RandomForestRegressor(n_estimators=400,random_state=0)

rregressor.fit(x_train,y_train)

rpred = rregressor.predict(x_val)



rpred = rregressor.predict(x_val)

print(root_mean_squared_error(y_val,rpred))
print(rregressor)

from sklearn.metrics import r2_score

print(r2_score(y_val,rpred))
testdf = pd.read_csv('/kaggle/input/bits-f464-l1/test.csv')

testdf.fillna(value=testdf.mean(),inplace=True)



x_test = testdf[features]

pred = rregressor.predict(x_test)

pred

testdf['label']=pred

testdf
testdf.drop(df.columns.difference(['id','label']), 1, inplace=True)

testdf
testdf.to_csv('out.csv', index=False) 