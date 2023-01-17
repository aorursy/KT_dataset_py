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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')



df
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')



def transform(db1):

    numeric_cols = [cname for cname in db1.columns if db1[cname].dtype in ['int64', 'float64']]

    db = db1[numeric_cols[:-1]].copy()

    return db 

    



from sklearn.model_selection import train_test_split

import xgboost as xgb

from sklearn.metrics import mean_squared_error

import statistics

 



X, y = transform(df[df.columns[:-1]]), df['SalePrice']



N, results, predictions = 30, [], []



for i in range(N):

   

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

    model = xgb.XGBRegressor(n_estimators=1000,max_depth=3,reg_lambda=2.0)



    model.fit(X_train, y_train)

    y_prediction = model.predict(X_val)



    results.append(mean_squared_error(y_val,y_prediction))

    predictions.append(model.predict(transform(test)))

    

print(sum(results)/N, statistics.stdev(results))



rating = []

for i in range(len(test)):

     rating.append(sum([pred[i] for pred in predictions])) 



rating = [1 if rate>16 else 0 for rate in rating]



y_prediction = rating        

submission['SalePrice'] = y_prediction

submission.to_csv('my_results.csv',index=False)
np.sqrt(995702321)