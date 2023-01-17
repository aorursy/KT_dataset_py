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
url = ('../input/heart-disease/heart.csv')
df = pd.read_csv(url)
df
df.head()
df.columns
df.isnull().values.any()
df.isnull().values.sum()
y = df.target
df.describe()
features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',

       'exang', 'oldpeak', 'slope', 'ca', 'thal']
X = df[features]
X.head()
X.describe()
from sklearn.model_selection import train_test_split



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0)

from xgboost import XGBRegressor



final_model = XGBRegressor()

final_model.fit(train_X, train_y)
from sklearn.metrics import mean_absolute_error



predictions = final_model.predict(val_X)

print("Mean Absolute Error: " + str(mean_absolute_error(predictions, val_y)))
predict_final = final_model.predict(val_X)
print(predict_final)
my_submission = pd.DataFrame({'Id': val_X.index, 'target': predict_final})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)