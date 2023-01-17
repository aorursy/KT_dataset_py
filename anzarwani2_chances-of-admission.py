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
path = '../input/graduate-admissions/Admission_Predict.csv'
df = pd.read_csv(path, index_col='Serial No.')

df.head()
df.isnull().sum()
df.columns
features = ['GRE Score', 'TOEFL Score', 'University Rating', 'SOP', 'LOR ', 'CGPA',

       'Research']
X = df[features]
X.describe()
y = df['Chance of Admit ']
y.describe()
from sklearn.model_selection import train_test_split



X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state = 0)

from xgboost import XGBRegressor



my_model = XGBRegressor()

my_model.fit(X_train, y_train)
from sklearn.metrics import mean_absolute_error



predictions = my_model.predict(X_valid)

print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))
print(predictions)
my_submission = pd.DataFrame({'Id': X_valid.index, 'chance': predictions})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)