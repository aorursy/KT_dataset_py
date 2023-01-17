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
numeric_columns = ['wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 'compression-ratio', 'city-mpg', 'highway-mpg']



X_train = train_df[numeric_columns].to_numpy()

y_train = train_df['symboling'].to_numpy()

X_test = test_df[numeric_columns].to_numpy()
from sklearn.linear_model import LinearRegression



model = LinearRegression()

model.fit(X_train, y_train)
p_test = model.predict(X_test)
submit_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/sampleSubmission.csv', index_col=0)

submit_df['symboling'] = p_test

submit_df
submit_df.to_csv('submission.csv')