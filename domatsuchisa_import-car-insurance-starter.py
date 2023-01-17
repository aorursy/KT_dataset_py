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
train_df = train_df.replace('?', np.NaN)

train_df
test_df = test_df.replace('?', np.NaN)

test_df
train_df.dtypes
test_df.dtypes
train_df['normalized-losses'] = train_df.astype({'normalized-losses': 'float64'})

train_df['bore'] = train_df.astype({'bore': 'float64'})

train_df['stroke'] = train_df.astype({'stroke': 'float64'})

train_df['horsepower'] = train_df.astype({'horsepower': 'float64'})

train_df['price'] = train_df.astype({'price': 'float64'})
train_df = train_df.fillna(train_df.median())
train_df.dtypes
test_df['normalized-losses'] = test_df.astype({'normalized-losses': 'float64'})

test_df['bore'] = test_df.astype({'bore': 'float64'})

test_df['stroke'] = test_df.astype({'stroke': 'float64'})

test_df['horsepower'] = test_df.astype({'horsepower': 'float64'})

test_df['price'] = test_df.astype({'price': 'float64'})
test_df = test_df.fillna(test_df.median())
test_df.dtypes
numeric_columns = ['normalized-losses', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 

                   'bore', 'stroke', 'compression-ratio', 'horsepower','city-mpg', 'highway-mpg', 'price']



X_train_valid = train_df[numeric_columns].to_numpy()

y_train_valid = train_df['symboling'].to_numpy()

X_test = test_df[numeric_columns].to_numpy()
%%time

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
#トレーニングデータ、テストデータの分離

X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.3, random_state=1)



#条件設定

param = {'n_estimators':[1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}





#ランダムフォレストの実行

clf = GridSearchCV(estimator=RandomForestClassifier(random_state=0), param_grid=param, \

                        scoring='r2', cv=3)

clf.fit(X_train, y_train)



print('n_estimators   :  %d'  %clf.best_estimator_.n_estimators)
p_test_c = clf.predict(X_test)
submit_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/sampleSubmission.csv', index_col=0)

submit_df['symboling'] = p_test_c

submit_df
submit_df.to_csv('car_submission.csv')