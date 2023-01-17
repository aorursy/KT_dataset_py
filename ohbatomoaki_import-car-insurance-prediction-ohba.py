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
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

sns.countplot(x='symboling', data=train_df)

plt.show()
import matplotlib.pyplot as plt

import seaborn as sns

# Class count

count_class_0, count_class_1, count_class_2, count_class_3, count_class_4, count_class_5 = train_df.symboling.value_counts()

 

# Divide by class

df_class_0 = train_df[train_df['symboling'] == 0]

df_class_1 = train_df[train_df['symboling'] == 1]

df_class_2 = train_df[train_df['symboling'] == 2]

df_class_3 = train_df[train_df['symboling'] == 3]

df_class_4 = train_df[train_df['symboling'] == -1]

df_class_5 = train_df[train_df['symboling'] == -2]



df_class_1_over = df_class_1.sample(count_class_0, replace=True)

df_class_2_over = df_class_2.sample(count_class_0, replace=True)

df_class_3_over = df_class_3.sample(count_class_0, replace=True)

df_class_4_over = df_class_4.sample(count_class_0, replace=True)

df_class_5_over = df_class_5.sample(count_class_0, replace=True)



train_df_over = pd.concat([df_class_0, df_class_1_over, df_class_2_over, df_class_3_over, df_class_4_over, df_class_5_over], axis=0)



print(train_df_over.symboling.value_counts())



%matplotlib inline

sns.countplot(x='symboling', data=train_df_over)

plt.show()
import matplotlib.pyplot as plt

import seaborn as sns



corr = train_df_over.corr()



plt.style.use('ggplot')

plt.figure()

sns.heatmap(corr, square=True, annot=True)

plt.show()
train_df_over = train_df_over[['wheel-base', 'length', 'width', 'height', 'curb-weight', 'highway-mpg', 'symboling']]
all_df = pd.concat([train_df_over.drop('symboling', axis=1), test_df])

all_df
all_df = all_df.replace('?', np.NaN)

all_df
columns = all_df.columns

for c in columns:

    all_df[c] = pd.to_numeric(all_df[c], errors='ignore')

all_df
all_df.dtypes
columns = all_df.columns

for c in columns:

    if all_df[c].isna().any():

        if all_df[c].dtypes != np.object:

            median = all_df[c].median()

            all_df[c] = all_df[c].replace(np.NaN, median)

        else:

            mfv = all_df[c].mode()[0]

            all_df[c] = all_df[c].replace(np.NaN, mfv)
columns = all_df.columns

for c in columns:

    if all_df[c].dtypes == np.object:

        all_df = pd.concat([all_df, pd.get_dummies(all_df[[c]])], axis=1)

        all_df = all_df.drop(c, axis=1)

all_df
X_train_all = all_df[:len(train_df_over)].to_numpy()

y_train_all = train_df_over['symboling'].to_numpy()



X_test_all = all_df[len(train_df_over):].to_numpy()
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error



X_train, X_valid, y_train, y_valid = train_test_split(X_train_all, y_train_all, test_size=0.2, random_state=0)  # 訓練用と検証用に分ける

model = RandomForestRegressor()

model.fit(X_train, y_train)  # 訓練用で学習

predict = model.predict(X_valid)

RMSE = np.sqrt(mean_squared_error(y_valid, predict))

RMSE
from sklearn.model_selection import GridSearchCV



params = {'max_depth':[12, 13, 14],'n_estimators':[70, 80, 90], 'random_state':[0]}

gscv = GridSearchCV(model, params, cv=5, scoring='neg_root_mean_squared_error') 

gscv.fit(X_train, y_train)
gscv.best_score_, gscv.best_params_
model = RandomForestRegressor(max_depth=13, n_estimators=80, random_state=0)

model.fit(X_train_all, y_train_all)
p_test = model.predict(X_test_all)
submit_df = pd.read_csv('/kaggle/input/1056lab-import-car-insurance-prediction/sampleSubmission.csv', index_col=0)

submit_df['symboling'] = p_test

submit_df
submit_df.to_csv('submission.csv')