# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
from sklearn import ensemble
pd.options.display.max_columns = None
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

df_macro = pd.read_csv('../input/macro.csv')
df_train.head()
df_test.head()
df_macro.head()
train_xy = df_train[['full_sq', 'life_sq', 'floor', 'max_floor', 'price_doc']].fillna(0)

train_x = train_xy[['full_sq', 'life_sq', 'floor', 'max_floor']]

train_y = train_xy['price_doc']
test_xy = df_test[['full_sq', 'life_sq', 'floor', 'max_floor']].fillna(0.0)

test_x = test_xy[['full_sq', 'life_sq', 'floor', 'max_floor']]
model = ensemble.GradientBoostingRegressor()

fit = model.fit(X=train_x[['full_sq', 'life_sq', 'floor', 'max_floor']], y=train_y)
fit.feature_importances_
test_prediction = fit.predict(test_x)
test_prediction[:10]
df_output = df_test[['id']].copy()

df_output['price_doc'] =  test_prediction
df_test.head()
df_output.to_csv('test_subm.csv')