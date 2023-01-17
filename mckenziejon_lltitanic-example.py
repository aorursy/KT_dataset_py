# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from fastai.imports import *
from fastai.structured import *
from sklearn.ensemble import RandomForestClassifier
from pandas.api.types import is_string_dtype, is_numeric_dtype

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# import os
# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
train_cats(df)
test_df = pd.read_csv('../input/test.csv')
train_cats(test_df)
test_df, y, nas = proc_df(test_df)
df, y, nas = proc_df(df, 'Survived')
rfc = RandomForestClassifier(n_jobs=-1)
rfc.fit(df, y)
rfc.score(df, y)
del test_df['Fare_na'] # To make the "shape" of test_df match the shape of df
predictions = zip(test_df['PassengerId'], rfc.predict(test_df))
output_df = pd.DataFrame(list(predictions), columns=['PassengerId', 'Survived'])
output_df.to_csv('csv_to_submit.csv', index = False)