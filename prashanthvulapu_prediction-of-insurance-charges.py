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
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
df = pd.read_csv("../input/insurance/insurance.csv")
df.columns
df.head()
df.isna().sum()
df.nunique()
df['region'].value_counts()
dummy_cat = pd.get_dummies(df,columns = ['sex','smoker','region'])
dummy_cat.head()
dummy_cat = dummy_cat.drop(['sex_female','smoker_no','region_northeast'],axis=1)
dummy_cat.columns
y = dummy_cat.loc[:,'charges'].values
y
X = dummy_cat.loc[:,['age', 'bmi', 'children','sex_male', 'smoker_yes',
       'region_northwest', 'region_southeast', 'region_southwest']].values
X
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)
len(X_train)
len(X_test)
len(y_train)
len(y_test)
lr = LinearRegression()
lr.fit(X_train,y_train)
y_predict = lr.predict(X_test)
lr.coef_
dummy_cat.columns
lr.intercept_
r2_score(y_test,y_predict)