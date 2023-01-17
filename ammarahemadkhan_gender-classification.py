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
df = pd.read_csv("/kaggle/input/gender-classification/Transformed Data Set - Sheet1.csv")
df
for col in df.columns.values:
    print({col:df[col].unique()})
def dummy_getter(df,col_name,to_con_df):
    dummies = pd.get_dummies(df)
    #dummies = 
    return pd.concat([to_con_df.drop([col_name],axis = 1),dummies.iloc[:,:-1]],axis = 1)
for col in df.columns.values:
    df = dummy_getter(df[col],col,df)
df.head()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(df.drop(["F"],axis = 1),df["F"])
from sklearn.linear_model import LogisticRegression
lgr = LogisticRegression().fit(X_train,y_train)
lgr.score(X_test,y_test)

