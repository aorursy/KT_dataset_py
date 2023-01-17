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
df_train= pd.read_csv("/kaggle/input/price-data/Price/Train.csv")
df_test= pd.read_csv("/kaggle/input/price-data/Price/Test.csv")
y_train=np.array(df_train['Low_Cap_Price'])
id_test=np.array(df_test['Item_Id'])
x_train=np.array(df_train.drop(['Item_Id', 'Date', 'Low_Cap_Price'], axis=1, index=None))
x_test=np.array(df_test.drop(['Item_Id', 'Date'], axis=1, index=None))
from sklearn.ensemble import RandomForestRegressor 

rforest = RandomForestRegressor(n_estimators = 5000, random_state = 1) 
rforest.fit(x_train,y_train)
y1_test= rforest.predict(x_test)
print(id_test, y1_test)
df3 = pd.DataFrame()
df3['Item_Id'] = id_test.tolist()
df3['Low_Cap_Price'] = y1_test.tolist()
df3.to_csv("./file1.csv", sep=',',index=True)