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
import numpy as np

import pandas as pd

train_data=pd.read_csv("/kaggle/input/into-the-future/train.csv")

test_data=pd.read_csv("/kaggle/input/into-the-future/test.csv")
train_data.head()
test_data.head()
train_data.describe()
from sklearn.linear_model import LinearRegression

f1=train_data['feature_1']

f1_df=pd.DataFrame(f1)

f1_df.head()
f2=train_data['feature_2']

f2_df=pd.DataFrame(f2)
f2_df.head()
LR=LinearRegression()

LR.fit(f1_df,f2_df)
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(f1_df,f2_df,test_size=0.2,random_state=0)
y_test_a=LR.predict(X_test)

y_train_a=LR.predict(X_train)
from sklearn.metrics import r2_score
r2_score(y_test,y_test_a)
r2_score(y_train,y_train_a)
x = test_data['feature_1']

x = x.values.reshape(-1,1)

y = LR.predict(x)





test_data['feature_2']=y
test_predicted=pd.DataFrame()

test_predicted['time']=test_data['time']

test_predicted['feature_1']=test_data['feature_1']

test_predicted['feature_2']=test_data['feature_2']



test_predicted.head()
test_predicted.to_csv("result.csv")