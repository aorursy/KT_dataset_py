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
df = pd.read_csv('../input/laptop-prices/laptops.csv',encoding='latin-1')
df.corr()
df.info()
df.drop(columns=['Unnamed: 0','Company','Inches','ScreenResolution','TypeName','Weight'],inplace=True)
df.head()

# these feature are unique and does nothing in our predictions, So, we drop it !!
# = = = = =  == = = =  = = =  = = = =  = = = = = = = = = =  == = = = =  = = =  = =  = = = = = = = =  = = = = == =  = ==  

df.reset_index(drop=True,inplace=True) # it is important do it anyways before encoder, then you will get no error


# = = = = = = = = = = = == = = = = = =  == = = = = =  == = =  = = = = = = = = = == = = = = = = = = = = = = = = = = = ==

# HELLO ! I AM HIGHLIGHTED !! LOL

# This will reset your index, so that, if there is any situation where you index got skipped,
# Then, it will again make it a continous sequence of no.s
df.info()
col = [feature for feature in df.columns if df[feature].dtype == 'O']

# Using List comprehension for extracting categorical column names 
for feature in col:
    labels_ordered= df.groupby([feature])['Price_euros'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    df[feature]=df[feature].map(labels_ordered)
df.head()
y = df['Price_euros']
X = df.drop(columns='Price_euros')
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler() # setting with_mean False, is for a reason !

temp = scaler.fit_transform(X)
X = pd.DataFrame(temp)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42,shuffle = True)
from sklearn.linear_model import LinearRegression
lr = LinearRegression() 
lr.fit(X_train,y_train)
lr.score(X_train,y_train) # train-set score
lr.score(X_test,y_test) # test-set score
y_pred_lr = lr.predict(X_test) # prediction 
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(random_state=1)
rfr.fit(X_train,y_train)
rfr.score(X_train,y_train) # train-set score 
rfr.score(X_test,y_test) # test-set score
y_pred_rfr = rfr.predict(X_test) #prediction
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_pred_lr,y_test)
mean = y_test.mean()
percentage_mae = mae/mean*100
percentage_mae

# we are having a mean_absolute_error of 13.36%
mae = mean_absolute_error(y_pred_rfr,y_test)
mean = y_test.mean()
percentage_mae = mae/mean*100
percentage_mae

# we are having a mean_absolute_error of 13.36%
