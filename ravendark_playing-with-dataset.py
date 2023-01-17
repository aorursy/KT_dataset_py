# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import sklearn
data = pd.read_csv('../input/books.csv',error_bad_lines=False)
data.head()
import seaborn as sns
data.language_code.replace(data['language_code']<7,1)
sns.kdeplot(data.language_code.value_counts())
import category_encoders as ce

data.info()
data.isbn13.value_counts().count()
data = data.drop(['isbn','isbn13','bookID'],axis = 1)
data.authors.value_counts().count()
encoder = ce.BinaryEncoder(cols = ['authors','title','language_code'])
df_binary = encoder.fit_transform(data)
df_binary.head()
df_binary.info()
y = df_binary.average_rating
df_binary = df_binary.drop('average_rating',axis  = 1)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(t,y,test_size = 0.25,random_state = 7124)
from sklearn.ensemble import RandomForestRegressor
m  = RandomForestRegressor(n_estimators = 80, n_jobs = -1,max_features = 0.2)
m.fit(x_train,y_train)
m.score(x_train,y_train)
m.score(x_test,y_test)
from sklearn.metrics import mean_squared_log_error
y_pred  = m.predict(x_test)
mean_squared_log_error(y_pred,y_test)
y = m.predict(x_train)
mean_squared_log_error(y,y_train)
def rmse(x,y) : return np.sqrt(np.mean((x-y)**2))
rmse(y_pred,y_test)
rmse(y,y_train)
y_test[10:20]
y_pred[10:20]
data
from sklearn  import preprocessing
min_max = preprocessing.MinMaxScaler()
x_scaled = min_max.fit_transform(df_binary)
t = pd.DataFrame(x_scaled)
t.info()
df_binary.info()
len(y)
y = data.average_rating
len(y)
m.fit(x_train,y_train)
y_pred = m.predict(x_test)
rmse(y_pred,y_test)
m.score(x_test,y_test)
m = RandomForestRegressor(n_estimators  = 20,n_jobs =-1,max_features=0.5)
m.fit(x_train,y_train)
y_pred = m.predict(x_test)
rmse(y_pred,y_test)
data.info()
y  = data.average_rating
data = data.drop('average_rating',axis = 1)
stringg = data.select_dtypes(include = ['object'])
stringg.info()
data = data.drop(['title','authors','language_code'],axis = 1)
new_data = min_max.fit_transform(data)
enc = ce.BackwardDifferenceEncoder(cols = ['title','authors','language_code'])
dfram = enc.fit_transform(stringg)
yo_data = pd.concat([dfram,yeah],axis = 1)
dfram.head()
new_data.head()
new_data
yeah = pd.DataFrame(new_data)
yeah.info()
yeah.head()
yo_data.info()
len(yo_data.head().transpose())
x_train,x_test,y_train,y_test  = train_test_split(yo_data,y,test_size = 0.25,random_state = 23423)
model = RandomForestRegressor(n_estimators = 40,n_jobs  = -1,max_features= 0.5,max_depth=2)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
rmse(y_pred,y_test)
model.score(x_train,y_train)
model.score(x_test,y_test)
y_test[1:10]
y_pred[1:10]