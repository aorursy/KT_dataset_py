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
#importing Pandas
import pandas as pd

#importing dataframe
df = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv',sep=',',engine='python',parse_dates=['date'])
df.head()
df.info()
#Data Merging/Joining

df1=pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv',sep=',')

df2=pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv',sep=',')

df3 = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv',sep=',')

merge_1 = pd.merge(df,df1,on='item_id')

merge_2 = pd.merge(merge_1,df2,on='shop_id')

merge_3 = pd.merge(merge_2,df3,on='item_category_id')

merge_3.tail()
merge_3.shape
#Grouping the Data Monthly

cat = ['date_block_num','shop_id','item_id']
test = merge_3.groupby(cat)[['item_cnt_day','item_price']].sum()
test
#Saving the file
test.to_csv('./D:\Kaggle\Future Sales\group_data.csv',header=True)
#Reading the Training Data

data = pd.read_csv('./D:\Kaggle\Future Sales\group_data.csv',sep=',')
data.head()
data.describe()
#import numpy and scikit learn models
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

#Spliting data into training and target sets
y = np.array(data['item_cnt_day'])
X = np.array(data.drop({'item_cnt_day'},axis=1))
np.reshape(X,(1609124,4))
np.reshape(y,(1609124,1))


#Spliting into train and test set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=26)


#Loading Model and Random Search CV
rd = Ridge(normalize=False)
a_space = np.logspace(-4,10,50)
param = {'alpha': a_space}
model = RandomizedSearchCV(rd,param,cv=8)
rd2=Ridge(alpha=0.4,normalize=False)
rd2.fit(X_train,y_train).coef_
model.fit(X_train,y_train)
y_pred = model.predict(X_test)
temp=y_pred-y_test
temp
from sklearn.metrics import mean_squared_error
error = mean_squared_error(y_pred,y_test,squared=False)
print(error)
from sklearn.linear_model import Lasso
lr = Lasso()
lasso_model = RandomizedSearchCV(lr,param,cv=8)
lasso_model.fit(X_train,y_train)
y_pred2=lasso_model.predict(X_test)
error = mean_squared_error(y_pred2,y_test,squared=False)
print(error)
from sklearn.ensemble import AdaBoostRegressor
ada = AdaBoostRegressor(learning_rate=0.2)
ada.fit(X_train,y_train)
y_pred3 = ada.predict(X_test)
error = mean_squared_error(y_pred3,y_test,squared=False)
print(error)
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor()
gbr.fit(X_train,y_train)
y_pred4 = gbr.predict(X_test)
error = mean_squared_error(y_pred4,y_test,squared=False)
print(error)