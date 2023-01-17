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
path = '../input/automobile-dataset/Automobile_data.csv'
df = pd.read_csv(path)
df
df.info()
s = (df.dtypes == 'object')
object_cols = list(s[s].index)
print(object_cols)
df['normalized-losses']=df['normalized-losses'].str.replace('?','140')
df['bore']=df['bore'].str.replace('?','3.50')
df['stroke']=df['stroke'].str.replace('?','3.20')
df['horsepower']=df['horsepower'].str.replace('?','140')
df['peak-rpm']=df['peak-rpm'].str.replace('?','4450')
df['price']=df['price'].str.replace('?','15000')
df['normalized-losses']=df['normalized-losses'].astype('int64')
df['bore']=df['bore'].astype('float')
df['stroke']=df['stroke'].astype('float')
df['horsepower']=df['horsepower'].astype('int64')
df['peak-rpm']=df['peak-rpm'].astype('int64')
df['price']=df['price'].astype('int64')
from catboost import CatBoostRegressor
df
y=df['price']
X=df.drop(['price'],axis=1)
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.2)
categorical_features_indices = np.where(X.dtypes != np.float)[0]

model=CatBoostRegressor(iterations=1000, depth=3, learning_rate=0.6, loss_function='RMSE', use_best_model=True, eval_metric='RMSE')
model.fit(X_train, y_train,cat_features=categorical_features_indices,eval_set=(X_test, y_test),plot=True)
pred_price = model.predict(X_test)
print(pred_price)
from sklearn.metrics import r2_score

score=r2_score(y_test,pred_price)
score
import seaborn as sns
sns.regplot(x = y_test, y = pred_price)
