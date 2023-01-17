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
train = pd.read_csv('/kaggle/input/Train.csv')
test = pd.read_csv('/kaggle/input/Test.csv')

train.head()
train.describe(include="O")
test.describe(include="O")
train.describe()
train.corr(method ='pearson')
train.describe(include='O')
train.Product_Brand.unique()
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
plt.figure(figsize=(20,3))
sns.countplot(x='Item_Category', data=train)
plt.xticks(plt.xticks()[0], train.Item_Category, rotation=90)
plt.show()
train.groupby('Item_Category')[['Item_Category','Subcategory_1','Subcategory_2']].count()

test
test.drop(columns=['Item_Rating','Date'], inplace=True)
test
test
df = train
df

df.drop(columns=['Selling_Price','Item_Rating','Date'],inplace=True)

data = pd.concat([df, test], ignore_index=True)
data.iloc[0:2452]
from sklearn.preprocessing import LabelEncoder

dum_df = pd.get_dummies(data["Item_Category"], prefix='Type_is_' )
dum_df
data = data.join(dum_df)

data.drop(columns=['Item_Category'], inplace=True)
data
dum_df = pd.get_dummies(data["Subcategory_1"], prefix='sub_1_Type_is_' )
dum_df

data = data.join(dum_df)
data.drop(columns=['Subcategory_1'], inplace=True)
data
dum_df = pd.get_dummies(data["Subcategory_2"], prefix='sub_2_Type_is_' )
dum_df

data = data.join(dum_df)
data.drop(columns=['Subcategory_2'], inplace=True)
data
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
data = data
data['Product_Brand'] = le.fit_transform(data['Product_Brand'])
data.tail(20)
data = data.drop(columns=['Product'])
data
x_data = data.iloc[0:2452]
x_data
test_y = data.iloc[2452:]
test_y
train

train = pd.read_csv('/kaggle/input/Train.csv')
y_data = train[['Selling_Price']]
y_data


from math import sqrt 
from sklearn.metrics import mean_squared_error, mean_squared_log_error
x_data.shape, y_data.shape

from xgboost import XGBRegressor
from sklearn.model_selection import KFold

errxgb = []
y_pred_totxgb = []

fold = KFold(n_splits=15, shuffle=True, random_state=42)

for train_index, test_index in fold.split(x_data):
#     X_train, X_test = .loc[train_index], x_data.loc[test_index]
#     y_train, y_test = y_data[train_index], [test_index]

    X_train, X_test = x_data.iloc[train_index], x_data.iloc[test_index]
    y_train, y_test = y_data.iloc[train_index], y_data.iloc[test_index]
    
    xgb = XGBRegressor(random_state=42)
    xgb.fit(X_train, y_train)

    y_pred_xgb = xgb.predict(X_test)
    y_pred_xgb = np.abs(y_pred_xgb)
  
    score = np.sqrt(mean_squared_log_error(y_test, y_pred_xgb))
    print(score)
#     print("RMSLE: ", sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred_xgb))))

#     errxgb.append(sqrt(mean_squared_log_error(np.exp(y_test), np.exp(y_pred_xgb))))
    p = xgb.predict(test_y)
    y_pred_totxgb.append(p)

final = (np.mean(y_pred_totxgb,0))
# for i in final:
#     print(i)
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x_data, y_data, test_size=0.05, random_state=43)


model = XGBRegressor(
                    
)
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.linear_model import Ridge
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# model = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=1,algorithm = "brute",p=1))

model.fit(x_data, y_data)
preds = model.predict(X_test)
preds = np.abs(preds)
from sklearn.metrics import mean_squared_log_error
score = np.sqrt(mean_squared_log_error(Y_test, preds))
print(score)
# from sklearn.metrics import r2_score
# print(r2_score(Y_test, preds))
test_predict = model.predict(test_y)
# xgb1 = XGBRegressor()
# parameters = {
#               'learning_rate': [.07,0.9,1.0,1.5,1.9,2.0,2.5,2.9,3.0,3.3,3.5], #so called `eta` value
#               'max_depth': [5,7,8,9],
#               'min_child_weight': [1,2,3,4,5],
#               }

# xgb_grid = GridSearchCV(xgb1,
#                         parameters,
#                         cv = 3,
#                         n_jobs = 5,
#                         verbose=True)

# xgb_grid.fit(X_train, Y_train)

# print(xgb_grid.best_score_)
# print(xgb_grid.best_params_)
# for i in test_predict:
#     print (i)
import lightgbm as lgb
new_model = lgb.LGBMRegressor()
new_model.fit(X_train, Y_train)
preds = new_model.predict(X_test)
import numpy as np
preds = np.abs(preds)
from sklearn.metrics import mean_squared_log_error
score = np.sqrt(mean_squared_log_error(Y_test, preds))
print(score)
new_model.score(X_test, Y_test)
import catboost as cb
cat_model = cb.CatBoostRegressor()
cat_model.fit(X_train, Y_train)
preds = cat_model.predict(X_test)
preds = np.abs(preds)
from sklearn.metrics import mean_squared_log_error
score = np.sqrt(mean_squared_log_error(Y_test, preds))
print(score)
cat_model.score(X_test, Y_test)
test_predict = cat_model.predict(test_y)
# for i in test_predict:
#     print (i)
from sklearn import linear_model
reg = linear_model.Lasso(alpha=0.1)

reg.fit(X_train, Y_train)
preds = cat_model.predict(X_test)
preds = np.abs(preds)
from sklearn import linear_model
reg = linear_model.Lasso(alpha=0.1)



reg.fit(X_train, Y_train)
preds = reg.predict(X_test)
preds = np.abs(preds)
from sklearn.metrics import mean_squared_log_error
score = np.sqrt(mean_squared_log_error(Y_test, preds))
print(score)
test_predict = reg.predict(test_y)
# for i in test_predict:
#     print (i)