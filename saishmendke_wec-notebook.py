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

from sklearn import svm

from sklearn.metrics import roc_auc_score
import os

print((os.listdir('../input/')))
df_train = pd.read_csv('../input/wecrec2020/Train_data.csv')

df_test = pd.read_csv('../input/wecrec2020/Test_data.csv')
test_index=df_test['Unnamed: 0']
df_train.drop(['F1', 'F2'], axis = 1, inplace = True)

#df_test.drop(['F1', 'F2'], axis = 1, inplace = True)
df_train
train_X = df_train.loc[:, 'F3':'F17']

train_y = df_train.loc[:, 'O/P']

df_test = df_test.loc[:, 'F3':'F17']
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=50, random_state=43)

rf.fit(train_X, train_y)
df_test = df_test.loc[:, 'F3':'F17']

pred = rf.predict(df_test)

pred
df_train = df_train.loc[:, 'F3':'F17']

train_pred = rf.predict(df_train)
from sklearn.metrics import mean_squared_error

mean_squared_error(train_pred,train_y)
rf1 = RandomForestRegressor(n_estimators=500, random_state=43)

rf1.fit(train_X, train_y)

train_pred1 = rf1.predict(df_train)

pred_rf=rf1.predict(df_test)

pred_rf
mean_squared_error(train_pred1,train_y)
from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train_X, train_y, test_size=0.25, random_state=42)
from sklearn.tree import DecisionTreeRegressor

max_depth=[5,10,12,15,18]

val_error=[]

train_error=[]

for i in max_depth:

    model=DecisionTreeRegressor(random_state=0,max_depth=i,min_samples_leaf=2)

    model.fit(X_train,y_train)

    pred_val = model.predict(X_val)

    pred_train = model.predict(X_train)

    error1=mean_squared_error(pred_val,y_val)

    val_error.append(error1)

    error2=mean_squared_error(pred_train,y_train)

    train_error.append(error2)
import matplotlib.pyplot as plt

%matplotlib inline

def make_figure(dim, title, xlabel, ylabel, legend):

    plt.rcParams['figure.figsize'] = dim

    plt.title(title)

    plt.xlabel(xlabel)

    plt.ylabel(ylabel)

    if legend is not None:

        plt.legend(loc=legend, prop={'size':15})

    plt.rcParams.update({'font.size': 16})

    plt.tight_layout()
plt.plot([5,10,12,15,18], val_error, linewidth=5.0, label='Validation error')

plt.plot([5,10,12,15,18], train_error, linewidth=5.0, label='Train error')



make_figure(dim=(10,5), title='Error vs Value of Max depth',

            xlabel='Value of max depth',

            ylabel='Error',

            legend='best')
from sklearn.tree import DecisionTreeRegressor

dc = DecisionTreeRegressor(random_state=0,max_depth=15)

dc.fit(train_X,train_y)
pred2 = dc.predict(df_train)
mean_squared_error(pred2,train_y)
max_depth=[8,10,12,15,18]

rf_val_error=[]

rf_train_error=[]

for i in max_depth:

    model=RandomForestRegressor(n_estimators=150,max_depth=i, random_state=43)

    model.fit(X_train,y_train)

    pred_val = model.predict(X_val)

    pred_train = model.predict(X_train)

    error1=mean_squared_error(pred_val,y_val)

    rf_val_error.append(error1)

    error2=mean_squared_error(pred_train,y_train)

    rf_train_error.append(error2)
plt.plot([8,10,12,15,18], rf_val_error, linewidth=5.00, label='Validation error')

plt.plot([8,10,12,15,18], rf_train_error, linewidth=5.00, label='Train error')



make_figure(dim=(10,5), title='Error vs Value of max depth',

            xlabel='Value of Max depth',

            ylabel='Error',

            legend='best')
rf_train_error
rf_val_error
rf_2 = RandomForestRegressor(n_estimators=50,max_depth=12,min_samples_leaf=3, random_state=43)

rf_2.fit(train_X,train_y)
pred=rf_2.predict(df_test)

pred
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(pred)

result.head()
result.to_csv('output.csv', index=False)
from sklearn.ensemble import GradientBoostingRegressor

max_depth=[3,5,7,9,12]

gbr_val_error=[]

gbr_train_error=[]

for i in max_depth:

    model=GradientBoostingRegressor(random_state=0,max_depth=i,n_estimators=150,min_samples_leaf=2)

    model.fit(X_train,y_train)

    pred_val = model.predict(X_val)

    pred_train = model.predict(X_train)

    error1=mean_squared_error(pred_val,y_val)

    gbr_val_error.append(error1)

    error2=mean_squared_error(pred_train,y_train)

    gbr_train_error.append(error2)
plt.plot([3,5,7,9,12], gbr_val_error, linewidth=5.00, label='Validation error')

plt.plot([3,5,7,9,12], gbr_train_error, linewidth=5.00, label='Train error')



make_figure(dim=(10,5), title='Gradient Boosting Regressor',

            xlabel='Value of max depth',

            ylabel='Error',

            legend='best')
gbr_model=GradientBoostingRegressor(random_state=0,n_estimators=75,max_depth=12,min_samples_leaf=2)

gbr_model.fit(train_X,train_y)

gbr_pred = gbr_model.predict(df_test)
gbr_pred_train = gbr_model.predict(train_X)

mean_squared_error(gbr_pred_train,train_y)
gbr_pred
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(gbr_pred)

result.head()
result.to_csv('gbr_output1.csv', index=False)
gbr_model1 = GradientBoostingRegressor(random_state=0,max_depth=12,n_estimators=50)

gbr_model1.fit(train_X,train_y)

gbr_pred1 = gbr_model1.predict(df_test)
gbr_pred1
gbr_pred_final = (gbr_pred+gbr_pred1)/2.0
gbr_pred_final
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(gbr_pred_final)

result.head()
result.to_csv('gbr_output3.csv', index=False)
df_train
gbr_model1 = GradientBoostingRegressor(random_state=0,max_depth=10,n_estimators=500,learning_rate=0.01)

gbr_model1.fit(train_X,train_y)

gbr_pred1 = gbr_model1.predict(df_test)

gbr_pred_train = gbr_model1.predict(train_X)

mean_squared_error(gbr_pred_train,train_y)
gbr_pred1
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(gbr_pred1)

result.head()
result.to_csv('gbr_output5.csv', index=False)
import xgboost as xgb

max_depth=[3,5,7,9,12]

xgb_val_error=[]

xgb_train_error=[]

for i in max_depth:

    model=xgb.XGBRegressor(random_state=0,max_depth=i,n_estimators=150)

    model.fit(X_train,y_train)

    pred_val = model.predict(X_val)

    pred_train = model.predict(X_train)

    error1=mean_squared_error(pred_val,y_val)

    xgb_val_error.append(error1)

    error2=mean_squared_error(pred_train,y_train)

    xgb_train_error.append(error2)
plt.plot([3,5,7,9,12], xgb_val_error, linewidth=5.00, label='Validation error')

plt.plot([3,5,7,9,12], xgb_train_error, linewidth=5.00, label='Train error')



make_figure(dim=(10,5), title='XGBoost',

            xlabel='Value of max depth',

            ylabel='Error',

            legend='best')
model = xgb.XGBRegressor(random_state=0,max_depth=7,n_estimators=150)

model.fit(train_X,train_y)

pred_train = model.predict(train_X)

pred = model.predict(df_test)

mean_squared_error(train_y,pred_train)
pred
pred_xgb_gbr = (pred+gbr_pred1)/2.0
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(pred_xgb_gbr)

result.head()
result.to_csv('xgb_output10.csv', index=False)
model=xgb.XGBRegressor(random_state=0,max_depth=12,n_estimators=500,learning_rate=0.01)

model.fit(train_X,train_y)

pred_train = model.predict(train_X)

pred_d12=model.predict(df_test)

mean_squared_error(train_y,pred_train)
pred_d12
pred_xgb_gbr1 = (pred+pred_d12)/2.0
pred_xgb_gbr1
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(pred_xgb_gbr1)

result.head()
result.to_csv('xgb_output13.csv', index=False)
model=xgb.XGBRegressor(random_state=0,max_depth=8,n_estimators=150)

model.fit(train_X,train_y)

pred_train = model.predict(train_X)

pred=model.predict(df_test)

mean_squared_error(train_y,pred_train)
pred
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(pred)

result.head()
result.to_csv('xgb_output14.csv', index=False)
xgb_model1=GradientBoostingRegressor(random_state=0,max_depth=14,n_estimators=500,learning_rate=0.01)

xgb_model1.fit(train_X,train_y)

xgb_pred1 = xgb_model1.predict(df_test)

xgb_pred_train = xgb_model1.predict(train_X)

mean_squared_error(xgb_pred_train,train_y)
xgb_pred1
pred1=(xgb_pred1+pred)/2.0
result=pd.DataFrame()

result['Id'] = test_index

result['PredictedValue'] = pd.DataFrame(pred1)

result.tail()
result.to_csv('xgb_output20.csv', index=False)