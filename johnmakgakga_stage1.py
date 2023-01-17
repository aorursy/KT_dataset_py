# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train=pd.read_csv("/kaggle/input/covid19/Train_maskedv2.csv")

test=pd.read_csv("/kaggle/input/covid19/Test_maskedv2.csv")

submission=pd.read_csv("/kaggle/input/covid19/samplesubmissionv2.csv")
train.shape,test.shape
1102/3174
1-0.34719596723377444


train.drop(["ward"], axis=1, inplace=True)

test.drop(["ward",], axis=1, inplace=True)
train.columns
from sklearn.model_selection import train_test_split
x=train.drop("target_pct_vunerable", axis=1)

y=train['target_pct_vunerable']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.34719596723377444,random_state=0)
#Lets run the model.

#As we have to build regression model, lets start with linear regression model

from sklearn.linear_model import LinearRegression
lrmode=LinearRegression(n_jobs=1,normalize=True)

lrmode.fit(x_train, y_train)
lrmodel=LinearRegression(n_jobs=1,normalize=True)

lrmodel.fit(x_test, y_test)
predictedvalue = lrmode.predict(x_train)

print(predictedvalue)
predictedvalue.shape
predictedvalues = lrmodel.predict(x_test)

print(predictedvalues)
predictedvalues.shape
#lets calculate rmse for linear Regression model

from sklearn.metrics import mean_squared_error

lrmodelrmse = np.sqrt(mean_squared_error(predictedvalues, y_test))

print("RMSE value for Linear regression is", lrmodelrmse)
train.columns
z = train.target_pct_vunerable

features = ['total_households', 'total_individuals',

       'dw_00', 'dw_01', 'dw_02', 'dw_03', 'dw_04', 'dw_05', 'dw_06', 'dw_07',

       'dw_08', 'dw_09', 'dw_10', 'dw_11', 'dw_12', 'dw_13', 'psa_00',

       'psa_01', 'psa_02', 'psa_03', 'psa_04', 'stv_00', 'stv_01', 'car_00',

       'car_01', 'lln_00', 'lln_01', 'lan_00', 'lan_01', 'lan_02', 'lan_03',

       'lan_04', 'lan_05', 'lan_06', 'lan_07', 'lan_08', 'lan_09', 'lan_10',

       'lan_11', 'lan_12', 'lan_13', 'lan_14', 'pg_00', 'pg_01', 'pg_02',

       'pg_03', 'pg_04', 'lgt_00']

X = train[features].copy()

X_test = test[features].copy()



# Break off validation set from training data

X_train, X_valid, z_train, z_valid = train_test_split(X, z, train_size=0.6528040327662256, test_size=0.34719596723377444,

                                                      random_state=69)
from sklearn.ensemble import RandomForestRegressor



# Define the models

#model_1 = RandomForestRegressor(n_estimators=500, random_state=500)

model_2 = RandomForestRegressor(n_estimators=650, random_state=69)

#model_3 = RandomForestRegressor(n_estimators=600, criterion='mae', random_state=1102)

#model_4 = RandomForestRegressor(n_estimators=50, min_samples_split=100, random_state=1102)

#model_5 = RandomForestRegressor(n_estimators=20, max_depth=7, random_state=1102)



models = [model_2]#,model_2,model_3, model_4, model_5]

from sklearn.metrics import mean_absolute_error
def score_model(model, X_t=X_train, X_v=X_valid, z_t=z_train, z_v=z_valid):

    model.fit(X_t, z_t)

    preds = model.predict(X_v)

    return mean_absolute_error(z_v, preds)



for i in range(0, len(models)):

    mae = score_model(models[i])

    print("Model %d MAE: %d" % (i+1, mae))

    

    print("Model MAE: ",(i+1, mae))

my_model =RandomForestRegressor(n_estimators=600, random_state=69)
my_model.fit(X, z)
preds_test = my_model.predict(X_test)

print(preds_test)
modelrr = np.sqrt(mean_squared_error(predictedvalues, z_valid))

print("RMSE value for Linear regression is", modelrr)
# = np.sqrt(mean_squared_error(predictedvalues, y_test))

#print("RMSE value for Linear regression is", lrmodelrmse)
#RandomForest Regressor is giving good value, so we can use it as final model

# Save predictions in format used for competition scoring

#predictedvalues= 635



output = pd.DataFrame({'ward':submission['ward'],

                       'total_households':preds_test})

output.to_csv('samplesubmissionv2.csv', index=False)
submission.shape
print(output)