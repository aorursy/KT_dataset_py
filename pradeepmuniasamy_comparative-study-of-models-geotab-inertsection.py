import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import folium

from folium.plugins import HeatMap
# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train_df=pd.read_csv("/kaggle/input/bigquery-geotab-intersection-congestion/train.csv")

test_df=pd.read_csv("/kaggle/input/bigquery-geotab-intersection-congestion/test.csv")
#Creating Dummies for train Data

dfen = pd.get_dummies(train_df["EntryHeading"],prefix = 'en')

dfex = pd.get_dummies(train_df["ExitHeading"],prefix = 'ex')

train_df = pd.concat([train_df,dfen],axis=1)

train_df = pd.concat([train_df,dfex],axis=1)



#Creating Dummies for test Data

dfent = pd.get_dummies(test_df["EntryHeading"],prefix = 'en')

dfext = pd.get_dummies(test_df["ExitHeading"],prefix = 'ex')

test_df = pd.concat([test_df,dfent],axis=1)

test_df = pd.concat([test_df,dfext],axis=1)

#Training Data

X = train_df[["IntersectionId","Hour","Weekend","Month",'en_E',

       'en_N', 'en_NE', 'en_NW', 'en_S', 'en_SE', 'en_SW', 'en_W', 'ex_E',

       'ex_N', 'ex_NE', 'ex_NW', 'ex_S', 'ex_SE', 'ex_SW', 'ex_W']]

y1 = train_df["TotalTimeStopped_p20"]

y2 = train_df["TotalTimeStopped_p50"]

y3 = train_df["TotalTimeStopped_p80"]

y4 = train_df["DistanceToFirstStop_p20"]

y5 = train_df["DistanceToFirstStop_p50"]

y6 = train_df["DistanceToFirstStop_p80"]
testX = test_df[["IntersectionId","Hour","Weekend","Month",'en_E','en_N', 'en_NE', 'en_NW', 'en_S', 

              'en_SE', 'en_SW', 'en_W', 'ex_E','ex_N', 'ex_NE', 'ex_NW', 'ex_S', 'ex_SE', 'ex_SW', 'ex_W']]
import statsmodels.api as sm

results=sm.OLS(y1,X).fit()

results.summary()
model_lr1=sm.OLS(y1,X).fit()

pred_lr1=model_lr1.predict(testX)

model_lr2=sm.OLS(y2,X).fit()

pred_lr2=model_lr2.predict(testX)

model_lr3=sm.OLS(y3,X).fit()

pred_lr3=model_lr3.predict(testX)

model_lr4=sm.OLS(y4,X).fit()

pred_lr4=model_lr4.predict(testX)

model_lr5=sm.OLS(y5,X).fit()

pred_lr5=model_lr5.predict(testX)

model_lr6=sm.OLS(y6,X).fit()

pred_lr6=model_lr6.predict(testX)
# Appending all predictions

prediction_lr = []

for i in range(len(pred_lr1)):

    for j in [pred_lr1,pred_lr2,pred_lr3,pred_lr4,pred_lr5,pred_lr6]:

        prediction_lr.append(j[i])

submission_lr = pd.read_csv("../input/bigquery-geotab-intersection-congestion/sample_submission.csv")

submission_lr["Target"] = prediction_lr

submission_lr.to_csv("Submission_lr.csv",index = False)        
from sklearn.linear_model import Lasso

from sklearn.kernel_ridge import KernelRidge

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import RobustScaler
#Build the model

lasso = make_pipeline(RobustScaler(), Lasso(alpha =0.0007, random_state=1))
lasso.fit(X,y1)

predict_l1= lasso.predict(testX)

lasso.fit(X,y2)

predict_l2= lasso.predict(testX)

lasso.fit(X,y3)

predict_l3= lasso.predict(testX)

lasso.fit(X,y4)

predict_l4= lasso.predict(testX)

lasso.fit(X,y5)

predict_l5= lasso.predict(testX)

lasso.fit(X,y6)

predict_l6= lasso.predict(testX)
# Appending all predictions

prediction_l = []

for i in range(len(predict_l1)):

    for j in [predict_l1,predict_l2,predict_l3,predict_l4,predict_l5,predict_l6]:

        prediction_l.append(j[i])

submission_l = pd.read_csv("../input/bigquery-geotab-intersection-congestion/sample_submission.csv")

submission_l["Target"] = prediction_l

submission_l.to_csv("Submission_l.csv",index = False)     
from sklearn.linear_model import Ridge

ridge = make_pipeline(RobustScaler(), Ridge(alpha =20, random_state=42))
ridge.fit(X,y1)

predict_r1= ridge.predict(testX)

ridge.fit(X,y2)

predict_r2= ridge.predict(testX)

ridge.fit(X,y3)

predict_r3= ridge.predict(testX)

lasso.fit(X,y4)

predict_r4= ridge.predict(testX)

ridge.fit(X,y5)

predict_r5= ridge.predict(testX)

ridge.fit(X,y6)

predict_r6= ridge.predict(testX)
# Appending all predictions

prediction_r = []

for i in range(len(predict_r1)):

    for j in [predict_r1,predict_r2,predict_r3,predict_r4,predict_r5,predict_r6]:

        prediction_r.append(j[i])

submission_r = pd.read_csv("../input/bigquery-geotab-intersection-congestion/sample_submission.csv")

submission_r["Target"] = prediction_r

submission_r.to_csv("Submission_r.csv",index = False)     
from catboost import CatBoostRegressor

cb_model= CatBoostRegressor(iterations=700,

                             learning_rate=0.02,

                             depth=12,

                             eval_metric='RMSE',

                             random_seed = 23,

                             bagging_temperature = 0.2,

                             od_type='Iter',

                             metric_period = 75,

                             od_wait=100)

cb_model.fit(X, y1)

pred_CB1=cb_model.predict(testX)

cb_model.fit(X, y2)

pred_CB2=cb_model.predict(testX)

cb_model.fit(X, y3)

pred_CB3=cb_model.predict(testX)

cb_model.fit(X, y4)

pred_CB4=cb_model.predict(testX)

cb_model.fit(X, y5)

pred_CB5=cb_model.predict(testX)

cb_model.fit(X, y6)

pred_CB6=cb_model.predict(testX)
# Appending all predictions

prediction_CB = []

for i in range(len(pred_CB1)):

    for j in [pred_CB1,pred_CB2,pred_CB3,pred_CB4,pred_CB5,pred_CB6]:

        prediction_CB.append(j[i])

        

submission_CB = pd.read_csv("../input/bigquery-geotab-intersection-congestion/sample_submission.csv")

submission_CB["Target"] = prediction_CB

submission_CB.to_csv("Submission_CB.csv",index = False)
from IPython.display import FileLink

FileLink(r'Submission_GB.csv')
submission_ensemble= pd.read_csv("../input/bigquery-geotab-intersection-congestion/sample_submission.csv")

submission_ensemble['Target'] =( 0.9* submission_CB["Target"] +0.025* submission_l["Target"]+0.025*submission_r["Target"] +0.025*submission_lr["Target"])

submission_ensemble.to_csv("submission_ensemble.csv", index = False)