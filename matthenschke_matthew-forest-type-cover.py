# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



# Any results you write to the current directory are saved as output.
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
train.head()
# from sklearn import preprocessing, pipeline, ensemble, impute

# categorical_cleanup = pipeline.make_pipeline(    impute.

#  )
# preprocess data

# feature engineering



X = train.drop(["Id", "Cover_Type"], axis=1)

y = train["Cover_Type"]
# create our models

import xgboost

xgb = xgboost.XGBClassifier()



from sklearn.linear_model import LogisticRegression as LR

lreg = LR(multi_class = "ovr")



from sklearn.ensemble import GradientBoostingClassifier as GBC 

gbc = GBC()



from sklearn.multiclass import OneVsRestClassifier as OVR

ovr1 = OVR(gbc)





from sklearn.neighbors import KNeighborsClassifier as KNC

knn = KNC()



ovr2 = OVR(knn)



from sklearn.ensemble import RandomForestClassifier as RFC

rf = RFC()





from sklearn import model_selection

# using train test split

X_train, X_val, y_train, y_val = model_selection.train_test_split(X, y, test_size =0.25,random_state =7)

from sklearn.model_selection import GridSearchCV as GS

# parameters

params = {

    "n_estimators" : [250, 350, 450, 550, 650]

}



# Grid Search

gs = GS(xgb, params)

# # set up the stack

# classifiers = [xgb, lreg, gbc, ovr1, knn, ovr2, rf]

# from mlxtend.classifier import StackingClassifier as SC

# stack = SC(classifiers = classifiers, meta_classifier = rf)
# # using grid search for parameter tuning

# from sklearn.model_selection import GridSearchCV as GS



# # parameters

# params = {

#     "n_estimators" : [25,50,75,100, 125,150,175,200]

# }



# # Grid Search

# gs = GS(rf, params)
# fit the data into all of the models

# stack.fit(X_train, y_train)

rf.fit(X_train, y_train)

gs.fit(X_train, y_train)

# xgb.fit(X_train, y_train)

# lreg.fit(X_train, y_train)

# ovr1.fit(X_train, y_train)

# knn.fit(X_train, y_train)

# ovr2.fit(X_train, y_train)

# # y_pred_xgb = xgb.predict(X_val)

# # y_pred_lreg = lreg.predict(X_val)

# # y_pred_gbc = gbc.predict(X_val)

# # y_pred_ovr1 = ovr.predict(X_val)

# # y_pred_knn = knn.predict(X_val)

# # y_pred_ovr2 = ovr2.predict(X_val)

y_pred_gs = gs.predict(X_val)

# y_pred_stack = stack.predict(X_val)

y_pred_rf = rf.predict(X_val)
# # calculate the root mean squared error for prediction for all models

from sklearn.metrics import mean_squared_log_error

from math import sqrt

# print("RMSE Scores:")

# # print("The RMSE for XGBoost is {}".format(sqrt(mean_squared_log_error(y_val, y_pred_xgb))))

# # print("The RMSE for Logistic Regression is {}".format(sqrt(mean_squared_log_error(y_val, y_pred_lreg))))

# # print("The RMSE for Gradient Boosting Classification is {}".format(sqrt(mean_squared_log_error(y_val, y_pred_gbc))))

# # print("The RMSE for One vs Rest Classifier is {}".format(sqrt(mean_squared_log_error(y_val, y_pred_ovr1))))

# # print("The RMSE for KNearest Classifier is {}".format(sqrt(mean_squared_log_error(y_val, y_pred_knn))))

# # print("The RMSE for One vs Rest Classifier is {}".format(sqrt(mean_squared_log_error(y_val, y_pred_ovr2))))

# print("The RMSE for Stacking Classifier is {}".format(sqrt(mean_squared_log_error(y_val, y_pred_stack))))

print("The RMSE for Grid Search is {}".format(sqrt(mean_squared_log_error(y_val, y_pred_gs))))

# print("The RMSE for Random Forest with Grid Search is {}".format(sqrt(mean_squared_log_error(y_val, y_pred_gs))))




y_pred = gs.predict(test.drop("Id", axis=1))

submission = pd.DataFrame(

{

    "Id": test["Id"],

    "Cover_Type": y_pred

})

submission.to_csv("submission.csv", index=False)
