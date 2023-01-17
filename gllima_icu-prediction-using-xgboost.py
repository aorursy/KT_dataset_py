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
import pandas as pd

# renamed dataset Kaggle_Sirio_Libanes_ICU_Prediction.xlsx

data = pd.read_excel("/kaggle/input/covid19/Kaggle_Sirio_Libanes_ICU_Prediction.xlsx")
data.head()
data.describe()
data.info()
def perc_to_int(percentile):

    #print(percentile, ''.join(filter(str.isdigit, percentile)))

    return(int(''.join(filter(str.isdigit, percentile))))

   

def wdw_to_int(window):

    if window == "ABOVE_12":

        window = "ABOVE-13"

    #print(window, ''.join(filter(str.isdigit, window.split("-")[1])))

    return(int(''.join(filter(str.isdigit, window.split("-")[1]))))

          



print("Convert Age Percentile to number")

print(data.AGE_PERCENTIL.unique())

data['AGE_PERC'] = data.AGE_PERCENTIL.apply(lambda x: perc_to_int(x))



print("Convert Window to number")

print(data.WINDOW.unique())

data['WINDOW_HOURS'] = data.WINDOW.apply(lambda x: wdw_to_int(x))



data.describe()
data.to_csv("dataset_prep.csv")
import xgboost as xgb

from sklearn.metrics import mean_squared_error

import pandas as pd

import numpy as np



from sklearn.metrics import confusion_matrix

from termcolor import colored
all_feats = ["PATIENT_VISIT_IDENTIFIER", "AGE_ABOVE65", "GENDER", "DISEASE GROUPING 1", "DISEASE GROUPING 2", "DISEASE GROUPING 3", 

             "DISEASE GROUPING 4", "DISEASE GROUPING 5", "DISEASE GROUPING 6", "HTN","IMMUNOCOMPROMISED", "OTHER", "ALBUMIN_MEDIAN",

             "ALBUMIN_MEAN", "ALBUMIN_MIN", "ALBUMIN_MAX", "ALBUMIN_DIFF", "BE_ARTERIAL_MEDIAN", "BE_ARTERIAL_MEAN", "BE_ARTERIAL_MIN",

             "BE_ARTERIAL_MAX", "BE_ARTERIAL_DIFF", "BE_VENOUS_MEDIAN", "BE_VENOUS_MEAN", "BE_VENOUS_MIN", "BE_VENOUS_MAX", "BE_VENOUS_DIFF",

             "BIC_ARTERIAL_MEDIAN", "BIC_ARTERIAL_MEAN", "BIC_ARTERIAL_MIN", "BIC_ARTERIAL_MAX", "BIC_ARTERIAL_DIFF", "BIC_VENOUS_MEDIAN",

             "BIC_VENOUS_MEAN", "BIC_VENOUS_MIN", "BIC_VENOUS_MAX", "BIC_VENOUS_DIFF", "BILLIRUBIN_MEDIAN", "BILLIRUBIN_MEAN", 

             "BILLIRUBIN_MIN", "BILLIRUBIN_MAX", "BILLIRUBIN_DIFF", "BLAST_MEDIAN", "BLAST_MEAN", "BLAST_MIN", "BLAST_MAX", "BLAST_DIFF", 

             "CALCIUM_MEDIAN", "CALCIUM_MEAN", "CALCIUM_MIN", "CALCIUM_MAX", "CALCIUM_DIFF", "CREATININ_MEDIAN", "CREATININ_MEAN", 

             "CREATININ_MIN", "CREATININ_MAX", "CREATININ_DIFF", "FFA_MEDIAN", "FFA_MEAN", "FFA_MIN", "FFA_MAX", "FFA_DIFF", "GGT_MEDIAN", 

             "GGT_MEAN", "GGT_MIN", "GGT_MAX", "GGT_DIFF", "GLUCOSE_MEDIAN", "GLUCOSE_MEAN", "GLUCOSE_MIN", "GLUCOSE_MAX", "GLUCOSE_DIFF", 

             "HEMATOCRITE_MEDIAN", "HEMATOCRITE_MEAN", "HEMATOCRITE_MIN", "HEMATOCRITE_MAX", "HEMATOCRITE_DIFF", "HEMOGLOBIN_MEDIAN", 

             "HEMOGLOBIN_MEAN", "HEMOGLOBIN_MIN", "HEMOGLOBIN_MAX", "HEMOGLOBIN_DIFF", "INR_MEDIAN", "INR_MEAN", "INR_MIN", "INR_MAX", 

             "INR_DIFF", "LACTATE_MEDIAN", "LACTATE_MEAN", "LACTATE_MIN", "LACTATE_MAX", "LACTATE_DIFF", "LEUKOCYTES_MEDIAN", "LEUKOCYTES_MEAN",

             "LEUKOCYTES_MIN", "LEUKOCYTES_MAX", "LEUKOCYTES_DIFF", "LINFOCITOS_MEDIAN", "LINFOCITOS_MEAN", "LINFOCITOS_MIN", "LINFOCITOS_MAX", 

             "LINFOCITOS_DIFF", "NEUTROPHILES_MEDIAN", "NEUTROPHILES_MEAN", "NEUTROPHILES_MIN", "NEUTROPHILES_MAX", "NEUTROPHILES_DIFF", 

             "P02_ARTERIAL_MEDIAN", "P02_ARTERIAL_MEAN", "P02_ARTERIAL_MIN", "P02_ARTERIAL_MAX", "P02_ARTERIAL_DIFF", "P02_VENOUS_MEDIAN", 

             "P02_VENOUS_MEAN", "P02_VENOUS_MIN", "P02_VENOUS_MAX", "P02_VENOUS_DIFF", "PC02_ARTERIAL_MEDIAN", "PC02_ARTERIAL_MEAN", 

             "PC02_ARTERIAL_MIN", "PC02_ARTERIAL_MAX", "PC02_ARTERIAL_DIFF", "PC02_VENOUS_MEDIAN", "PC02_VENOUS_MEAN", "PC02_VENOUS_MIN", 

             "PC02_VENOUS_MAX", "PC02_VENOUS_DIFF", "PCR_MEDIAN", "PCR_MEAN", "PCR_MIN", "PCR_MAX", "PCR_DIFF", "PH_ARTERIAL_MEDIAN", 

             "PH_ARTERIAL_MEAN", "PH_ARTERIAL_MIN", "PH_ARTERIAL_MAX", "PH_ARTERIAL_DIFF", "PH_VENOUS_MEDIAN", "PH_VENOUS_MEAN", "PH_VENOUS_MIN",

             "PH_VENOUS_MAX", "PH_VENOUS_DIFF", "PLATELETS_MEDIAN", "PLATELETS_MEAN", "PLATELETS_MIN", "PLATELETS_MAX", "PLATELETS_DIFF", 

             "POTASSIUM_MEDIAN", "POTASSIUM_MEAN", "POTASSIUM_MIN", "POTASSIUM_MAX", "POTASSIUM_DIFF", "SAT02_ARTERIAL_MEDIAN", 

             "SAT02_ARTERIAL_MEAN", "SAT02_ARTERIAL_MIN", "SAT02_ARTERIAL_MAX", "SAT02_ARTERIAL_DIFF", "SAT02_VENOUS_MEDIAN", "SAT02_VENOUS_MEAN",

             "SAT02_VENOUS_MIN", "SAT02_VENOUS_MAX", "SAT02_VENOUS_DIFF", "SODIUM_MEDIAN", "SODIUM_MEAN", "SODIUM_MIN", "SODIUM_MAX", 

             "SODIUM_DIFF", "TGO_MEDIAN", "TGO_MEAN", "TGO_MIN", "TGO_MAX", "TGO_DIFF", "TGP_MEDIAN", "TGP_MEAN", "TGP_MIN", "TGP_MAX", 

             "TGP_DIFF", "TTPA_MEDIAN", "TTPA_MEAN", "TTPA_MIN", "TTPA_MAX", "TTPA_DIFF", "UREA_MEDIAN", "UREA_MEAN", "UREA_MIN", "UREA_MAX", 

             "UREA_DIFF", "DIMER_MEDIAN", "DIMER_MEAN", "DIMER_MIN", "DIMER_MAX", "DIMER_DIFF", "BLOODPRESSURE_DIASTOLIC_MEAN", 

             "BLOODPRESSURE_SISTOLIC_MEAN", "HEART_RATE_MEAN", "RESPIRATORY_RATE_MEAN", "TEMPERATURE_MEAN", "OXYGEN_SATURATION_MEAN", 

             "BLOODPRESSURE_DIASTOLIC_MEDIAN", "BLOODPRESSURE_SISTOLIC_MEDIAN", "HEART_RATE_MEDIAN", "RESPIRATORY_RATE_MEDIAN", 

             "TEMPERATURE_MEDIAN", "OXYGEN_SATURATION_MEDIAN", "BLOODPRESSURE_DIASTOLIC_MIN", "BLOODPRESSURE_SISTOLIC_MIN", "HEART_RATE_MIN", 

             "RESPIRATORY_RATE_MIN", "TEMPERATURE_MIN", "OXYGEN_SATURATION_MIN", "BLOODPRESSURE_DIASTOLIC_MAX", "BLOODPRESSURE_SISTOLIC_MAX", 

             "HEART_RATE_MAX", "RESPIRATORY_RATE_MAX", "TEMPERATURE_MAX", "OXYGEN_SATURATION_MAX", "BLOODPRESSURE_DIASTOLIC_DIFF", 

             "BLOODPRESSURE_SISTOLIC_DIFF", "HEART_RATE_DIFF", "RESPIRATORY_RATE_DIFF", "TEMPERATURE_DIFF", "OXYGEN_SATURATION_DIFF", 

             "BLOODPRESSURE_DIASTOLIC_DIFF_REL", "BLOODPRESSURE_SISTOLIC_DIFF_REL", "HEART_RATE_DIFF_REL", "RESPIRATORY_RATE_DIFF_REL", 

             "TEMPERATURE_DIFF_REL", "OXYGEN_SATURATION_DIFF_REL"]



top_42_feats = ["RESPIRATORY_RATE_MAX", "RESPIRATORY_RATE_DIFF", "RESPIRATORY_RATE_DIFF_REL", "RESPIRATORY_RATE_MEAN", "BLOODPRESSURE_SISTOLIC_DIFF", 

                "BLOODPRESSURE_DIASTOLIC_MIN", "RESPIRATORY_RATE_MIN", "BLOODPRESSURE_SISTOLIC_DIFF_REL", "RESPIRATORY_RATE_MEDIAN", "BLOODPRESSURE_SISTOLIC_MAX", 

                "BLOODPRESSURE_DIASTOLIC_DIFF_REL", "HEART_RATE_DIFF_REL", "TEMPERATURE_DIFF_REL", "BLOODPRESSURE_SISTOLIC_MIN", "TEMPERATURE_MIN", "LACTATE_MAX", 

                "BLOODPRESSURE_DIASTOLIC_MEDIAN", "BLOODPRESSURE_DIASTOLIC_MEAN", "OXYGEN_SATURATION_MIN", "BLOODPRESSURE_DIASTOLIC_MAX", "HEART_RATE_MAX", "HEART_RATE_MIN", "TEMPERATURE_MAX", 

                "BLOODPRESSURE_SISTOLIC_MEAN", "BLOODPRESSURE_SISTOLIC_MEDIAN", "TEMPERATURE_MEAN", "TEMPERATURE_MEDIAN", "WINDOW_HOURS", "CALCIUM_MAX", "OXYGEN_SATURATION_MEAN", 

                "HEART_RATE_MEDIAN", "AGE_PERC", "LEUKOCYTES_MAX", "NEUTROPHILES_MAX", "PCR_MAX", "HTN", "GLUCOSE_MAX", "DISEASE GROUPING 2", "PATIENT_VISIT_IDENTIFIER", 

                "OTHER", "GENDER", "IMMUNOCOMPROMISED"]





top_30_feats = ["RESPIRATORY_RATE_MAX", "RESPIRATORY_RATE_DIFF", "RESPIRATORY_RATE_DIFF_REL", "RESPIRATORY_RATE_MEAN", "BLOODPRESSURE_SISTOLIC_DIFF", 

                "BLOODPRESSURE_DIASTOLIC_MIN", "RESPIRATORY_RATE_MIN", "BLOODPRESSURE_SISTOLIC_DIFF_REL", "RESPIRATORY_RATE_MEDIAN", "BLOODPRESSURE_SISTOLIC_MAX", 

                "BLOODPRESSURE_DIASTOLIC_DIFF_REL", "HEART_RATE_DIFF_REL", "TEMPERATURE_DIFF_REL", "BLOODPRESSURE_SISTOLIC_MIN", "TEMPERATURE_MIN", "LACTATE_MAX", 

                "BLOODPRESSURE_DIASTOLIC_MEDIAN", "BLOODPRESSURE_DIASTOLIC_MEAN", "OXYGEN_SATURATION_MIN", "BLOODPRESSURE_DIASTOLIC_MAX", "HEART_RATE_MAX", "HEART_RATE_MIN", "TEMPERATURE_MAX", 

                "BLOODPRESSURE_SISTOLIC_MEAN"]





top_20_feats = ["RESPIRATORY_RATE_MAX", "RESPIRATORY_RATE_DIFF", "RESPIRATORY_RATE_DIFF_REL", "RESPIRATORY_RATE_MEAN", "BLOODPRESSURE_SISTOLIC_DIFF", 

                "BLOODPRESSURE_DIASTOLIC_MIN", "RESPIRATORY_RATE_MIN", "BLOODPRESSURE_SISTOLIC_DIFF_REL", "RESPIRATORY_RATE_MEDIAN", "BLOODPRESSURE_SISTOLIC_MAX", 

                "BLOODPRESSURE_DIASTOLIC_DIFF_REL", "HEART_RATE_DIFF_REL", "TEMPERATURE_DIFF_REL", "BLOODPRESSURE_SISTOLIC_MIN", "TEMPERATURE_MIN", "LACTATE_MAX", 

                "BLOODPRESSURE_DIASTOLIC_MEDIAN", "BLOODPRESSURE_DIASTOLIC_MEAN", "OXYGEN_SATURATION_MIN"]



top_10_feats = ["RESPIRATORY_RATE_MAX", "RESPIRATORY_RATE_DIFF", "RESPIRATORY_RATE_DIFF_REL", "RESPIRATORY_RATE_MEAN", "BLOODPRESSURE_SISTOLIC_DIFF", 

                "BLOODPRESSURE_DIASTOLIC_MIN", "RESPIRATORY_RATE_MIN", "BLOODPRESSURE_SISTOLIC_DIFF_REL", "RESPIRATORY_RATE_MEDIAN", "BLOODPRESSURE_SISTOLIC_MAX"]


params = {

    "eta": 0.2,

    "max_depth": 4,

    "objective": "binary:logistic",

    "silent": 1,

    'n_estimators': 1000,

    "eval_metric": "logloss"

}

X = data.loc[:, all_feats]

y = data.ICU
data_dmatrix = xgb.DMatrix(data=X,label=y)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_by_trainree = 0.3, learning_rate = 0.1,

                max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(X_train,y_train)



preds = xg_reg.predict(X_test)
rmse_all = np.sqrt(mean_squared_error(y_test, preds))

print("RMSE: %f" % (rmse_all))
cv_results_all = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,

                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
cv_results_all.head()
print((cv_results_all["test-rmse-mean"]).tail(1))
dt = xgb.DMatrix(X_train.to_numpy(),label=y_train.to_numpy())

dv = xgb.DMatrix(X_test.to_numpy(),label=y_test.to_numpy())





model = xgb.train(params, dt, 3000, [(dt, "train"),(dv, "valid")], verbose_eval=200)



#Prediction on validation set

y_pred = model.predict(dv)



# Making the Confusion Matrix

cm = confusion_matrix(y_test, (y_pred>0.5))

print(colored('The Confusion Matrix is: ', 'red'),'\n', cm)

# Calculate the accuracy on test set

predict_accuracy_on_test_set = (cm[0,0] + cm[1,1])/(cm[0,0] + cm[1,1]+cm[1,0] + cm[0,1])

print(colored('The Accuracy on Test Set is: ', 'blue'), colored(predict_accuracy_on_test_set, 'blue'))
X = data.loc[:, top_42_feats]

y = data.ICU
data_dmatrix = xgb.DMatrix(data=X,label=y)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_by_trainree = 0.3, learning_rate = 0.1,

                max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(X_train,y_train)



preds = xg_reg.predict(X_test)
rmse_42 = np.sqrt(mean_squared_error(y_test, preds))

print("RMSE: %f" % (rmse_42))
cv_results_42 = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,

                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
cv_results_42.head()
print((cv_results_42["test-rmse-mean"]).tail(1))
dt = xgb.DMatrix(X_train.to_numpy(),label=y_train.to_numpy())

dv = xgb.DMatrix(X_test.to_numpy(),label=y_test.to_numpy())



model = xgb.train(params, dt, 3000, [(dt, "train"),(dv, "valid")], verbose_eval=200)



#Prediction on validation set

y_pred = model.predict(dv)



# Making the Confusion Matrix

cm = confusion_matrix(y_test, (y_pred>0.5))

print(colored('The Confusion Matrix is: ', 'red'),'\n', cm)

# Calculate the accuracy on test set

predict_accuracy_on_test_set = (cm[0,0] + cm[1,1])/(cm[0,0] + cm[1,1]+cm[1,0] + cm[0,1])

print(colored('The Accuracy on Test Set is: ', 'blue'), colored(predict_accuracy_on_test_set, 'blue'))
X = data.loc[:, top_30_feats]

y = data.ICU
data_dmatrix = xgb.DMatrix(data=X,label=y)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_by_trainree = 0.3, learning_rate = 0.1,

                max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(X_train,y_train)



preds = xg_reg.predict(X_test)
rmse_30 = np.sqrt(mean_squared_error(y_test, preds))

print("RMSE: %f" % (rmse_30))
cv_results_30 = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,

                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
cv_results_30.head()
print((cv_results_30["test-rmse-mean"]).tail(1))
dt = xgb.DMatrix(X_train.to_numpy(),label=y_train.to_numpy())

dv = xgb.DMatrix(X_test.to_numpy(),label=y_test.to_numpy())



model = xgb.train(params, dt, 3000, [(dt, "train"),(dv, "valid")], verbose_eval=200)



#Prediction on validation set

y_pred = model.predict(dv)



# Making the Confusion Matrix

cm = confusion_matrix(y_test, (y_pred>0.5))

print(colored('The Confusion Matrix is: ', 'red'),'\n', cm)

# Calculate the accuracy on test set

predict_accuracy_on_test_set = (cm[0,0] + cm[1,1])/(cm[0,0] + cm[1,1]+cm[1,0] + cm[0,1])

print(colored('The Accuracy on Test Set is: ', 'blue'), colored(predict_accuracy_on_test_set, 'blue'))
X = data.loc[:, top_20_feats]

y = data.ICU
data_dmatrix = xgb.DMatrix(data=X,label=y)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_by_trainree = 0.3, learning_rate = 0.1,

                max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(X_train,y_train)



preds = xg_reg.predict(X_test)
rmse_20 = np.sqrt(mean_squared_error(y_test, preds))

print("RMSE: %f" % (rmse_20))
cv_results_20 = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,

                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
cv_results_20.head()
print((cv_results_20["test-rmse-mean"]).tail(1))
dt = xgb.DMatrix(X_train.to_numpy(),label=y_train.to_numpy())

dv = xgb.DMatrix(X_test.to_numpy(),label=y_test.to_numpy())





model = xgb.train(params, dt, 3000, [(dt, "train"),(dv, "valid")], verbose_eval=200)



#Prediction on validation set

y_pred = model.predict(dv)



# Making the Confusion Matrix

cm = confusion_matrix(y_test, (y_pred>0.5))

print(colored('The Confusion Matrix is: ', 'red'),'\n', cm)

# Calculate the accuracy on test set

predict_accuracy_on_test_set = (cm[0,0] + cm[1,1])/(cm[0,0] + cm[1,1]+cm[1,0] + cm[0,1])

print(colored('The Accuracy on Test Set is: ', 'blue'), colored(predict_accuracy_on_test_set, 'blue'))
X = data.loc[:, top_10_feats]

y = data.ICU
data_dmatrix = xgb.DMatrix(data=X,label=y)
from sklearn.model_selection import train_test_split



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_by_trainree = 0.3, learning_rate = 0.1,

                max_depth = 5, alpha = 10, n_estimators = 10)
xg_reg.fit(X_train,y_train)



preds = xg_reg.predict(X_test)
rmse_10 = np.sqrt(mean_squared_error(y_test, preds))

print("RMSE: %f" % (rmse_10))
cv_results_10 = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,

                    num_boost_round=50,early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)
cv_results_10.head()
print((cv_results_10["test-rmse-mean"]).tail(1))
y_test.head(), preds[:5]
dt = xgb.DMatrix(X_train.to_numpy(),label=y_train.to_numpy())

dv = xgb.DMatrix(X_test.to_numpy(),label=y_test.to_numpy())



model = xgb.train(params, dt, 3000, [(dt, "train"),(dv, "valid")], verbose_eval=200)



#Prediction on validation set

y_pred = model.predict(dv)



# Making the Confusion Matrix

cm = confusion_matrix(y_test, (y_pred>0.5))

print(colored('The Confusion Matrix is: ', 'red'),'\n', cm)

# Calculate the accuracy on test set

predict_accuracy_on_test_set = (cm[0,0] + cm[1,1])/(cm[0,0] + cm[1,1]+cm[1,0] + cm[0,1])

print(colored('The Accuracy on Test Set is: ', 'blue'), colored(predict_accuracy_on_test_set, 'blue'))