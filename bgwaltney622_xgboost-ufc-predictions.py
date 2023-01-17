%matplotlib inline



import numpy as np

import pandas as pd

import xgboost

import shap

import matplotlib.pyplot as plt

from xgboost import XGBRegressor

from sklearn.feature_selection import VarianceThreshold

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

from sklearn import metrics
data = pd.read_csv('../input/ufcdata/preprocessed_data.csv')
#Instantiate classes need to pre_process data

scaler = StandardScaler()

label_encoder = LabelEncoder()

variance_selector = VarianceThreshold()



# Create a function to select features off variance threshold

def variance_threshold_selector(data, features, threshold=0.5):

    variance_selector = VarianceThreshold(threshold)

    variance_selector.fit_transform(data[features])

    return data[data.columns[variance_selector.get_support(indices=True)]]
#Label encode Winner col for XGBRegression

data["Winner"] = label_encoder.fit_transform(data["Winner"])

y = data["Winner"]



# Inital selections and split dataset

X = data.drop(['Winner'], axis=1)

init_features = X.columns

X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,

                                                      random_state=0)



# Select and assign features with variance threshold

X_train = variance_threshold_selector(X_train, init_features)

features = X_train.columns

X_valid = X_valid[features]



# Scale features

X_train_scal = scaler.fit_transform(X_train)

X_valid_scal = scaler.fit_transform(X_valid)
# Creat the Model

ufc_XGB_model = XGBRegressor(n_estimators=500, learning_rate=0.5)



# Fit the Model

ufc_XGB_model.fit(X_train_scal, y_train, 

             early_stopping_rounds=5, 

             eval_set=[(X_valid_scal, y_valid)],

             verbose=False)



#Make predictions and show score

y_pred = ufc_XGB_model.predict(X_valid_scal)

score = metrics.roc_auc_score(y_valid, y_pred)

print(f"ufc_XGB_model Test score: {score}")
# Init Visualization

explainer = shap.TreeExplainer(ufc_XGB_model)

shap_values = explainer.shap_values(X_valid)

shap.initjs()
# Summary of feature importance

shap.summary_plot(shap_values, X_valid)
# Impacts on indvidual data (can be modified to view others)

shap.force_plot(explainer.expected_value, shap_values[0,:], X_valid.iloc[0,:])
# Bar plot of most important features

shap.summary_plot(shap_values, X_valid, plot_type="bar")