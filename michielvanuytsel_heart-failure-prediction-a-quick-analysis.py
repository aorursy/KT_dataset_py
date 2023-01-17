import numpy as np

import pandas as pd



random_seed = 297



import os

clinical_data_filepath = "../input/heart-failure-clinical-data/heart_failure_clinical_records_dataset.csv"

clinical_data = pd.read_csv(clinical_data_filepath)
import pandas_profiling

pandas_profile = pandas_profiling.ProfileReport(clinical_data)
pandas_profile.to_widgets()
clean_data = clinical_data.astype({'anaemia': 'bool', 'diabetes': 'bool', 'high_blood_pressure':'bool', 'smoking':'bool'})

clean_data.head()
from sklearn.model_selection import train_test_split



# Split into X and y dataset

y = clean_data.DEATH_EVENT

all_features = ['age', 'anaemia', 'creatinine_phosphokinase', 'diabetes','ejection_fraction', 'high_blood_pressure', 'platelets','serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time']

features = all_features #For this initial version, we'll just take all features for input. Due to the low amount of samples compared to features, we do fear for overfitting

X = clean_data[features]



# Create train/test set

train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2, random_state=random_seed)
from sklearn.ensemble import RandomForestClassifier

# Specify Model

hearth_failure_RFCmodel = RandomForestClassifier(random_state=random_seed)

# Fit Model

hearth_failure_RFCmodel.fit(train_X, train_y)
from sklearn.metrics import plot_roc_curve

plot_roc_curve(hearth_failure_RFCmodel, val_X, val_y)
import xgboost



# Specify Model

hearth_failure_XGBmodel = xgboost.XGBClassifier(random_state=random_seed)

# Fit Model

hearth_failure_XGBmodel.fit(train_X, train_y)
from sklearn.metrics import plot_roc_curve

plot_roc_curve(hearth_failure_XGBmodel, val_X, val_y)
xgboost.plot_importance(hearth_failure_XGBmodel)