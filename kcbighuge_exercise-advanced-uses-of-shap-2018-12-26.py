import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap

# Environment Set-Up for feedback system.
from learntools.core import binder
binder.bind(globals())
from learntools.ml_insights.ex5 import *
print("Setup Complete")


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('../input/hospital-readmissions/train.csv')
y = data.readmitted
base_features = ['number_inpatient', 'num_medications', 'number_diagnoses', 'num_lab_procedures', 
                 'num_procedures', 'time_in_hospital', 'number_outpatient', 'number_emergency', 
                 'gender_Female', 'payer_code_?', 'medical_specialty_?', 'diag_1_428', 'diag_1_414', 
                 'diabetesMed_Yes', 'A1Cresult_None']

# Some versions of shap package error when mixing bools and numerics
X = data[base_features].astype(float)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# For speed, we will calculate shap values on smaller subset of the validation data
small_val_X = val_X.iloc[:150]
small_val_y = val_y[:150]
my_model = RandomForestClassifier(n_estimators=30, random_state=1).fit(train_X, train_y)
data.describe()
row_to_predict = 10
data_for_prediction = val_X.iloc[row_to_predict]  # use 1 row of data here. Could use multiple rows if desired
data_for_prediction[:8]
# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)
shap_values = explainer.shap_values(data_for_prediction.astype('float'))
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0], data_for_prediction)
[display(x) for x in shap_values]
explainer = shap.TreeExplainer(my_model)
shap_values = explainer.shap_values(small_val_X)

shap.summary_plot(shap_values[1], small_val_X)
# set following variable to 'diag_1_428' or 'payer_code_?'
feature_with_bigger_range_of_effects = 'payer_code_?'
q_1.check()
q_1.solution()
q_2.solution()
shap.summary_plot(shap_values[1], small_val_X)
# Set following var to "diag_1_428" if changing it to 1 has bigger effect.  Else set it to 'payer_code_?'
bigger_effect_when_changed = 'payer_code_?'
q_3.check()
q_3.solution()
q_4.solution()
q_5.solution()
# create dataframe of all data
df = small_val_X.join(small_val_y)
df.head()
df.columns
# https://www.kaggle.com/dansbecker/advanced-uses-of-shap-values
# calculate shap values. This is what we will plot.
shap_df = explainer.shap_values(df)

# make plot.
feat = 'number_inpatient'
target = 'readmitted'
shap.dependence_plot(feat, shap_df[1], df, interaction_index=target)
shap.summary_plot(shap_values[1], small_val_X)
shap.dependence_plot('num_lab_procedures', shap_values[1], small_val_X)
shap.dependence_plot('num_medications', shap_values[1], small_val_X)
q_6.solution()