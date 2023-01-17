# setup feedback system

from learntools.core import binder

binder.bind(globals())

from learntools.ml_explainability.ex5 import *

print("Setup is completed.")



# import numpy as np

# from sklearn.ensemble import RandomForestRegressor





import pandas as pd

data = pd.read_csv('../input/hospital-readmissions/train.csv')



y = data['readmitted']



base_features = [

    'number_inpatient', 'num_medications', 'number_diagnoses', 'num_lab_procedures',

    'num_procedures', 'time_in_hospital', 'number_outpatient', 'number_emergency',

    'gender_Female', 'payer_code_?', 'medical_specialty_?', 'diag_1_428', 'diag_1_414',

    'diabetesMed_Yes', 'A1Cresult_None'

]

# some versions of shap package error when mixing bools and numerics

X = data[base_features].astype(float)



from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# for speed, we will calculate shap values on smaller subset of the validation data

small_val_X = val_X.iloc[:150]



from sklearn.ensemble import RandomForestClassifier

my_model = RandomForestClassifier(n_estimators=30, random_state=1).fit(train_X, train_y)
data.describe()
# create object can calculate SHAP values

import shap

explainer = shap.TreeExplainer(my_model)



# calculate SHAP values

shap_values = explainer.shap_values(small_val_X)



# visualize

shap.summary_plot(shap_values[1], small_val_X)
# set following variable to 'diag_1_428' or 'payer_code_?'

feature_with_bigger_range_of_effects = 'diag_1_428'



# check your answer

q_1.check()
# uncomment the line below to see the solution and explanation

# q_1.solution()
# check your answer (run this code cell to receive credit!)

q_2.solution()
shap.summary_plot(shap_values[1], small_val_X)
# set following var to "diag_1_428" if changing it to 1 has bigger effect.  else set it to 'payer_code_?'

bigger_effect_when_changed = 'diag_1_428'



# check your answer

q_3.check()
# for a solution and explanation, uncomment the line below

q_3.solution()
# check your answer (run this code cell to receive credit!)

q_4.solution()
# check your answer (run this code cell to receive credit!)

q_5.solution()
shap.summary_plot(shap_values[1], small_val_X)
shap.dependence_plot("num_medications", shap_values[1], small_val_X, interaction_index="num_lab_procedures")

shap.dependence_plot("num_lab_procedures", shap_values[1], small_val_X, interaction_index="num_medications")
# check your answer (run this code cell to receive credit!)

q_6.solution()