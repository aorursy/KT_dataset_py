import sys
sys.path.append('../input/ml-insights-tools')
from ex4 import *
print("Setup Complete")
import pandas as pd
data = pd.read_csv('../input/hospital-readmissions/train.csv')
data.columns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


data = pd.read_csv('../input/hospital-readmissions/train.csv')

y = data.readmitted

base_features = [c for c in data.columns if c != "readmitted"]

X = data[base_features]


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(n_estimators=30, random_state=1).fit(train_X, train_y)

data.head()
# Your code here
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names=val_X.columns.tolist())
# q_1.solution()
# Your Code Here

# PDP for number_inpatient feature

from pdpbox import pdp

feature_names = val_X.columns.tolist()
pdp_number_inpatient = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=feature_names, feature='number_inpatient')
pdp.pdp_plot(pdp_number_inpatient, 'number_inpatient')
#q_2.solution()
# Your Code Here
# PDP for number_inpatient feature

from pdpbox import pdp

feature_names = val_X.columns.tolist()
pdp_time_in_hospital = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=feature_names, feature='time_in_hospital')
pdp.pdp_plot(pdp_time_in_hospital, 'time_in_hospital')
#q_3.solution()
# Your Code Here
# A simple pandas groupby showing the average readmission rate for each time_in_hospital.

# Do concat to keep validation data separate, rather than using all original data
all_train = pd.concat([train_X, train_y], axis=1)

all_train.groupby(['time_in_hospital']).mean().readmitted.plot()
q_4.hint()
#q_4.solution()
# Your Code Here
# Use SHAP values to show the effect of each feature of a given patient

import shap  # package used to calculate Shap values

sample_data_for_prediction = val_X.iloc[0].astype(float)  # to test function

def patient_risk_factors(model, patient_data):
    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient_data)
    shap.initjs()
    return shap.force_plot(explainer.expected_value[0], shap_values[0], patient_data)
patient_risk_factors(my_model, sample_data_for_prediction)
q_5.hint()
# q_5.solution()