import sys
sys.path.append('../input/ml-insights-tools')
from ex4 import *
print("Setup Complete")
import pandas as pd
%matplotlib inline
data = pd.read_csv('../input/hospital-readmissions/train.csv')
data.columns
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


data = pd.read_csv('../input/hospital-readmissions/train.csv') # load data

y = data.readmitted # create target df

base_features = [c for c in data.columns if c != "readmitted"] # create list of all features except for readmitted

X = data[base_features] # create features df


train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1) # make the train and test dfs
my_model = RandomForestClassifier(n_estimators=30, random_state=1).fit(train_X, train_y) # initiate the model

# Permutation Importance as a model summary
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y) # initiate the permuation model
eli5.show_weights(perm, feature_names = val_X.columns.tolist()) # show the weights of the features
#q_1.solution()
# Partial Dependence Plot for number_inpatient feature

from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

feature_name = 'number_inpatient'
# Create the data that we will plot
my_pdp = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=val_X.columns, feature=feature_name)

# plot it
pdp.pdp_plot(my_pdp, feature_name)
plt.show()
#q_2.solution()
# Partial Dependence Plot for time_in_hospital feature

from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

feature_name = 'time_in_hospital'
# Create the data that we will plot
my_pdp = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=val_X.columns, feature=feature_name)

# plot it
pdp.pdp_plot(my_pdp, feature_name)
plt.show()
#q_3.solution()
# A simple pandas groupby showing the average readmission rate for each time_in_hospital.

# Do concat to keep validation data separate, rather than using all original data
all_train = pd.concat([train_X, train_y], axis=1) # Merge the target back with the features df

all_train.groupby(['time_in_hospital']).mean().readmitted.plot() # Average readmission rate by time.
plt.show()
#q_4.hint()
#q_4.solution()
# Use SHAP values to show the effect of each feature of a given patient

import shap  # package used to calculate Shap values

sample_data_for_prediction = val_X.iloc[0].astype(float)  # to test function

def patient_risk_factors(model, patient_data):
    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient_data)
    shap.initjs()
    return shap.force_plot(explainer.expected_value[1], shap_values[1], patient_data)


    
    
# Use the function to see the shap values for different patients

sample_data_for_prediction = val_X.iloc[0].astype(float)  # to test function
#sample_data_for_prediction = val_X.iloc[10].astype(float)  # try another row

patient_risk_factors(my_model, sample_data_for_prediction)
#q_5.hint()
#q_5.solution()