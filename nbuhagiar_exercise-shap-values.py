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

# Your code here
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=0).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())
q_1.solution()
# Your Code Here
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

pdp_goals = pdp.pdp_isolate(model=my_model, 
                            dataset=val_X, 
                            model_features=base_features, 
                            feature='number_inpatient')
ax = pdp.pdp_plot(pdp_goals, 'number_inpatient')
ax[1]["pdp_ax"].set_xlabel("Inpatient Number")
ax[1]["pdp_ax"].set_ylabel("Readmittance Rate Impact")
q_2.solution()
# Your Code Here
pdp_goals = pdp.pdp_isolate(model=my_model, 
                            dataset=val_X, 
                            model_features=base_features, 
                            feature='time_in_hospital')
ax = pdp.pdp_plot(pdp_goals, 'time_in_hospital')
ax[1]["pdp_ax"].set_xlabel("Time Spent in Hospital")
ax[1]["pdp_ax"].set_ylabel("Readmittance Rate Impact")
q_3.solution()
# Your Code Here
train = pd.concat([train_X, train_y], axis=1)
ax = train.groupby("time_in_hospital").mean().readmitted.plot()
ax.set_xlabel("Time Spent in Hospital")
ax.set_ylabel("Readmittance Rate")
ax.set_ylim((0, 1))
q_4.hint()
q_4.solution()
# Your Code Here
import shap

def patient_risk_factors(model, patient_data):
    data_for_prediction = patient_data.astype(float)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data_for_prediction)
    shap.initjs()
    return shap.force_plot(explainer.expected_value[1], 
                           shap_values[1], 
                           data_for_prediction)
patient_risk_factors(my_model, val_X.iloc[0])
q_5.hint()
q_5.solution()