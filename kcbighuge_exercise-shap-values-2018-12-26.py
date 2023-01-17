from learntools.ml_insights.ex4 import *
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

import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())
q_1.solution()
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot
feat = 'number_inpatient'
pdp_goals = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=val_X.columns.tolist(), 
                            feature=feat)

# plot it
pdp.pdp_plot(pdp_goals, feat);
q_2.solution()
# Create the data that we will plot
feat = 'time_in_hospital'
pdp_goals = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=val_X.columns.tolist(), 
                            feature=feat)

# plot it
pdp.pdp_plot(pdp_goals, feat);
q_3.solution()
train_X['time_in_hospital'].value_counts()
train_X['time_in_hospital'].plot('hist');
all_train = pd.concat([train_X, train_y], axis=1)

all_train.groupby(['time_in_hospital']).mean().readmitted.plot();
q_4.solution()
import shap  # package used to calculate Shap values

row_to_predict = 10
data_for_prediction = val_X.iloc[row_to_predict]  # use 1 row of data here. Could use multiple rows if desired
data_for_prediction[:8]
# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)
shap_values = explainer.shap_values(data_for_prediction.astype('float'))
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0], data_for_prediction)
row_to_predict = 100
sample_data_for_prediction = val_X.iloc[row_to_predict].astype(float)  # to test function

def patient_risk_factors(model, patient_data):
    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient_data)
    shap.initjs()
    return shap.force_plot(explainer.expected_value[1], shap_values[1], patient_data)

patient_risk_factors(my_model, sample_data_for_prediction)
q_5.solution()
