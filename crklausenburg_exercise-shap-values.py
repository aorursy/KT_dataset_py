from learntools.ml_explainability.ex4 import *

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



perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)

eli5.show_weights(perm, feature_names = val_X.columns.tolist())
q_1.solution()
# Your Code Here

from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots



# Create the data that we will plot

mainf = 'number_inpatient'

pdp_goals = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=base_features, feature=mainf)



# plot it

pdp.pdp_plot(pdp_goals, mainf)

plt.show()
q_2.solution()
# Your Code Here

from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots



# Create the data that we will plot

feat = 'time_in_hospital'

pdp_goals = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=base_features, feature=feat)



# plot it

pdp.pdp_plot(pdp_goals, feat)

plt.show()
q_3.solution()
# Your Code Here

all_train = pd.concat([train_X, train_y], axis=1)



#print(all_train.groupby(['time_in_hospital']).mean())



all_train.groupby(['time_in_hospital']).mean().readmitted.plot()

plt.show()
q_4.hint()

q_4.solution()
# Your Code Here

import shap  # package used to calculate Shap values

#print(my_model.predict_proba(val_X.iloc[1].values.reshape(1,-1)))



def patient_risk_factors(row):

    data_for_prediction = val_X.iloc[row]  # use 1 row of data here. Could use multiple rows if desired



    # Create object that can calculate shap values

    explainer = shap.TreeExplainer(my_model)

    shap_values = explainer.shap_values(data_for_prediction)

    shap.initjs()

    return shap.force_plot(explainer.expected_value[0], shap_values[0], data_for_prediction)



patient_risk_factors(1)
#SHAP bars for positive label, sum of baselines is (perfectly normal) one



data_for_prediction = val_X.iloc[1]



explainer = shap.TreeExplainer(my_model)

shap_values = explainer.shap_values(data_for_prediction)

shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
import numpy as np

data_for_prediction = val_X.iloc[1]



explainer = shap.KernelExplainer(my_model.predict_proba, train_X)

shap_values = explainer.shap_values(data_for_prediction)

shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
q_5.hint()

q_5.solution()