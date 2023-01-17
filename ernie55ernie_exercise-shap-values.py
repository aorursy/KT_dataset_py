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



base_features = [c for c in data.columns if c != "T"]



X = data[base_features]





train_X, val_X, train_y, val_y = train_test_split(X, y, random_state =1)

my_model = RandomForestClassifier(n_estimators = 30, random_state = 1).fit(train_X, train_y)

# Your code here

import eli5

from eli5.sklearn import PermutationImportance



perm = PermutationImportance(my_model, random_state = 1).fit(val_X, val_y)

eli5.show_weights(perm, feature_names = base_features)
q_1.solution()
# Your Code Here

import matplotlib.pyplot as plt

from pdpbox import pdp



pdp_inpatient = pdp.pdp_isolate(model = my_model, dataset = val_X, model_features = base_features, feature = 'number_inpatient')



pdp.pdp_plot(pdp_inpatient, 'Number of inpatient')

plt.show()
q_2.solution()
# Your Code Here

pdp_time = pdp.pdp_isolate(model = my_model, dataset = val_X, model_features = base_features, feature = 'time_in_hospital')



pdp.pdp_plot(pdp_time, 'time_in_hospital')

plt.show()



features_to_plot = ['number_inpatient', 'time_in_hospital']



inter = pdp.pdp_interact(model = my_model, dataset = val_X, model_features = base_features, features = features_to_plot)



pdp.pdp_interact_plot(inter, feature_names = features_to_plot, plot_type = 'contour')

plt.show()
q_3.solution()
# Your Code Here

data.groupby('time_in_hospital').mean()['readmitted'].plot()

plt.show()
q_4.hint()

q_4.solution()
# Your Code Here

import shap



def patient_risk_factors(row):

    explainer = shap.TreeExplainer(my_model)

    shap_values = explainer.shap_values(row)

    shap.initjs()

    return shap.force_plot(explainer.expected_value[1], shap_values[1], row)

    

sample = val_X.iloc[0].astype(float)

patient_risk_factors(sample)
q_5.hint()

q_5.solution()