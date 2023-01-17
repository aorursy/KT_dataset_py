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

import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())
q_1.solution()
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

feature_names=val_X.columns.tolist()

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=feature_names, feature='number_inpatient')

# plot it
pdp.pdp_plot(pdp_goals, 'number_inpatient')
plt.show()
q_2.solution()
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

feature_names=val_X.columns.tolist()

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=feature_names, feature='time_in_hospital')

# plot it
pdp.pdp_plot(pdp_goals, 'time_in_hospital')
plt.show()
q_3.solution()
print(data.loc[val_X.index, :].groupby('time_in_hospital')['readmitted'].mean())

data.loc[val_X.index, :].groupby(['time_in_hospital']).mean().readmitted.plot()
q_4.hint()
q_4.solution()
import shap  # package used to calculate Shap values
import numpy as np
from IPython.display import display

def patient_risk_factors(x):
    # Create object that can calculate shap values
    x1 = np.array(x, dtype='float64').reshape((1, -1))
    explainer = shap.TreeExplainer(my_model)
    shap_values = explainer.shap_values(x1)
    shap.initjs()
    display(shap.force_plot(explainer.expected_value[0], shap_values[0], x))

patient_risk_factors(val_X.iloc[0, :])
# q_5.hint()
q_5.solution()