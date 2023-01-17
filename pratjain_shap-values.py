import numpy as np

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier



data = pd.read_csv('../input/fifa-2018-match-statistics/FIFA 2018 Statistics.csv')

y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary

feature_names = [i for i in data.columns if data[i].dtype in [np.int64, np.int64]]

X = data[feature_names]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
import shap  # package used to calculate the Shap values.



row_to_show = 5

data_for_prediction = val_X.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired

data_for_prediction_array = data_for_prediction.values.reshape(1, -1)



# Create object that can calculate shap values

explainer = shap.TreeExplainer(my_model)



# Calculate Shap values

shap_values = explainer.shap_values(data_for_prediction)
my_model.predict_proba(data_for_prediction_array)
shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
# use Kernel SHAP to explain test set predictions

k_explainer = shap.KernelExplainer(my_model.predict_proba, train_X)

k_shap_values = k_explainer.shap_values(data_for_prediction)

shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
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
#q_1.solution()
from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots



# Create the data that we will plot

pdp_goals = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=base_features, feature='number_inpatient')



# plot it

pdp.pdp_plot(pdp_goals, 'number_inpatient')

plt.show()
#q_2.solution()
from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots



# Create the data that we will plot

pdp_goals = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=base_features, feature='time_in_hospital')



# plot it

pdp.pdp_plot(pdp_goals, 'number_inpatient')

plt.show()
#q_3.solution()
# A simple pandas groupby showing the average readmission rate for each time_in_hospital.

# Do concat to keep validation data separate, rather than using all original data

all_train = pd.concat([train_X, train_y], axis=1)



all_train.groupby(['time_in_hospital']).mean().readmitted.plot()

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



patient_risk_factors(my_model,sample_data_for_prediction)
#q_5.hint()

#q_5.solution()