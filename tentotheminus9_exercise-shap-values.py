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

import eli5

from eli5.sklearn import PermutationImportance



features = val_X.columns.tolist()



perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)

eli5.show_weights(perm, feature_names = features)
# q_1.solution()
from matplotlib import pyplot as plt

from pdpbox import pdp, get_dataset, info_plots



# Create the data that we will plot

pdp_inpatient = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=features, feature='number_inpatient')



# plot it

pdp.pdp_plot(pdp_inpatient, 'number_inpatient')

plt.show()
# q_2.solution()
# Create the data that we will plot

pdp_time = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=features, feature='time_in_hospital')



# plot it

pdp.pdp_plot(pdp_time, 'time_in_hospital')

plt.show()
# q_3.solution()
data_bytime = data.groupby(by=['time_in_hospital']).mean().readmitted.plot()

#data_bytime['readmitted']
# q_4.hint()

# q_4.solution()
import shap  # package used to calculate Shap values



def patient_risk_factors(model, patient):



    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(patient)

    shap.initjs()

    return shap.force_plot(explainer.expected_value[0], shap_values[0], patient)



    

    

data_for_prediction = data.iloc[0,:].astype(float)

patient_risk_factors(my_model, data_for_prediction)
# q_5.hint()

# q_5.solution()