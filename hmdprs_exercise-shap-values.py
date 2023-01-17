from learntools.ml_explainability.ex4 import *

print("Setup is completed.")
import pandas as pd

data = pd.read_csv('../input/hospital-readmissions/train.csv')

data.columns
import pandas as pd

data = pd.read_csv('../input/hospital-readmissions/train.csv')



y = data["readmitted"]



base_features = [c for c in data.columns if c != "readmitted"]

X = data[base_features]



from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



from sklearn.ensemble import RandomForestClassifier

my_model = RandomForestClassifier(n_estimators=30, random_state=1).fit(train_X, train_y)
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)



from eli5 import show_weights

show_weights(perm, feature_names = val_X.columns.tolist())
# run this code cell to receive credit!

q_1.solution()
# create the data that we will plot

from pdpbox import pdp, get_dataset, info_plots

pdp_goals = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=base_features, feature='number_inpatient')



# plot it

pdp.pdp_plot(pdp_goals, 'number_inpatient')

from matplotlib import pyplot as plt

plt.show()
# check your answer (run this code cell to receive credit!)

q_2.solution()
# create the data that we will plot

from pdpbox import pdp, get_dataset, info_plots

pdp_goals = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=base_features, feature='time_in_hospital')



# plot it

pdp.pdp_plot(pdp_goals, 'time_in_hospital')

from matplotlib import pyplot as plt

plt.show()
# check your answer (run this code cell to receive credit!)

q_3.solution()
all_train = pd.concat([train_X, train_y], axis=1)

all_train.groupby("time_in_hospital")["readmitted"].mean().plot()
# q_4.hint()
# check your answer (run this code cell to receive credit!)

q_4.solution()
# function shows a patient's risk factors

def patient_risk_factors(model, data_for_prediction):

    # create object that can calculate shap values

    import shap

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(data_for_prediction)



    # plot

    shap.initjs()

    return shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
sample_data_for_prediction = val_X.iloc[0]

patient_risk_factors(my_model, sample_data_for_prediction)
# q_5.hint()
# check your answer (run this code cell to receive credit!)

q_5.solution()