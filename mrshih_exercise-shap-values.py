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

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())

# q_1.solution()
# Your Code Here

from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot
feature2plot = 'number_inpatient'
pdp_number_inpatient = pdp.pdp_isolate(model=my_model, dataset=val_X,
                            model_features=val_X.columns,
                            feature=feature2plot)

# plot it
pdp.pdp_plot(pdp_number_inpatient, feature2plot)
plt.show()
# q_2.solution()
# Your Code Here
# Create the data that we will plot
feature2plot = 'time_in_hospital'
pdp_number_inpatient = pdp.pdp_isolate(model=my_model, dataset=val_X,
                            model_features=val_X.columns,
                            feature=feature2plot)

# plot it
pdp.pdp_plot(pdp_number_inpatient, feature2plot)
plt.show()
# q_3.solution()
# Your Code Here

train_df = pd.concat([train_X, train_y], axis=1)
# Group by time_in_hospital
train_df.groupby(['time_in_hospital']).mean().readmitted.plot()
plt.show()
q_4.hint()
# q_4.solution()
# Your Code Here
import shap  # package used to calculate Shap values

# Create function that can calculate and show shap values
def get_SHAP_factors(model, val_X, row):
    data_for_prediction = val_X.iloc[row,:].astype('float')
    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data_for_prediction)
    shap.initjs()
    return shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
# Create function that can calculate and show shap values
get_SHAP_factors(my_model, val_X, 6)
# q_5.hint()
# q_5.solution()