from learntools.ml_explainability.ex4 import *

print("Setup Complete")
import pandas as pd

data = pd.read_csv('../input/hospital-readmissions/train.csv')

data.columns

data.head()
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



#data.head()
from matplotlib import pyplot as plt

%matplotlib inline

from pdpbox import pdp, get_dataset, info_plots



impact_features = ['number_inpatient', 'number_emergency', 'number_outpatient']

for feat_name in impact_features:

    # Create the data that we will plot

    pdp_goals = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=val_X.columns.tolist(), feature=feat_name)

    # plot it

    pdp.pdp_plot(pdp_goals, feat_name)

    plt.show()



inter1  =  pdp.pdp_interact(model=my_model, dataset=val_X, model_features=val_X.columns.tolist(), features=['number_inpatient', 'number_emergency'])



pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=['number_inpatient', 'number_emergency'], plot_type='contour')

plt.show()
import shap  # package used to calculate Shap values



data_for_prediction = val_X.iloc[0,:]  # use 1 row of data here. Could use multiple rows if desired



# Create object that can calculate shap values

explainer = shap.TreeExplainer(my_model)

shap_values = explainer.shap_values(data_for_prediction)

shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction, matplotlib=True)
shap_values = explainer.shap_values(val_X.iloc[0:10,:])

shap.summary_plot(shap_values, val_X, plot_type='bar')
 #q_1.solution()
feat_name = 'number_inpatient'

pdp_goals = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=val_X.columns.tolist(), feature=feat_name)

pdp.pdp_plot(pdp_goals, feat_name)

plt.show()
 #q_2.solution()
feat_name = 'time_in_hospital'

pdp_goals = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=val_X.columns.tolist(), feature=feat_name)

pdp.pdp_plot(pdp_goals, feat_name)

plt.show()
# q_3.solution()
import seaborn as sns



all_train = pd.concat([train_X, train_y], axis=1)



all_train = all_train.groupby(['time_in_hospital']).mean()

sns.relplot(x=all_train.index, y="readmitted", kind="line", aspect=2, data=all_train)
# q_4.hint()

# q_4.solution()
import shap



sample_data_for_prediction = val_X.iloc[0].astype(float)



def patient_risk_factors(model, data_for_prediction):

    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(data_for_prediction)

    shap.initjs()

    return shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction, matplotlib=True)



patient_risk_factors(my_model, sample_data_for_prediction)
#q_5.hint()

#q_5.solution()