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

# Your code here
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())
from sklearn import metrics
import numpy as np

my_preds = my_model.predict(val_X)
# A random baseline to compare our predictions to. Flip a coin for each patient.
baseline_preds = np.random.rand(len(val_y))

def report_summary_stats(predictions):
    # Round to binary 0/1 predictions
    pred_labels = predictions.round()
    acc = metrics.accuracy_score(val_y, pred_labels)
    pre = metrics.precision_score(val_y, pred_labels)
    rec = metrics.recall_score(val_y, pred_labels)
    fs = metrics.f1_score(val_y, pred_labels)
    print("Accuracy = {:.1%}, Precision = {:.1%}, Recall = {:.1%}, F1 Score: {:.1%}".format(
        acc, pre, rec, fs
    ))
    print("We correctly predicted {:.1%} out of those we predicted to readmit.".format(pre))
    print("Of those who actually readmitted, we correctly predicted {:.1%}.".format(rec))

print("** Our model **")
report_summary_stats(my_preds)
print("** Random baseline **")
report_summary_stats(baseline_preds)
# q_1.solution()
# Your Code Here
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=val_X.columns.tolist(), feature='number_inpatient')

# plot it
pdp.pdp_plot(pdp_goals, 'number_inpatient')
plt.show()
# q_2.solution()
# Your Code Here
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=val_X.columns.tolist(), feature='time_in_hospital')

# plot it
pdp.pdp_plot(pdp_goals, 'time_in_hospital')
plt.show()
# q_3.solution()
# Your Code Here
all_train = pd.concat([train_X, train_y], axis=1)
# all_train.groupby(['time_in_hospital']).mean()
all_train.groupby(['time_in_hospital']).mean().readmitted.plot()
plt.show()
# q_4.hint()
# q_4.solution()
# Your Code Here
import shap  # package used to calculate Shap values

def patient_risk_factors(model, row):
    data_for_prediction = val_X.iloc[row,:]  # use 1 row of data here. Could use multiple rows if desired

    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(data_for_prediction)
    shap.initjs()
    return shap.force_plot(explainer.expected_value[0], shap_values[0], data_for_prediction)

patient_risk_factors(my_model, 3)
# q_5.hint()
q_5.solution()