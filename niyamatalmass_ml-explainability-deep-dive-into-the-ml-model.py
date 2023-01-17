import numpy as np

import pandas as pd



from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score



import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
df_cancer = pd.read_csv('../input/kag_risk_factors_cervical_cancer.csv')

df_cancer = df_cancer.replace('?', np.nan)

# df_cancer = df_cancer.drop(DROP_COLUMNS, axis = 1)

df_cancer = df_cancer.rename(columns={'Biopsy': 'Cancer'})

df_cancer = df_cancer.apply(pd.to_numeric)

df_cancer = df_cancer.fillna(df_cancer.mean().to_dict())



X = df_cancer.drop('Cancer', axis=1)

y = df_cancer['Cancer']



df_cancer.head()
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size = 0.25,random_state=2019)

random_forest = RandomForestClassifier(n_estimators=500, random_state=2019).fit(X_train, y_train)

y_pred = random_forest.predict(X_test)

print('Accuracy score: ' + str(accuracy_score(y_test, y_pred)))

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X[['IUD']], y, shuffle=True, test_size = 0.25,random_state=2019)

random_forest = RandomForestClassifier(n_estimators=500, random_state=2019).fit(X_train_2, y_train_2)

y_pred_2 = random_forest.predict(X_test_2)

print('Accuracy score: ' + str(accuracy_score(y_test_2, y_pred_2)))
from sklearn.metrics import recall_score

from sklearn.metrics import f1_score



recall_score(y_test, y_pred)
from imblearn.combine import SMOTETomek





X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, test_size = 0.25,random_state=2019)



cc = SMOTETomek(random_state=2019)

X_res, y_res = cc.fit_resample(X_train, y_train)



random_forest = RandomForestClassifier(n_estimators=500, random_state=2019).fit(X_res, y_res)

y_pred = random_forest.predict(X_test)

print('Accuracy score: ' + str(accuracy_score(y_test, y_pred)))

print('Recall score: ' + str(recall_score(y_test, y_pred)))
import eli5

from eli5.sklearn import PermutationImportance



permumtation_impor = PermutationImportance(random_forest, random_state=2019).fit(X_test, y_test)

eli5.show_weights(permumtation_impor, feature_names = X_test.columns.tolist())
from pdpbox import pdp, get_dataset, info_plots



def pdpplot( feature_to_plot, pdp_model = random_forest, pdp_dataset = X_test, pdp_model_features = list(X)):

    pdp_cancer = pdp.pdp_isolate(model=pdp_model, dataset=pdp_dataset, model_features=pdp_model_features, feature=feature_to_plot)

    fig, axes = pdp.pdp_plot(pdp_cancer, feature_to_plot, figsize = (10, 5),plot_params={})

#     _ = axes['pdp_ax'].set_ylabel('Probability of Cancer')

    

pdpplot('Schiller')

pdpplot('First sexual intercourse')

pdpplot('Num of pregnancies')



plt.show()
row_to_show = 10

data_for_prediction = X_test.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired

data_for_prediction_array = data_for_prediction.values.reshape(1, -1)





random_forest.predict_proba(data_for_prediction_array)
import shap  # package used to calculate Shap values



# Create object that can calculate shap values

explainer = shap.TreeExplainer(random_forest)



# Calculate Shap values

shap_values = explainer.shap_values(data_for_prediction)
shap.initjs()

shap.force_plot(explainer.expected_value[0], shap_values[0], data_for_prediction)
row_to_show = 94

data_for_prediction = X_test.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired

data_for_prediction_array = data_for_prediction.values.reshape(1, -1)





random_forest.predict_proba(data_for_prediction_array)
# Create object that can calculate shap values

explainer = shap.TreeExplainer(random_forest)



# Calculate Shap values

shap_values = explainer.shap_values(data_for_prediction)
shap.initjs()

shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)