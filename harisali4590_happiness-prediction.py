# importing required modules



import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

from sklearn import model_selection

import statsmodels.formula.api as sm

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from IPython.core.display import display, HTML

from pdpbox import pdp, get_dataset, info_plots

import shap



import eli5

from eli5.sklearn import PermutationImportance

dataset = pd.read_csv("../input/dataset.csv")

# happiness scores

y = dataset.iloc[:, 5].values



# independent variables

X = dataset.iloc[:, :-1].values



# feature names

feature_names = ['Economy (GDP per Capita)', 'Trust (Government Corruption)', 'HDI', 'Inflation', 'Employment Ratio']



X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state = 1)
# Using Permutation importance to find the most important variables

# Using a random forest regression for this

from sklearn.ensemble import RandomForestRegressor

randomForestRegressor = RandomForestRegressor(n_estimators = 300, random_state = 0)

y = y.ravel()

randomForestModel = randomForestRegressor.fit(X_train, y_train)



perm = PermutationImportance(randomForestModel, random_state = 1).fit(X_test, y_test)

display(HTML(eli5.show_weights(perm, feature_names = feature_names).data))
# plotting partial dependency plot for HDI

pdp_goals = pdp.pdp_isolate(model=randomForestModel, dataset=dataset, model_features=feature_names, feature='HDI')

pdp.pdp_plot(pdp_goals, 'HDI')

plt.show()
# plotting partial dependency plot for Employment Ratio

pdp_goals = pdp.pdp_isolate(model=randomForestModel, dataset=dataset, model_features=feature_names, feature='Employment Ratio')

pdp.pdp_plot(pdp_goals, 'Employment Ratio')

plt.show()
# plotting partial dependency plot for Inflation

pdp_goals = pdp.pdp_isolate(model=randomForestModel, dataset=dataset, model_features=feature_names, feature='Inflation')

pdp.pdp_plot(pdp_goals, 'Inflation')

plt.show()


prediction_data = X[38]  # Using 1 row here, having high value of happiness score

explainer = shap.TreeExplainer(randomForestModel)

shap_values = explainer.shap_values(prediction_data)

shap.initjs()

shap.force_plot(explainer.expected_value, shap_values[0:], prediction_data, feature_names = feature_names)


prediction_data = X[75]  # Using 1 row here, having moderate value of happiness score

explainer = shap.TreeExplainer(randomForestModel)

shap_values = explainer.shap_values(prediction_data)

shap.initjs()

shap.force_plot(explainer.expected_value, shap_values[0:], prediction_data, feature_names = feature_names)
prediction_data = X[90]  # Using 1 row here, having low value of happiness score

explainer = shap.TreeExplainer(randomForestModel)

shap_values = explainer.shap_values(prediction_data)

shap.initjs()

shap.force_plot(explainer.expected_value, shap_values[0:], prediction_data, feature_names = feature_names)
prediction_data = X[1:100]  # Using 100 rows here

explainer = shap.TreeExplainer(randomForestModel)

shap_values = explainer.shap_values(prediction_data)

shap.initjs()

shap.force_plot(explainer.expected_value, shap_values[0:], prediction_data, feature_names = feature_names)
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names = feature_names)
# make a dependence plot of Employment Ratio

shap_values = explainer.shap_values(X)

shap.dependence_plot('Employment Ratio', shap_values, X, feature_names = feature_names)