from numpy import random
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import shap


sample_size = 1000
X1 = 2*random.rand(sample_size)
X2 = 2*random.rand(sample_size)
X3 = random.rand(sample_size)
y = X1**1.5 + 1.7 * X1 - 2 * X1 * X2 + 3 * X2

q4_data = pd.DataFrame({'feature_of_interest': X1,
                        'other_feature': X2,
                        'y': y})


q4_X = q4_data.iloc[:,:2]
first_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(q4_X, q4_data.y)

n_to_plot = 200

explainer = shap.TreeExplainer(first_model)
q4_shap_values = explainer.shap_values(q4_X)
shap.summary_plot(q4_shap_values, q4_X)

shap.dependence_plot("feature_of_interest", q4_shap_values, q4_X, interaction_index="other_feature")

second_model = LinearRegression().fit(q4_X, q4_data.y)
n_to_plot = 200

explainer = shap.KernelExplainer(second_model.predict, q4_X)
second_shap_values = explainer.shap_values(q4_X, nsamples=50)
shap.summary_plot(second_shap_values, q4_X)

shap.dependence_plot("feature_of_interest", second_shap_values, q4_X, interaction_index="other_feature")

stump_model = RandomForestRegressor(n_estimators=500, max_depth=1, random_state=1).fit(q4_X, q4_data.y)

explainer = shap.TreeExplainer(stump_model)
stump_shap_values = explainer.shap_values(q4_X)
shap.summary_plot(stump_shap_values, q4_X)

shap.dependence_plot("feature_of_interest", stump_shap_values, q4_X, interaction_index="other_feature")

q4_data.plot.scatter('feature_of_interest', 'y')
q4_data.plot.scatter('other_feature', 'y')