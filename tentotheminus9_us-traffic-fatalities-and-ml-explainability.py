import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from google.cloud import bigquery #For BigQuery

from bq_helper import BigQueryHelper #For BigQuery

from sklearn.ensemble import RandomForestClassifier #for the model

from sklearn.metrics import roc_curve, auc #for model evaluation

from sklearn.metrics import classification_report #for model evaluation

from sklearn.metrics import confusion_matrix #for model evaluation

from sklearn.model_selection import train_test_split #for data splitting

import eli5 #for purmutation importance

from eli5.sklearn import PermutationImportance

import shap #for SHAP values

from pdpbox import pdp, info_plots #for partial plots

np.random.seed(123) #ensure reproducibility



pd.options.mode.chained_assignment = None  #hide any pandas warnings
us_traffic = BigQueryHelper("bigquery-public-data", "nhtsa_traffic_fatalities")

us_traffic.head("accident_2015")
accidents_query_2015 = """SELECT month_of_crash,

                                 day_of_week,

                                 hour_of_crash,

                                 manner_of_collision_name,

                                 light_condition_name,

                                 land_use_name,

                                 latitude,

                                 longitude,

                                 atmospheric_conditions_1_name,

                                 number_of_drunk_drivers,

                                 number_of_fatalities

                          FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

                          WHERE number_of_fatalities = 1

                          AND longitude < 0

                          AND longitude > -140

                          LIMIT 5000

                      """ 



accidents_query_2016 = """SELECT month_of_crash,

                                 day_of_week,

                                 hour_of_crash,

                                 manner_of_collision_name,

                                 light_condition_name,

                                 land_use_name,

                                 latitude,

                                 longitude,

                                 atmospheric_conditions_1_name,

                                 number_of_drunk_drivers,

                                 number_of_fatalities

                          FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`

                          WHERE number_of_fatalities = 1

                          AND longitude < 0

                          AND longitude > -140

                          LIMIT 5000

                      """ 



accidents_query_multiple_2015 = """SELECT month_of_crash,

                                          day_of_week,

                                          hour_of_crash,

                                          manner_of_collision_name,

                                          light_condition_name,

                                          land_use_name,

                                          latitude,

                                          longitude,

                                          atmospheric_conditions_1_name,

                                          number_of_drunk_drivers,

                                          number_of_fatalities

                                    FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2015`

                                    WHERE number_of_fatalities > 1

                                    AND longitude < 0

                                    AND longitude > -140

                      """ 



accidents_query_multiple_2016 = """SELECT month_of_crash,

                                          day_of_week,

                                          hour_of_crash,

                                          manner_of_collision_name,

                                          light_condition_name,

                                          land_use_name,

                                          latitude,

                                          longitude,

                                          atmospheric_conditions_1_name,

                                          number_of_drunk_drivers,

                                          number_of_fatalities

                                    FROM `bigquery-public-data.nhtsa_traffic_fatalities.accident_2016`

                                    WHERE number_of_fatalities > 1

                                    AND longitude < 0

                                    AND longitude > -140

                      """ 
accidents_2015 = us_traffic.query_to_pandas(accidents_query_2015)

accidents_2015_multiple = us_traffic.query_to_pandas(accidents_query_multiple_2015)



accidents_2016 = us_traffic.query_to_pandas(accidents_query_2016)

accidents_2016_multiple = us_traffic.query_to_pandas(accidents_query_multiple_2016)



frames = [accidents_2015, accidents_2015_multiple, accidents_2016, accidents_2016_multiple]

accidents_all = pd.concat(frames)
accidents_all['number_of_fatalities'].hist()
accidents_all['drunk_driver_involved'] = 0

accidents_all['drunk_driver_involved'][accidents_all['number_of_drunk_drivers'] > 1] = 1

accidents_all = accidents_all.drop('number_of_drunk_drivers', 1)



accidents_all['Multiple_fatalities'] = 0

accidents_all['Multiple_fatalities'][accidents_all['number_of_fatalities'] > 1] = 1

accidents_all = accidents_all.drop('number_of_fatalities', 1)
accidents_all.groupby(['land_use_name']).size()
accidents_all = accidents_all[accidents_all['hour_of_crash'] != 99]

accidents_all = accidents_all[accidents_all['manner_of_collision_name'] != 'Unknown']

accidents_all = accidents_all[accidents_all['light_condition_name'] != 'Unknown']

accidents_all = accidents_all[accidents_all['atmospheric_conditions_1_name'] != 'Unknown']

accidents_all = accidents_all[accidents_all['land_use_name'] != 'Unknown']
accidents_all = accidents_all[accidents_all['land_use_name'] != 'Trafficway Not in State Inventory']
x = accidents_all['longitude']

y = accidents_all['latitude']



plt.plot(x, y)
accidents_all.head(10)
accidents_all['month_of_crash'] = accidents_all['month_of_crash'].astype('category')

accidents_all['day_of_week'] = accidents_all['day_of_week'].astype('category')

accidents_all['hour_of_crash'] = accidents_all['hour_of_crash'].astype('category')

accidents_all['drunk_driver_involved'] = accidents_all['drunk_driver_involved'].astype('category')



accidents_all.dtypes
accidents_all = pd.get_dummies(accidents_all, drop_first=True) #from the reduntant dummy categories
accidents_all.head()
accidents_all.shape
X_train, X_test, y_train, y_test = train_test_split(accidents_all.drop('Multiple_fatalities', 1), accidents_all['Multiple_fatalities'], test_size = .3, random_state=25)
model = RandomForestClassifier()

model.fit(X_train, y_train)
y_predict = model.predict(X_test)

y_pred_quant = model.predict_proba(X_test)[:, 1]

y_pred_bin = model.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred_bin)

confusion_matrix
total=sum(sum(confusion_matrix))



sensitivity = confusion_matrix[0,0]/(confusion_matrix[0,0]+confusion_matrix[0,1])

print('Sensitivity : ', sensitivity )



specificity = confusion_matrix[1,1]/(confusion_matrix[1,0]+confusion_matrix[1,1])

print('Specificity : ', specificity)
fpr, tpr, thresholds = roc_curve(y_test, y_pred_quant)



fig, ax = plt.subplots()

ax.plot(fpr, tpr)

ax.plot([0, 1], [0, 1], transform=ax.transAxes, ls="--", c=".3")

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.0])

plt.rcParams['font.size'] = 12

plt.title('ROC curve for diabetes classifier')

plt.xlabel('False Positive Rate (1 - Specificity)')

plt.ylabel('True Positive Rate (Sensitivity)')

plt.grid(True)
auc(fpr, tpr)
perm = PermutationImportance(model, random_state=1).fit(X_test, y_test)

eli5.show_weights(perm, feature_names = X_test.columns.tolist())
X_test_sample = X_test.iloc[:200]



base_features = accidents_all.columns.values.tolist()

base_features.remove('Multiple_fatalities')



feat_name = 'manner_of_collision_name_Front-to-Front'

pdp_dist = pdp.pdp_isolate(model=model, dataset=X_test_sample, model_features=base_features, feature=feat_name)



pdp.pdp_plot(pdp_dist, feat_name)

plt.show()
inter1  =  pdp.pdp_interact(model=model, dataset=X_test_sample, model_features=base_features, features=['land_use_name_Urban', 'manner_of_collision_name_Front-to-Front'])



pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=['land_use_name_Urban', 'manner_of_collision_name_Front-to-Front'], plot_type='contour')

plt.show()
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test_sample)



shap.summary_plot(shap_values[1], X_test_sample, plot_type="bar")
shap.summary_plot(shap_values[1], X_test_sample)
def accident_risk_factors(model, accident):



    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(accident)

    shap.initjs()

    return shap.force_plot(explainer.expected_value[1], shap_values[1], accident)
data_for_prediction = X_test.iloc[1,:].astype(float)

accident_risk_factors(model, data_for_prediction)
data_for_prediction = X_test.iloc[5,:].astype(float)

accident_risk_factors(model, data_for_prediction)
explainer = shap.TreeExplainer(model)

shap_values = explainer.shap_values(X_test_sample)



shap.dependence_plot('latitude', shap_values[1], X_test_sample, interaction_index="manner_of_collision_name_Front-to-Front")
shap_values = explainer.shap_values(X_train.iloc[:100])

shap.force_plot(explainer.expected_value[1], shap_values[1], X_train.iloc[:100])