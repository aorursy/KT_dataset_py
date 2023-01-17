import numpy as np # linear algebra
import pandas as pd # data processing, 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('../input/fifa-2018-match-statistics/FIFA 2018 Statistics.csv')
y = (df['Man of the Match'] == 'Yes') # convert from string "Yes/No" to binary
feature_names = [i for i in df.columns if df[i].dtype in [np.int64]]
X = df[feature_names]
train_X, valid_X, train_y, valid_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)

# Get feature importance
import eli5
from eli5.sklearn import PermutationImportance

permutation = PermutationImportance(my_model, random_state=1).fit(valid_X, valid_y)
eli5.show_weights(permutation, feature_names = feature_names)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
# Load data
df_taxi = pd.read_csv('../input/new-york-city-taxi-fare-prediction/train.csv', nrows=50000)
# Remove data with extreme outlier coordinates ot negative fares
df_taxi = df_taxi.query('pickup_latitude > 40.7 and pickup_latitude < 40.8 and ' +
                      'dropoff_latitude > 40.7 and dropoff_latitude < 40.8 and ' +
                      'pickup_longitude > -74 and pickup_longitude < -73.9 and ' +
                      'dropoff_longitude > -74 and dropoff_longitude < -73.9 and ' +
                      'fare_amount > 0'
                       )
y = df_taxi.fare_amount
# Model 
base_features = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 
                 'dropoff_latitude', 'passenger_count']
X = df_taxi[base_features]

train_taxi_X, valid_taxi_X, train_taxi_y, valid_taxi_y = train_test_split(X, y, random_state=1)
first_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(train_taxi_X, train_taxi_y)
train_taxi_X.describe()
train_taxi_y.describe()
# Permutation performance
perm = PermutationImportance(first_model, random_state=1).fit(valid_taxi_X, valid_taxi_y)
eli5.show_weights(perm, feature_names=valid_taxi_X.columns.tolist())
# Create new features
df_taxi['abs_lon_change'] = abs(df_taxi.dropoff_longitude - df_taxi.pickup_longitude)
df_taxi['abs_lat_change'] = abs(df_taxi.dropoff_latitude - df_taxi.pickup_latitude)

# Add the new featues to the base features
features_2  = ['pickup_longitude',
               'pickup_latitude',
               'dropoff_longitude',
               'dropoff_latitude',
               'abs_lat_change',
               'abs_lon_change']

X = df_taxi[features_2]
new_train_taxi_X, new_valid_taxi_X, new_train_taxi_y, new_valid_taxi_y = train_test_split(X, y, random_state=1)
second_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(new_train_taxi_X, new_train_taxi_y)
# Create a PermutationImportance object on second_model and fit it with the new valid data
perm_2 = PermutationImportance(second_model, random_state=1).fit(new_valid_taxi_X, new_valid_taxi_y)
# Show the weights for the permutation importance 
eli5.show_weights(perm_2, feature_names=features_2)
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import graphviz

# Create tree_based model
tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(train_X, train_y)
tree_graph = tree.export_graphviz(tree_model, out_file=None, feature_names=feature_names)
# Visualize tree structure
graphviz.Source(tree_graph)
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# Create the dataset for plotting
pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=valid_X, model_features=feature_names, feature='Goal Scored')

# Plot it
pdp.pdp_plot(pdp_goals, 'Goal Scored')
plt.show()
# Another example plot
feature_to_plot = 'Distance Covered (Kms)'
pdp_dist = pdp.pdp_isolate(model=tree_model, dataset=valid_X, model_features=feature_names, feature=feature_to_plot)
pdp.pdp_plot(pdp_dist, feature_to_plot)
plt.show()
# Build Random Forest model
rf_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)

pdp_dist_2 = pdp.pdp_isolate(model=rf_model, dataset=valid_X,
                            model_features=feature_names, feature=feature_to_plot)
pdp.pdp_plot(pdp_dist_2, feature_to_plot)
plt.show()
# Use pdp_interact and pdp_interact_plot instead of pdp_isolate and pdp_isolate_plot, respectively
features_to_plot = ['Goal Scored', 'Distance Covered (Kms)']
inter_1 = pdp.pdp_interact(model=tree_model, dataset=valid_X, 
                           model_features=feature_names, features=features_to_plot)
pdp.pdp_interact_plot(pdp_interact_out=inter_1, feature_names=features_to_plot, plot_type='contour')
plt.show()
# Partial dependece plot for pickup_longitude
feat_name = 'pickup_longitude'
pdp_dist = pdp.pdp_isolate(model=first_model, dataset=valid_taxi_X, model_features=base_features, feature=feat_name)
pdp.pdp_plot(pdp_dist, feat_name)
plt.show()
from plotly import tools
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
# Create all partial plots for NYC taxi-fare
def plot_pdp():
#     fig, axes = plt.subplots(nrows=3, ncols=2, 
#                              figsize=(13, 16))
    
    for feat_name in base_features:
        pdp_dist = pdp.pdp_isolate(model=first_model, dataset=valid_taxi_X, 
                                  model_features=base_features, feature=feat_name,n_jobs=3)
        pdp.pdp_plot(pdp_dist, feat_name)
        
#     plt.subplots_adjust(top=0.9)
#     plt.show()
    return None
plot_pdp()

# 2D partial plot for NYC taxi fare
fnames = ['pickup_longitude', 'dropoff_longitude']
longitude_pdp = pdp.pdp_interact(model = first_model, dataset=valid_taxi_X,
                                model_features = base_features, features = fnames)
pdp.pdp_interact_plot(pdp_interact_out=longitude_pdp, feature_names=fnames, plot_type='contour')
plt.show()
# PDP for pickup_longitude without absolute difference features
feat_name = 'pickup_longitude'
pdp_dist_original = pdp.pdp_isolate(model=first_model, dataset=valid_taxi_X, 
                                    model_features=base_features, feature=feat_name)
pdp.pdp_plot(pdp_dist_original, feat_name)
plt.show()

feat_name = 'pickup_longitude'
pdp_dist = pdp.pdp_isolate(model=second_model, dataset=new_valid_taxi_X, model_features=features_2, feature=feat_name)
pdp.pdp_plot(pdp_dist, feat_name)
plt.show()
from numpy.random import rand
n_sample = 20000

#  Creates two features, `X1` and `X2`, having random values in the range [-2, 2].
X1 = 4 * rand(n_sample) - 2
X2 = 4 * rand(n_sample) - 2

# Creates a target variable `y`, which is always 1.
y = -2 * X1 * (X1<-1) + X1 -2 * X1 * (X1 > 1) - X2
# Trains a `RandomForestRegressor` model to predict `y` given `X1` and `X2`
my_df = pd.DataFrame({'X1':X1, 'X2':X2, 'y':y})
predictors_df = my_df.drop(['y'], axis=1)
my_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(predictors_df, my_df.y)
# Creates a PDP plot for `X1` and a scatter plot of `X1` vs. `y`
pdp_dist = pdp.pdp_isolate(model=my_model, dataset=my_df, model_features=['X1', 'X2'], feature='X1')
# Visualize results
pdp.pdp_plot(pdp_dist, 'X1')
plt.show()
# Create array holding predictive feature
X1 = 4 * rand(n_sample) - 2
X2 = 4 * rand(n_sample) - 2

# Create y
y =  X1 * X2
# create dataframe because pdp_isolate expects a dataFrame as an argument
my_df = pd.DataFrame({'X1': X1, 'X2': X2, 'y': y})
predictors_df = my_df.drop(['y'], axis=1)

my_model = RandomForestRegressor(n_estimators=30, random_state=1).fit(predictors_df, my_df.y)

pdp_dist = pdp.pdp_isolate(model=my_model, dataset=my_df, model_features=['X1', 'X2'], feature='X1')
pdp.pdp_plot(pdp_dist, 'X1')
plt.show()

perm = PermutationImportance(my_model).fit(predictors_df, my_df.y)
# show the weights for the permutation importance you just calculated
eli5.show_weights(perm, feature_names = ['X1', 'X2'])
df = pd.read_csv('../input/fifa-2018-match-statistics/FIFA 2018 Statistics.csv')
y = (df['Man of the Match'] == 'Yes') # convert from string "Yes/No" to binary
feature_names = [i for i in df.columns if df[i].dtype in [np.int64]]
X = df[feature_names]
X_fifa = X
train_X, valid_X, train_y, valid_y = train_test_split(X, y, random_state=1)
my_model_fifa = RandomForestClassifier(random_state=0).fit(train_X, train_y)
# Package used to calculate Shap values
import shap
row_to_show = 5
data_for_prediction = valid_X.iloc[row_to_show]    # use 1 row of data 
data_for_predicition_array = data_for_prediction.values.reshape(1, -1)

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model_fifa)

# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)

my_model_fifa.predict_proba(data_for_predicition_array)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
# Example using KernelExplainer 
k_explainer = shap.KernelExplainer(my_model_fifa.predict_proba, train_X)
k_shap_values = k_explainer.shap_values(data_for_prediction)
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
hosp_re_data = pd.read_csv('../input/hospital-readmissions/train.csv')
y = hosp_re_data.readmitted
base_features_hosp = [c for c in hosp_re_data.columns if c != 'readmitted']
# Split data into training and validation set
X = hosp_re_data[base_features_hosp]
train_X_hosp, valid_X_hosp, train_y_hosp, valid_y_hosp = train_test_split(X, y, random_state=1)
# Create model 
model_hosp = RandomForestClassifier(n_estimators=30, random_state=1).fit(train_X_hosp, train_y_hosp)
# Prepate the condensed exhibits for the doctor
# Use permutation importance as a suucinct model summary
perm = PermutationImportance(model_hosp, random_state=1).fit(valid_X_hosp, valid_y_hosp)
eli5.show_weights(perm, feature_names = valid_X_hosp.columns.tolist())
# Using PDP for number_inpatient feature
feature_name = 'number_inpatient'
# Create the data for ploting
my_pdp = pdp.pdp_isolate(model=model_hosp, dataset=valid_X_hosp, model_features=valid_X_hosp.columns, feature=feature_name)
# plot
pdp.pdp_plot(my_pdp, feature_name)
plt.show()
feature_name = 'time_in_hospital'
# Create the data for ploting
my_pdp = pdp.pdp_isolate(model=model_hosp, dataset=valid_X_hosp, model_features=valid_X_hosp.columns, feature=feature_name)
# plot
pdp.pdp_plot(my_pdp, feature_name)
plt.show()
# Get the average readmission rate for each time_in_hospital
# Do concat to keep validation data separate, rather than using all original data
all_train_hosp = pd.concat([train_X_hosp, train_y_hosp], axis=1)
all_train_hosp.groupby(['time_in_hospital']).mean().readmitted.plot()
plt.show()
# Use SHAP 
# Create sample data to test the function
sample_data_for_prediction = valid_X_hosp.iloc[0].astype(float)

# Create function
def patient_risk_factors(model, patient_data):
    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient_data)
    shap.initjs()
    return shap.force_plot(explainer.expected_value[1], shap_values[1], patient_data)
patient_risk_factors(model_hosp, sample_data_for_prediction)
# Summary SHAP plot for FIFA data

# Create pbbject that can calculate shap values 
explainer = shap.TreeExplainer(my_model_fifa)
# Calculate the shap values for all validation data for plotting
shap_values = explainer.shap_values(valid_X)
# Make plot
shap.summary_plot(shap_values[1], valid_X)    # shap_values[1] is for prediction of 'True'
# Create pbbject that can calculate shap values 
explainer = shap.TreeExplainer(my_model_fifa)

# Calculate the shap values for all validation data for plotting
shap_values = explainer.shap_values(X_fifa)

# Make plot
shap.dependence_plot('Ball Possession %', shap_values[1], X_fifa, interaction_index='Goal Scored')
base_features = ['number_inpatient', 'num_medications', 'number_diagnoses', 'num_lab_procedures', 
                 'num_procedures', 'time_in_hospital', 'number_outpatient', 'number_emergency', 
                 'gender_Female', 'payer_code_?', 'medical_specialty_?', 'diag_1_428', 'diag_1_414', 
                 'diabetesMed_Yes', 'A1Cresult_None']

X_hosp = hosp_re_data[base_features].astype(float)
y_hosp = hosp_re_data.readmitted
train_X_hosp_2, valid_X_hosp_2, train_y_hosp_2,valid_taxi_y_2 = train_test_split(X_hosp, y_hosp, random_state=1) 
# sample data for speed
small_valid_X_hosp_2 = valid_X_hosp_2[:150]
model_hosp_2 = RandomForestClassifier(n_estimators=30, random_state=1).fit(train_X_hosp_2, train_y_hosp_2)

hosp_re_data.describe()
explainer = shap.TreeExplainer(model_hosp_2)
shap_values = explainer.shap_values(small_valid_X_hosp_2)
shap.summary_plot(shap_values[1], small_valid_X_hosp_2)
shap.dependence_plot('num_lab_procedures', shap_values[1], small_valid_X_hosp_2)
shap.dependence_plot('num_medications', shap_values[1], small_valid_X_hosp_2)