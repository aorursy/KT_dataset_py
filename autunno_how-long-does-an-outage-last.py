import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.neighbors import KNeighborsRegressor
# Allow us to see all columns when printing
pd.set_option('display.max_columns', 1000)

# Disable chained assignment warning message
pd.options.mode.chained_assignment = None 

# Read the dataset and print the first 5 rows
dataset = pd.read_csv('../input/Grid_Disruption_00_14_standardized - Grid_Disruption_00_14_standardized.csv')
dataset.head()
print("Number of entries: " + str(len(dataset.index)))
len(pd.to_numeric(dataset['Year'], 'coerce').dropna().astype(int))
len(pd.to_numeric(dataset['Demand Loss (MW)'], 'coerce').dropna().astype(int))
len(pd.to_numeric(dataset['Number of Customers Affected'], 'coerce').dropna().astype(int))
len(dataset[pd.isnull(pd.to_datetime(dataset['Date Event Began'], 'coerce'))])
len(dataset[pd.isnull(pd.to_datetime(dataset['Date of Restoration'], 'coerce'))])
len(dataset[pd.isnull(pd.to_datetime(dataset['Time Event Began'], 'coerce'))])
len(dataset[pd.isnull(pd.to_datetime(dataset['Time of Restoration'], 'coerce'))])
dataset = dataset[dataset.columns.difference(['Demand Loss (MW)', 'Number of Customers Affected', 'Event Description'])]
dataset.replace('Unknown', np.nan, inplace=True)
dataset.replace('Ongoing', np.nan, inplace=True)
dataset.isnull().any()
print("Total number of rows: " + str(len(dataset.index)))
print("Number of empty values:")
for column in dataset.columns:
    print(" * " + column + ": " + str(dataset[column].isnull().sum()))
dataset = dataset.dropna()
print("Total number of rows: " + str(len(dataset.index)))
print("Number of empty values:")
for column in dataset.columns:
    print(" * " + column + ": " + str(dataset[column].isnull().sum()))
wrong_date_began = dataset[pd.isnull(pd.to_datetime(dataset['Date Event Began'], 'coerce'))].index
print(wrong_date_began)
wrong_date_restoration = dataset[pd.isnull(pd.to_datetime(dataset['Date of Restoration'], 'coerce'))].index
print(wrong_date_restoration)
wrong_time_began = dataset[pd.isnull(pd.to_datetime(dataset['Time Event Began'], 'coerce'))].index
print(wrong_time_began)
wrong_time_restoration = dataset[pd.isnull(pd.to_datetime(dataset['Time of Restoration'], 'coerce'))].index
print(wrong_time_restoration)
# append all wrong dates into a single array, turn into a set to remove duplicated indexes and drop them from the feature map
wrong_dates = set(wrong_date_began.append(wrong_date_restoration).append(wrong_time_began).append(wrong_time_restoration))
dataset = dataset.drop(wrong_dates)
print(len(dataset))
dataset.reset_index()
dataset.head()
date_start = pd.to_datetime(dataset['Date Event Began'] + ' ' + dataset['Time Event Began'])
date_end = pd.to_datetime(dataset['Date of Restoration'] + ' ' + dataset['Time of Restoration'])
date_end.head()
# Create the attribute'Duration in Minutes', which is the difference between the end and start date of the event.
dataset['Duration in Minutes'] = (date_end - date_start).dt.total_seconds() / 60
dataset = dataset[dataset.columns.difference(['Time Event Began', 'Time of Restoration'])]

dataset.head()
dataset = dataset[(dataset['Duration in Minutes'] > 0)]
dataset.loc[dataset['Duration in Minutes'].idxmax()]
print('Number of outages that lasted over 3 days: ' + str(len(dataset[(dataset['Duration in Minutes'] > 4320)])))
long_outages = dataset[(dataset['Duration in Minutes'] >= 4320)]
dataset = dataset[(dataset['Duration in Minutes'] < 4320)]
dataset.iloc[:, 6].head()
test_split = dataset['Tags'].str.split(',', expand=True)
test_split.head()
tags = dataset.Tags.str.split(',').tolist()
unique_events = set(x.lstrip() for lst in tags for x in lst)
print(unique_events)
tags = dataset['Tags']
unique_events.remove('unknown')
labelencoder = LabelEncoder()

# Create the new features and encode them
for event in unique_events:
    dataset[event] = tags.str.contains(event)
    dataset[event] = labelencoder.fit_transform(dataset[event])
    
dataset = dataset[dataset.columns.difference(['Tags'])]   
dataset.head()
areas = dataset['Geographic Areas'].unique()
print('Ammount of unique areas: ', len(areas))
print('Unique areas: ' + str(len(areas)))
dataset = dataset[dataset.columns.difference(['Geographic Areas'])]
nerc_regions = dataset['NERC Region'].unique()
print('Unique NERC Regions (' + str(len(nerc_regions)) + '):')
print(nerc_regions)
respondent = dataset['Respondent'].unique()
print('Unique Respondents: ' + str(len(respondent)) )
# Apply one hot encoding to NERC Region
nerc_region = pd.get_dummies(dataset['NERC Region'], drop_first=True)
print(nerc_region.columns)

# Append the NERC Region OneHotEncoded attributes to the feature map
dataset = pd.concat([dataset, nerc_region], axis=1)

# Remove the original attributes
dataset = dataset[dataset.columns.difference(['Respondent'])]
# Deal with 'RFC, SERC' attribute
dataset['RFC'] = np.where(dataset['RFC, SERC'] + dataset['RFC'] > 0, 1, 0)
dataset['SERC'] = np.where(dataset['RFC, SERC'] + dataset['SERC'] > 0, 1, 0)

# Deal with 'NPCC, RFC' attribute
dataset['NPCC'] = np.where(dataset['NPCC, RFC'] + dataset['NPCC'] > 0, 1, 0)
dataset['RFC'] = np.where(dataset['NPCC, RFC'] + dataset['RFC'] > 0, 1, 0)

# Deal with 'FRCC, SERC' attribute
dataset['FRCC'] = np.where(dataset['FRCC, SERC'] + dataset['FRCC'] > 0, 1, 0)
dataset['SERC'] = np.where(dataset['FRCC, SERC'] + dataset['SERC'] > 0, 1, 0)

# Remove the old (and now unneeded) attributes
dataset = dataset[dataset.columns.difference(['RFC, SERC', 'NPCC, RFC', 'FRCC, SERC'])]
dataset.head()
dim = (15, 10)
fig, ax = plt.subplots(figsize=dim)
tag_plot = sns.countplot(x="NERC Region", ax=ax, data=dataset)

for item in tag_plot.get_xticklabels():
    item.set_rotation(45)
dim = (15, 10)
fig, ax = plt.subplots(figsize=dim)
demand_plot = sns.boxplot(x="Year", y="Duration in Minutes", ax=ax, data=dataset)

for item in demand_plot.get_xticklabels():
    item.set_rotation(45)
dim = (15, 10)
fig, ax = plt.subplots(figsize=dim)
demand_plot = sns.boxplot(x="NERC Region", y="Duration in Minutes", ax=ax, data=dataset)

for item in demand_plot.get_xticklabels():
    item.set_rotation(45)
dim = (15, 10)
fig, ax = plt.subplots(figsize=dim)
sns.countplot(x="Year", ax=ax, data=dataset)
dim = (15, 10)
fig, ax = plt.subplots(figsize=dim)
demand_plot = sns.boxplot(x="Year", y="Duration in Minutes", ax=ax, data=long_outages)

for item in demand_plot.get_xticklabels():
    item.set_rotation(45)
dim = (20, 10)
fig, ax = plt.subplots(figsize=dim)
tag_plot = sns.countplot(x="NERC Region", ax=ax, data=long_outages)

for item in tag_plot.get_xticklabels():
    item.set_rotation(45)
dim = (20, 10)
fig, ax = plt.subplots(figsize=dim)
tag_plot = sns.countplot(x="Tags", ax=ax, data=long_outages)

for item in tag_plot.get_xticklabels():
    item.set_rotation(45)
#correlation matrix
corrmat = dataset.corr()
f, ax = plt.subplots(figsize=(15, 13))
sns.heatmap(corrmat, vmax=.8, square=True);
output = dataset.iloc[:, 2]
features = dataset[dataset.columns.difference(['Duration in Minutes','Date Event Began', 'Date of Restoration', 'Year', 'NERC Region'])]
features['severe weather'] = np.where(features['storm'] + features['severe weather'] > 0, 1, 0)
features = features[features.columns.difference(['storm'])]
features['physical'] = np.where(features['vandalism'] + features['physical'] > 0, 1, 0)
features = features[features.columns.difference(['physical'])]
features.head()
output.head()
print('Features size: ' + str(len(features)) + ' - Output size: ' + str(len(output)))
features_train, features_test, duration_train, duration_test = train_test_split(features, output, test_size = 0.3, random_state = 0)
# Run grid search, get the prediction array and print the accuracy and best combination
def fit_and_pred_grid_classifier(regressor, param_grid, X_train, X_test, y_train, y_test, folds=5):
    # Apply grid search with F1 Score to help balance the results (avoid bias on "no attrition")
    grid_search = GridSearchCV(estimator = regressor, param_grid = param_grid, cv = folds, n_jobs = -1, verbose = 0)
    grid_search.fit(X_train, y_train)
    best_accuracy = grid_search.best_score_
    best_parameters = grid_search.best_params_

    # Get the prediction array
    grid_search_pred = grid_search.predict(X_test)

    # Print the MSE, R2 score and best parameter combination
    print("MSE: " + str(mean_squared_error(y_test, grid_search_pred))) 
    print("R2: " + str(r2_score(y_test, grid_search_pred))) 
    print("Best parameter combination: " + str(best_parameters)) 
    
    return grid_search_pred
regressor = SVR(kernel='rbf', C=10, gamma=0.1)
param_grid = [
    {
        'C': [400, 450, 500, 550, 600], 
        'kernel': ['linear'],
        'epsilon': [550, 600, 650, 700, 750],
    }, 
    {
        'C': [12, 13, 14, 15, 20], 
        'kernel': ['rbf', 'sigmoid'], 
        'gamma': [0.3, 0.4, 0.5, 1],
        'epsilon': [10, 100, 500, 750, 1000],
    },
]
# Build and fit the grid search SVR model
pred = fit_and_pred_grid_classifier(regressor, param_grid, features_train, features_test, duration_train, duration_test)
X_grid = np.arange(1, 367)

dim = (20, 10)
fig, ax = plt.subplots(figsize=dim)
ax.set_xticks([])
plot_1 = sns.swarmplot(x=X_grid, y=duration_test, color='green')
plot_2 = sns.swarmplot(x=X_grid, y=pred, color='red')
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

for item in plot_1.get_xticklabels():
    item.set_visible(False)
    
for item in plot_2.get_xticklabels():
    item.set_visible(False)
regressor = DecisionTreeRegressor()
param_grid = {
    'max_depth': [ 50, 75, 85, 90, 95, 100, 105, 110],
    'max_features': [10, 13, 14, 15, 16, 17, 18, 19, 20],
    'min_samples_leaf': [10, 11, 12, 13, 14, 15],
    'min_samples_split': [5, 6, 7, 8, 9, 10]
}
pred = fit_and_pred_grid_classifier(regressor, param_grid, features_train, features_test, duration_train, duration_test)
regressor = BayesianRidge()
param_grid = [
    {
        'alpha_1': [125, 150, 175, 200, 225, 250],
        'alpha_2': [0.00000001, 0.0000001, 0.000001],
        'lambda_1': [0.00000001, 0.0000001, 0.000001],
        'lambda_2': [125, 150, 175, 200, 225, 250]
    }, 
]
pred = fit_and_pred_grid_classifier(regressor, param_grid, features_train, features_test, duration_train, duration_test)
regressor = KNeighborsRegressor()
param_grid = {
    'n_neighbors': [1,2,4,5,10,20],
    'weights': ['distance', 'uniform'],
    'algorithm': ['ball_tree', 'kd_tree', 'brute'],
    'metric': ['minkowski','euclidean','manhattan'], 
    'p': [1, 2]
}
pred = fit_and_pred_grid_classifier(regressor, param_grid, features_train, features_test, duration_train, duration_test)