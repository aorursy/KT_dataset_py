import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel
from numpy import sort
import seaborn as sns
from sklearn.externals import joblib
from pdpbox import pdp

plt.style.use('ggplot')

data = pd.read_csv('../input/avocado.csv')
data.head()
data['Date'] = pd.to_datetime(data.Date)
data['day_of_week'] = data['Date'].dt.weekday_name
data.day_of_week.unique()
data['month'] = data['Date'].dt.month
data['day'] = data['Date'].dt.day
data = data.rename(columns={'Unnamed: 0': 'Store'})
data = data.rename(columns={'4046': 'small Hass', '4225':  'large Hass', '4770':  'extra large Hass'})
print('Unique values in columns with text:\n\n Dates: {0} \n\n Data type: {1} \n\n Year: {2} \n\n Region: {3} \n\n Day of week: {4} \n\n Month: {5}'.format(data.Date.unique(), data.type.unique(), data.year.unique(), data.region.unique(), data['day_of_week'].unique(), data.month.unique()))
data.info()
mappings_type = {'conventional':0, 'organic':1}

mappings_dayofweek = {'Sunday':1}

mappings_region = {}

v = 0

regions = list(data.region.unique())

numbers = []

for i in regions:
    v = v+1
    numbers.append(v)

d = zip(regions, numbers)

mappings_region = dict(d)

data.type.replace(mappings_type, inplace=True)
data.day_of_week.replace(mappings_dayofweek, inplace=True)
data.region.replace(mappings_region, inplace=True)
data.head()
data.describe()
skew_df = pd.DataFrame(data.skew(), columns={'Skewness'})
skew_df
kurt_df = pd.DataFrame(data.kurtosis(), columns={'Kurtosis'})
kurt_df
X = data.drop(['AveragePrice', 'Date'], axis=1)

y = data['AveragePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

model = XGBRegressor(n_jobs=4)
model.fit(X_train, 
            y_train,
            verbose=True)
predictions = model.predict(X_test)
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=5)
print(scores)
print('Mean Absolute Error: %2f' %(-1 * scores.mean()))
mae = mean_absolute_error(predictions, y_test)
print("Mean Absolute Error : " + str(mae))

error_percent = mae/data['AveragePrice'].mean()*100
print(str(error_percent) + ' %')
# plot feature importance
fig, ax = plt.subplots(figsize=(15, 15))
imp_plt = plot_importance(model, ax=ax)
features_to_plot = ['region', 'year']
inter1  =  pdp.pdp_interact(model=model, dataset=X_test, model_features=X.columns.tolist(), features=features_to_plot)

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=model, dataset=X_test, model_features=X.columns.tolist(), feature='month')

# plot it
pdp.pdp_plot(pdp_goals, 'Month')

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=model, dataset=X_test, model_features=X.columns.tolist(), feature='year')

# plot it
pdp.pdp_plot(pdp_goals, 'Year')

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=model, dataset=X_test, model_features=X.columns.tolist(), feature='region')

# plot it
pdp.pdp_plot(pdp_goals, 'Region')
plt.show()
row_to_show = 5
data_for_prediction = X
import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(model)

# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)


mae = mean_absolute_error(predictions, y_test)
error_percent = mae/data['AveragePrice'].mean()*100

accuracy = mean_absolute_error(predictions, y_test)
print("Mean Absolute Error : " + str(mae) + "\t" + str(error_percent) + ' %')

#scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=5)
#print(scores)
#print('Mean Absolute Error: %2f' %(-1 * scores.mean()))
# Fit model using each importance as a threshold
thresholds = sort(model.feature_importances_)

best_score = {}

for thresh in thresholds:
    # select features using threshold
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    select_X_train = selection.transform(X_train)
    # train model
    selection_model = XGBRegressor(n_jobs=4)
    selection_model.fit(select_X_train, y_train)
    # eval model
    select_X_test = selection.transform(X_test)
    predictions = selection_model.predict(select_X_test)
    #print('Mean Absolute Error: %2f' %(-1 * scores.mean()))
    accuracy = mean_absolute_error(predictions, y_test)
    mae = mean_absolute_error(predictions, y_test)
    error_percent = mae/data['AveragePrice'].mean()*100
    #print("Thresh={0:f}, n={1:f}, Accuracy: {2:f}, Mean Absolute Error {3:f}: , err_perct: {4:f}%".format(thresh, select_X_train.shape[1], accuracy * 100, mae, error_percent))
    
    best_score[select_X_train.shape[1]] = str(error_percent) + ' %'

    
print(best_score)
value = best_score.values()
key = best_score.keys()

min_val = min(value)
min_key = min(best_score, key=best_score.get)
print('Best amout of features: key: {0}, value: {1}'.format(min_key, min_val))
X_opt = data.drop(['AveragePrice', 'Date', 'XLarge Bags'], axis=1)
y_opt = data['AveragePrice']

opt_X_train, opt_X_test, opt_y_train, opt_y_test = train_test_split(X_opt, y_opt, test_size=0.33, random_state=42)
opt_pipeline = Pipeline([('xgb', XGBRegressor(n_jobs=4))])

param_grid = {
    "xgb__n_estimators": [100, 250, 500, 1000],
    "xgb__learning_rate": [0.1, 0.25, 0.5, 1],
    "xgb__max_depth": [6, 7, 8],
    "xgb__min_child_weight": [0.25, 0.5, 1, 1.5]
}

fit_params = {"xgb__eval_set": [(opt_X_test, opt_y_test)], 
              "xgb__early_stopping_rounds": 10, 
              "xgb__verbose": False} 

searchCV = GridSearchCV(opt_pipeline, cv=5, param_grid=param_grid, fit_params=fit_params)
searchCV.fit(opt_X_train, opt_y_train)
searchCV.best_params_
searchCV.cv_results_['mean_train_score']
searchCV.cv_results_['mean_test_score']
searchCV.cv_results_['mean_train_score'].mean(), searchCV.cv_results_['mean_test_score'].mean()
opt_predictions = searchCV.predict(opt_X_test)
mae = mean_absolute_error(opt_predictions, opt_y_test)
print("Mean Absolute Error : " + str(mae))

error_percent = mae/data['AveragePrice'].mean()*100
print("Error percentage: " + str(error_percent) + ' %')
joblib.dump(searchCV, "xgboostmodel.joblib.dat")