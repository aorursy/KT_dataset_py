# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

os.symlink('../input', 'input')

monthly_data = pd.read_csv('input/housing-in-london/housing_in_london_monthly_variables.csv',
                                  parse_dates=['date'])
yearly_data = pd.read_csv('input/housing-in-london/housing_in_london_yearly_variables.csv',
                                 parse_dates=['date'])
monthly_data.head()
yearly_data.head()
# fraction values that are not null
yearly_data.notnull().sum()/len(yearly_data)
monthly_data.notnull().sum()/len(monthly_data)
# keep only the rows corresponding to locations in London
london_yearly = yearly_data.loc[yearly_data['borough_flag'] == 1]
london_monthly = monthly_data.loc[monthly_data['borough_flag'] == 1]
# check to make sure the areas match between the yearly and monthly datasets
set(london_monthly['area'].unique()) == set(london_yearly['area'].unique())
# the sum of prices over all the houses sold (average times sale count)
sum_prices = london_monthly['average_price'] * london_monthly['houses_sold']
# group prices by year and area, and sum for each distinct (year, area) pair
sum_prices = pd.concat([london_monthly['date'], london_monthly['area'], sum_prices], axis=1)
sum_prices = sum_prices.groupby([sum_prices['date'].dt.year, sum_prices['area']]).sum()

# total number of houses sold
sum_sales = london_monthly['houses_sold'].groupby([london_monthly['date'].dt.year, london_monthly['area']]).sum()

# element wise division of the average prices by the number of houses sold
monthly_average = sum_prices.div(sum_sales, axis=0)
monthly_average
missing_crimes = london_monthly[london_monthly['no_of_crimes'].isnull()]
missing_crimes.groupby(missing_crimes['date'].dt.year).size()
# total number of crimes by year and area
sum_crimes = london_monthly['no_of_crimes'].groupby([london_monthly['date'].dt.year, london_monthly['area']]).sum()
sns.lineplot(sum_crimes.reset_index()['date'], sum_crimes.reset_index()['no_of_crimes'])
from sklearn.impute import SimpleImputer

# imputation
imputer = SimpleImputer(strategy='mean')
imputed_crimes = pd.DataFrame(imputer.fit_transform(np.array(london_monthly['no_of_crimes']).reshape(-1, 1)))
imputed_crimes[['date', 'area']] = london_monthly[['date', 'area']]

# sum of crimes by year and area
sum_crimes = imputed_crimes.groupby([imputed_crimes['date'].dt.year, imputed_crimes['area']]).sum()

# putting it together with our averge prices
yearly_aggregates = pd.concat([sum_crimes, monthly_average], axis=1)
yearly_aggregates.columns = ['average_price', 'num_crimes']
yearly_aggregates
# for example: the average house price and number of crimes in Hounslow, in 2014 
yearly_aggregates.loc[(2014, 'hounslow')]
features_to_use = ['area', 'date', 'median_salary', 'mean_salary', 'recycling_pct', 'population_size', 'number_of_jobs', 'area_size', 'no_of_houses']
total_data = london_yearly[features_to_use]
# the date is the same day of each year, so we can simplify our values by dropping day and month
total_data['date'] = total_data['date'].dt.year

# join with aggregated monthly data
total_data = total_data.set_index(['date', 'area']).join(yearly_aggregates).reset_index()
total_data
sns.barplot(total_data['area'], total_data['average_price'])
total_data.loc[total_data['mean_salary'] == '#']
total_data['mean_salary'] = total_data.replace('#', 'NaN')['mean_salary'].astype(float)
sns.scatterplot(total_data['median_salary'], total_data['average_price'], hue=total_data['area'])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.scatterplot(total_data['mean_salary'], total_data['average_price'], hue=total_data['area'])
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
total_data['recycling_pct'] = total_data['recycling_pct'].replace('na', 'NaN').astype(float)
total_data.dtypes
from sklearn.model_selection import train_test_split

X = total_data.drop(['average_price'], axis=1)
y = total_data['average_price']

X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

categorical_features = [col for col in X_train.columns if X_train[col].dtype == "object"]
numeric_features = [col for col in X_train.columns if X_train[col].dtype in ['float64']]

numerical_preprocessor = Pipeline([('imputer', SimpleImputer())])
categorical_preprocessor = Pipeline([('encoder', OneHotEncoder(sparse=False))])

preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', numerical_preprocessor, numeric_features),
                        ('cat', categorical_preprocessor, categorical_features)])

def train_model(model):
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_val)

    score = mean_absolute_error(predictions, y_val)
    return score
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

xgb_model = XGBRegressor()
random_forest_model = RandomForestRegressor()

random_forest_score = train_model(random_forest_model)
xgb_score = train_model(xgb_model)
print(f"MAE for random forest: {random_forest_score}\nMAE for XGB Regressor: {xgb_score}")
from xgboost import plot_importance as plot_xgb_importance

# the xgboost importance plots are less versatile than is available from sklearn models, but we can still see what we get
plot_xgb_importance(xgb_model)
import eli5
from eli5.sklearn import PermutationImportance

preprocessed_features = list(X_val.columns[2:]) + list(X_val['area'].unique())

preprocessed = pd.DataFrame(preprocessor.fit_transform(X_val, y_val))
preprocessed.columns = preprocessed_features
perm = PermutationImportance(model, random_state=1).fit(preprocessed, y_val)
eli5.show_weights(perm, feature_names=preprocessed.columns.tolist())
from pdpbox import pdp, get_dataset, info_plots

pdp_num_jobs = pdp.pdp_isolate(model=model, dataset=preprocessed, model_features=preprocessed_features,
                                   feature='number_of_jobs')

pdp_recycling_pct = pdp.pdp_isolate(model=model, dataset=preprocessed, model_features=preprocessed_features,
                                   feature='recycling_pct')

pdp.pdp_plot(pdp_num_jobs, 'Number of Jobs')
pdp.pdp_plot(pdp_recycling_pct, 'Recycling Percentage')
plt.show()
import shap

data_for_prediction = preprocessed.iloc[-10]

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(data_for_prediction)

shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, data_for_prediction)
shap_values = explainer.shap_values(preprocessed)
shap.summary_plot(shap_values, preprocessed)
shap.dependence_plot('no_of_houses', shap_values, preprocessed, interaction_index='area_size')
