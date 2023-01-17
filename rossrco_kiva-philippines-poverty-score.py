#numeric
import numpy as np
import pandas as pd

#visualization
import matplotlib.pyplot as plt
import seaborn as sns
import folium

from IPython.display import display

plt.style.use('bmh')
%matplotlib inline
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['axes.titlepad'] = 25
sns.set_color_codes('pastel')

#Pandas warnings
import warnings
warnings.filterwarnings('ignore')

#system
import os
print(os.listdir("../input"));
all_loans = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_loans.csv', index_col = 0)
all_loans.drop(['activity', 'use', 'country_code', 'posted_time', 'disbursed_time', 'funded_time', 'date', 'lender_count', 'tags'], axis = 1, inplace = True)
count_loans = all_loans[['country', 'loan_amount']].groupby(by = 'country').count().reset_index()
count_loans.rename(columns = {'loan_amount' : 'num_loans'}, inplace = True)
to_plot = count_loans.sort_values(by = 'num_loans', ascending = False)[:5]
x_ticks = np.arange(len(to_plot))
plt.xticks(x_ticks, to_plot.country, rotation = 45)
plt.ylabel('loan count')
plt.xlabel('country')
plt.title('Number of Loans - Top 5 Countries')
plt.bar(x_ticks, to_plot.num_loans)
loans_ph = pd.read_csv('../input/kiva-loans-philippines-transformed/kiva_loans_ph_transofrmed.csv', index_col = 0)
income_ph = pd.read_csv('../input/philippines-census-data-cleaned/philippines_census_data_cleaned.csv', index_col = 0)
income_ph_model = income_ph.join(pd.get_dummies(income_ph.region))
income_ph_model.drop('region', axis = 1, inplace = True)
income_ph_model.columns
inc_corr = income_ph_model[['household_income',
                            'income_from_entrepreneur_activities',
                            'main_inc_entrepreneur',
                            'main_inc_other',
                            'main_inc_wage']].corr()
inc_corr
fig = plt.figure(figsize = (18, 14))
plt.xticks(np.arange(len(inc_corr.columns)), inc_corr.columns, rotation = 90)
plt.yticks(np.arange(len(inc_corr.index)), inc_corr.index)
plt.title('Correlation Matrix - Income Columns (PH FIES Dataset)')
plt.imshow(inc_corr)
plt.colorbar()

income_ph_model.drop(['income_from_entrepreneur_activities',
                      'main_inc_entrepreneur',
                      'main_inc_other',
                      'main_inc_wage'],
                     axis = 1,
                     inplace = True)
fig = plt.figure(figsize = (18, 14))
plt.title('Correlation Matrix - All Columns (PH FIES Dataset)')
plt.xticks(np.arange(len(income_ph_model.corr().columns)), income_ph_model.corr().columns, rotation = 90)
plt.yticks(np.arange(len(income_ph_model.corr().index)), income_ph_model.corr().index)
plt.imshow(income_ph_model.corr())
plt.colorbar()
expense_corr = income_ph_model[['food_expenses',
                                'clothing_expenses',
                                'house_and_water_expenses', 
                                'medical_expenses',
                                'transport_expenses',
                                'comm_expenses',
                                'education_expenses',
                                'misc_expenses',
                                'special_occasion_expenses',
                                'farming_gardening_expenses',
                                'non_essential_expenses']].corr()
expense_corr
fig = plt.figure(figsize = (18, 14))
plt.xticks(np.arange(len(expense_corr.columns)), expense_corr.columns, rotation = 90)
plt.yticks(np.arange(len(expense_corr.index)), expense_corr.index)
plt.title('Correlation Matrix - Expense Columns (PH FIES Dataset)')
plt.imshow(expense_corr)
plt.colorbar()
income_ph_model.drop(['clothing_expenses',
                      'house_and_water_expenses', 
                      'medical_expenses',
                      'transport_expenses',
                      'comm_expenses',
                      'education_expenses',
                      'misc_expenses',
                      'special_occasion_expenses',
                      'non_essential_expenses'],
                     axis = 1,
                     inplace = True)
income_ph_model.drop(['house_singl_family',
                      'house_head_single',
                      'roof_material_strong',
                      'wall_material_strong',
                      'ws_other_toilet'],
                     axis = 1,
                     inplace = True)
fig = plt.figure(figsize = (18, 14))
plt.title('Correlation Matrix - All Columns (PH FIES Dataset)')
plt.xticks(np.arange(len(income_ph_model.corr().columns)), income_ph_model.corr().columns, rotation = 90)
plt.yticks(np.arange(len(income_ph_model.corr().index)), income_ph_model.corr().index)
plt.imshow(income_ph_model.corr())
plt.colorbar()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, normalize

scaled_income_ph_model = pd.DataFrame(normalize(income_ph_model), columns = income_ph_model.columns)
scaled_income_ph_model.head()

X_train, X_test, y_train, y_test = train_test_split(scaled_income_ph_model.drop('household_income', axis = 1),\
                                                    scaled_income_ph_model.household_income,\
                                                    test_size = 0.3,\
                                                    random_state = 42)

X_train.head()
print('The training data contains {} rows and {} columns'.format(X_train.shape[0], X_train.shape[1]))
valid_len = 9000
class DummyRegressor:
    def __init__ (self, pred_val = 1.):
        self.pred_val = pred_val
    
    def fit (self, X, y):
        pass
    
    def predict (self, X):
        return [[self.pred_val] for i in range(X.shape[0])]
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn import tree
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, explained_variance_score

regressors = [AdaBoostRegressor, SVR, SGDRegressor, tree.DecisionTreeRegressor, LinearRegression, MLPRegressor]
regressor_names = ['ada_boost', 'svr', 'sgd', 'decision_tree', 'linear', 'mlp']
metric_names = ['mean_squared_error', 'r2_score', 'mean_squared_log_error', 'explained_variance_score']
scorers = [mean_squared_error, r2_score, mean_squared_log_error, explained_variance_score]

valid_res = pd.DataFrame(columns = ['regressor'] + metric_names)

#Calculate the selected regressor results
for r, r_name, i in zip(regressors, regressor_names, range(len(regressor_names))):
    clf = r()
    clf.fit(X_train[valid_len:], y_train[valid_len:])
    
    metrics = []
    for scorer in scorers:
        metrics.append(scorer(y_train[:valid_len], clf.predict(X_train[:valid_len])))
    
    valid_res.loc[i] = [r_name] + metrics

valid_res_dummy = pd.DataFrame(columns = ['regressor'] + metric_names)
dummy_vals = (0., 0.5, 1)

#Append the dummy regressor results
for v, i in zip(dummy_vals, range(len(dummy_vals))):
    clf = DummyRegressor(v)
    metrics = []
    for scorer in scorers:
        metrics.append(scorer(y_train[:valid_len], clf.predict(X_train[:valid_len])))
    
    valid_res_dummy.loc[i] = ['dummy_%s' % v] + metrics

valid_res = valid_res.append(valid_res_dummy, ignore_index = True)
valid_res
regressors = [AdaBoostRegressor, tree.DecisionTreeRegressor, LinearRegression, MLPRegressor]
regressor_names = ['ada_boost', 'decision_tree', 'linear', 'mlp']

top_n_features = 10

for r, r_name in zip(regressors, regressor_names):
    clf = r()
    clf.fit(X_train[valid_len:], y_train[valid_len:])
    if r_name == 'linear':
        feature_importances = pd.Series(clf.coef_, index = X_train.columns)
    elif r_name == 'mlp':
        all_features = pd.DataFrame(clf.coefs_[0], index = X_train.columns)
        feature_importances = all_features.ix[:, 0]
    else:
        feature_importances = pd.Series(clf.feature_importances_, index = X_train.columns)
    
    print('The top %s feature importances for the %s are:' % (top_n_features, r_name))
    display(feature_importances.sort_values(ascending = False)[:top_n_features])
final_features = ['food_expenses', 'farming_gardening_expenses']

X_train, X_test = X_train[final_features], X_test[final_features]
X_train.head()
final_clf = tree.DecisionTreeRegressor(random_state = 42)
final_clf.fit(X_train, y_train)
metrics = pd.DataFrame(columns = ['metric', 'value'])

for metric, scorer, i in zip(metric_names, scorers, range(len(metric_names))):
    for set_type, set_data in zip(('train', 'test'), ([y_train, X_train], [y_test, X_test])):
        metrics.loc[i] = [metric + '_' + set_type, scorer(set_data[0], final_clf.predict(set_data[1]))]

metrics
from sklearn.externals.six import StringIO  
from IPython.display import Image as PImage  
from sklearn.tree import export_graphviz
import graphviz
from subprocess import check_call
from PIL import Image, ImageDraw, ImageFont

with open('kiva_decision.dot', 'w') as f:
     f = tree.export_graphviz(final_clf,
                              out_file = f,
                              max_depth = 3,
                              feature_names = X_train.columns,
                              rounded = True,
                              filled = True)
        
check_call(['dot', '-Tpng', 'kiva_decision.dot', '-o', 'kiva_decision.png'])

img = Image.open('kiva_decision.png')
draw = ImageDraw.Draw(img)
draw.text((10, 0),
          'Kiva Decision Process For Poverty Score Assessment',
          (0, 0, 255))
img.save('tree-out.png')
PImage('tree-out.png')
pred_train = pd.DataFrame(final_clf.predict(X_train), index = X_train.index, columns = ['inc_pred'])
pred_test = pd.DataFrame(final_clf.predict(X_test), index = X_test.index, columns = ['inc_pred'])

all_predictions = pred_train.append(pred_test)

all_predictions['poverty_score'] = 1 - all_predictions['inc_pred']
all_predictions.describe()
fig = plt.figure(figsize = (18, 10), dpi = 120)
fig.subplots_adjust(hspace = 0.3, wspace = 0.35)

ax1 = fig.add_subplot(2, 3, 1)
sns.distplot(all_predictions.poverty_score, label = 'inc_pred')

ax2 = fig.add_subplot(2, 3, 2)
sns.distplot(income_ph_model.household_income[income_ph_model.household_income < 1200000])
all_predictions = all_predictions.join(income_ph[['region', 'house_head_sex_f']])
all_predictions.head()
sns.boxplot(x = 'region', y = 'poverty_score', hue = 'house_head_sex_f', data = all_predictions, palette = "PRGn")
sns.despine(offset=10, trim=True)
plt.xticks(rotation = 45)
predictions_grouped = all_predictions.drop('inc_pred', axis = 1).groupby(by = ['region', 'house_head_sex_f']).mean().reset_index()
sns.barplot(x = 'region', y = 'poverty_score', data = predictions_grouped, color = 'b')
plt.title('Poverty Score by Region')
plt.xticks(rotation = 45)
merged_loans = pd.merge(left = loans_ph, right = predictions_grouped, how = 'left', on = ['region', 'house_head_sex_f'])
merged_loans.to_csv('kiva_loans_plus_poverty_score.csv')
merged_loans.head()
loan_regions = pd.read_csv('../input/data-science-for-good-kiva-crowdfunding/kiva_mpi_region_locations.csv')
loan_regions.head()

loan_regions_ph = loan_regions[loan_regions.country == 'Philippines']
loan_regions_ph

region_mapping_phil_loans = {'National Capital Region' : 'ncr',\
                             'Cordillera Admin Region' : 'car',\
                             'Ilocos Region' : 'ilocos',\
                             'Cagayan Valley' : 'cagayan valley',\
                             'Central Luzon' : 'central luzon',\
                             'Calabarzon' : 'calabarzon',\
                             'Mimaropa' : 'mimaropa',\
                             'Bicol Region' : 'bicol',\
                             'Western Visayas' : 'western visayas',\
                             'Central Visayas' : 'central visayas',\
                             'Eastern Visayas' : 'eastern visayas',\
                             'Zamboanga Peninsula' : 'zamboanga',\
                             'Northern Mindanao' : 'northern mindanao',\
                             'Davao Peninsula' : 'davao',\
                             'Soccsksargen' : 'soccsksargen',\
                             'CARAGA' : 'caraga',\
                             'Armm' : 'armm'}

loan_regions_ph.region = loan_regions_ph.region.map(region_mapping_phil_loans)

merged_loans = merged_loans.merge(loan_regions_ph, how = 'left', on = ['region'])
merged_loans.to_csv('kiva_loans_plus_poverty_score_regional_coordinates.csv')
vis_loans = merged_loans[['loan_amount', 'poverty_score', 'lat', 'lon', 'region']].groupby(by = 'region').mean().reset_index()

fig, ax1 = plt.subplots()
ax1.bar(np.arange(len(vis_loans)), vis_loans.loan_amount, align = 'center', label = 'loan_amount', width = 0.2, color = 'r')

ax1.set_xticks(np.arange(len(vis_loans)))
ax1.set_xticklabels(vis_loans.region, rotation = 45)

ax2 = ax1.twinx()
ax2.bar(np.arange(len(vis_loans)) + 0.2, vis_loans.poverty_score, align = 'center', label = 'poverty_score', width = 0.2, color = 'b')
plt.legend()
ax1.set_ylabel('loan_amount')
ax2.set_ylabel('poverty_score')
fig.tight_layout()
vis_loans = merged_loans[['loan_amount', 'poverty_score', 'lat', 'lon', 'region']].groupby(by = 'region').max().reset_index()

m = folium.Map(location = [13.5, 119], tiles = 'Mapbox Bright', zoom_start = 5.8)
for i in range(0, len(vis_loans)):
    folium.Circle(\
                  location = [vis_loans.iloc[i].lat ,\
                              vis_loans.iloc[i].lon],\
                  radius = vis_loans.iloc[i].poverty_score * 250000,\
                  color = 'red',\
                  fill = True,\
                  stroke = False,\
                  fillOpacity = .2
   ).add_to(m)
    
'''    folium.Circle(\
                  location = [vis_loans.iloc[i].lat ,\
                              vis_loans.iloc[i].lon],\
                  radius = vis_loans.iloc[i].loan_amount * 50,\
                  color = 'blue',\
                  fill = True,\
                  stroke = False,\
                  fillOpacity = .2
   ).add_to(m)'''

mapWidth, mapHeight = (400, 500)
m
