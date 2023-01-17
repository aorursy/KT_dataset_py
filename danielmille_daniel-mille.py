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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import ravel
import seaborn as sns
import pickle
from joblib import dump, load
import scipy as sp

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
import lightgbm
train = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip')
test = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/test.csv.zip')
features = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip')
stores = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv')
train = pd.merge(train, stores, on = 'Store', how = 'left')
train = pd.merge(train, features, on = ['Store', 'Date'], how = 'left')
test = pd.merge(test, stores, on = 'Store', how = 'left')
test = pd.merge(test, features, on = ['Store', 'Date'], how = 'left')
#Creating new columns with Year, Month and Week of Year
train.Date = pd.to_datetime(train.Date)
train['Year'], train['Month'], train['Week'] = train.Date.dt.year, train.Date.dt.month, train.Date.dt.strftime('%V')
test.Date = pd.to_datetime(test.Date)
test['Year'], test['Month'], test['Week'] = test.Date.dt.year, test.Date.dt.month, test.Date.dt.strftime('%V')
# Grouping Weekly Sales by week and extracting min, max and median values for each week
train_summary = train.groupby(['Week']).agg({'Weekly_Sales': [np.min, np.max, np.mean]})
train_median = train[['Week', 'Weekly_Sales']].groupby('Week').median()
#Ploting Min, Max and Median weekly sales trend with holidays highlighted
plt.plot(train_summary.index, train_summary.Weekly_Sales.amin, color = "red", label = "Min Weekly Sales")
plt.plot(train_summary.index, train_summary.Weekly_Sales.amax, color = "blue", label = "Max Weekly Sales")
plt.plot(train_median.index, train_median.values, color = 'green', label = "Median Weekly Sales")
plt.axvspan(4.5, 5.5, alpha=0.5, color='yellow')
plt.axvspan(34.5, 35.5, alpha=0.5, color='red')
plt.axvspan(45.5, 46.5, alpha=0.5, color='blue')
plt.axvspan(49.5, 50.5, alpha=0.5, color='green')
plt.axvspan(26.5, 27.5, alpha=0.5, color='brown')
plt.rcParams["figure.figsize"] = (18,8)
plt.fill_between(train_summary.index, train_summary.Weekly_Sales.amin, train_summary.Weekly_Sales.amax, 
                 facecolor = "grey", alpha = 0.3)
plt.rcParams.update({'font.size': 14})
plt.legend()
plt.xlabel("Weeks over the year")
plt.ylabel("Weekly Sales US$")
plt.title("Weekly Sales over the Weeks")
plt.show()
train2010 = train[train.Year == 2010]
train2011 = train[train.Year == 2011]
train2012 = train[train.Year == 2012]

train2010mean = train2010[['Weekly_Sales', 'Week']].groupby('Week').mean()
train2011mean = train2011[['Weekly_Sales', 'Week']].groupby('Week').mean().tail(48)
train2012mean = train2012[['Weekly_Sales', 'Week']].groupby('Week').mean().tail(39)
train2010min = train2010[['Weekly_Sales', 'Week']].groupby('Week').min()
train2011min = train2011[['Weekly_Sales', 'Week']].groupby('Week').min().tail(48)
train2012min = train2012[['Weekly_Sales', 'Week']].groupby('Week').min().tail(39)
plt.plot(train2010mean.index, train2010mean.values, color = "blue", label = "Mean Weekly Sales 2010")
plt.plot(train2011mean.index, train2011mean.values, color = "red", label = "Mean Weekly Sales 2011")
plt.plot(train2012mean.index, train2012mean.values, color = 'green', label = "Mean Weekly Sales 2012")
plt.plot(train2010min.index, train2010min.values, color = "blue", label = "Min Weekly Sales 2010")
plt.plot(train2011min.index, train2011min.values, color = "red", label = "Min Weekly Sales 2011")
plt.plot(train2012min.index, train2012min.values, color = 'green', label = "Min Weekly Sales 2012")

plt.axvspan(0.5, 1.5, alpha=0.5, color='yellow')
plt.axvspan(30.5, 31.5, alpha=0.5, color='red')
plt.axvspan(41.5, 42.5, alpha=0.5, color='blue')
plt.axvspan(45.5, 46.5, alpha=0.5, color='green')
plt.rcParams["figure.figsize"] = (18,8)
plt.rcParams.update({'font.size': 14})
plt.legend()
plt.xticks(np.arange(1, 53, step=1))
plt.xlabel("Weeks over the year")
plt.ylabel("Weekly Sales US$")
plt.title("Weekly Sales over the Weeks")
plt.show()
train_A = train[train.Type == 'A']
train_B = train[train.Type == 'B']
train_C = train[train.Type == 'C']

trainA_median = train_A[['Week', 'Weekly_Sales']].groupby('Week').median()
trainB_median = train_B[['Week', 'Weekly_Sales']].groupby('Week').median()
trainC_median = train_C[['Week', 'Weekly_Sales']].groupby('Week').median()

plt.plot(trainA_median.index, trainA_median.values, color = "red", label = "Median Weekly Sales Type A Stores")
plt.plot(trainB_median.index, trainB_median.values, color = 'green', label = "Median Weekly Sales Type B Stores")
plt.plot(trainC_median.index, trainC_median.values, color = 'blue', label = "Median Weekly Sales Type C Stores")
plt.axvspan(4.5, 5.5, alpha=0.5, color='yellow')
plt.axvspan(34.5, 35.5, alpha=0.5, color='red')
plt.axvspan(45.5, 46.5, alpha=0.5, color='blue')
plt.axvspan(49.5, 50.5, alpha=0.5, color='green')
plt.rcParams["figure.figsize"] = (18,8)
plt.rcParams.update({'font.size': 14})
plt.legend()
plt.xlabel("Weeks over the year")
plt.ylabel("Weekly Sales US$")
plt.title("Median Weekly Sales over the Weeks")
plt.show()
train.loc[(train.Year==2010) & (train.Week=='13'), 'IsEaster'] = True
train.loc[(train.Year==2011) & (train.Week=='16'), 'IsEaster'] = True
train.loc[(train.Year==2012) & (train.Week=='14'), 'IsEaster'] = True
train['IsEaster'] = train.IsEaster.apply(lambda x: 1 if x == True else 0)
train['IsChristmas'] = train.Week.apply(lambda x: 1 if (x == '51' or x == '52') else 0)
train['IsSuperBowl'] = train.Week.apply(lambda x: 1 if x == '06' else 0)
train['IsLaborDay'] = train.Week.apply(lambda x: 1 if x == '36' else 0)
train['IsThanksGiving'] = train.Week.apply(lambda x: 1 if x == '47' else 0)
train['IsEaster'] = train.IsEaster.apply(lambda x: 1 if x == True else 0)

train.drop(['IsHoliday_x'], axis = 1, inplace = True)
test.drop(['IsHoliday_x'], axis = 1, inplace = True)
train_superbowl = train[train.IsSuperBowl == 1]
superBowl = train_superbowl[['Dept', 'Weekly_Sales']].groupby('Dept').sum().sort_values(by = 'Weekly_Sales', ascending = False).head(10)
train_christmas = train[train.IsChristmas == 1]
christmas = train_christmas[['Dept', 'Weekly_Sales']].groupby('Dept').sum().sort_values(by = 'Weekly_Sales', ascending = False).head(10)
train_thanksgiving = train[train.IsThanksGiving == 1]
thanksGiving = train_thanksgiving[['Dept', 'Weekly_Sales']].groupby('Dept').sum().sort_values(by = 'Weekly_Sales', ascending = False).head(10)
train_laborday = train[train.IsLaborDay == 1]
laborDay = train_laborday[['Dept', 'Weekly_Sales']].groupby('Dept').sum().sort_values(by = 'Weekly_Sales', ascending = False).head(10)
train_easter = train[train.IsEaster == 1]
easter = train_easter[['Dept', 'Weekly_Sales']].groupby('Dept').sum().sort_values(by = 'Weekly_Sales', ascending = False).head(10)
# Checking top departments for each holiday
from IPython.core.display import HTML

def show_dataframes(listOfTables):
    ''' Accepts a list of IpyTable objects and returns a table which contains each IpyTable in a cell
    '''
    return HTML(
        '<table><tr style="background-color:white;">' + 
        ''.join(['<td>' + table.to_html(max_rows=10) + '</td>' for table in listOfTables]) +
        '</tr></table>'
    )

show_dataframes([superBowl, christmas, thanksGiving, laborDay, easter])
results_type = {}

results_type['A'] = [train[train.Type == 'A']['Weekly_Sales'].min(), 
                     train[train.Type == 'A']['Weekly_Sales'].median(), 
                     train[train.Type == 'A']['Weekly_Sales'].max()]
results_type['B'] = [train[train.Type == 'B']['Weekly_Sales'].min(), 
                     train[train.Type == 'B']['Weekly_Sales'].median(), 
                     train[train.Type == 'B']['Weekly_Sales'].max()]
results_type['C'] = [train[train.Type == 'C']['Weekly_Sales'].min(), 
                     train[train.Type == 'C']['Weekly_Sales'].median(), 
                     train[train.Type == 'C']['Weekly_Sales'].max()]

results_type = pd.DataFrame(results_type)
results_type = results_type.T
results_type
plt.figure(figsize=(10,8))
sns.boxplot(x = "Type", y = "Size", data=train)
plt.show()
#transforming Type to numerical values to use on future models
train.Type = train.Type.map({'C': 0, 'B': 1, 'A': 2})
test.Type = test.Type.map({'C': 0, 'B': 1, 'A': 2})
#Correlation between Store Type and Unemployment rate
fig, axs = plt.subplots(1,3, figsize = (10,5.5), sharey = True)
axs[0].boxplot(train[train.Type == 2]['Unemployment'])
axs[0].set_title('Type A')
axs[1].boxplot(train[train.Type == 1]['Unemployment'])
axs[1].set_title('Type B')
axs[2].boxplot(train[train.Type == 0]['Unemployment'])
axs[2].set_title('Type C')
plt.show()
size_sales = train[['Size', 'Weekly_Sales']].groupby('Size').median()
fig, axs = plt.subplots(1,1, figsize = (6,4))
axs.scatter(size_sales.index, size_sales.values)
axs.set_title('Store Size x Weekly Sales median')
plt.xlabel('Store Size')
plt.ylabel('Weekly Sales')
plt.show()
sp.stats.pearsonr(train.Size, train.Weekly_Sales), sp.stats.spearmanr(train.Size, train.Weekly_Sales)
sp.stats.pearsonr(train.Type, train.Weekly_Sales), sp.stats.spearmanr(train.Type, train.Weekly_Sales)
# Trying to understand if there is correlation between Temperature or Fuel Price with Weekly Sales
fig, axs = plt.subplots(1, 2, figsize = (15,6), sharey = True)
axs[0].scatter(train.Fuel_Price, train.Weekly_Sales, alpha = 0.05)
axs[0].set_title('Fuel Price vs Weekly Sales') 
axs[1].scatter(train.Temperature, train.Weekly_Sales, alpha = 0.05)
axs[1].set_title('Temperature vs Weekly Sales') 
plt.show()
train.drop('Fuel_Price', axis = 1, inplace = True)
test.drop('Fuel_Price', axis = 1, inplace = True)
#Evaluating how the temperature may affect departments
low_temp = train[train.Temperature <= 34]
medium_temp = train[(train.Temperature > 34) & (train.Temperature <=66)]
high_temp = train[train.Temperature > 67]

low_temp_df = low_temp[['Dept', 'Weekly_Sales']].groupby('Dept').median().sort_values(by = 'Weekly_Sales', ascending = False).head(10)
med_temp_df = medium_temp[['Dept', 'Weekly_Sales']].groupby('Dept').median().sort_values(by = 'Weekly_Sales', ascending = False).head(10)
high_temp_df = high_temp[['Dept', 'Weekly_Sales']].groupby('Dept').median().sort_values(by = 'Weekly_Sales', ascending = False).head(10)
show_dataframes([low_temp_df, med_temp_df, high_temp_df])
train[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']].isnull().sum(), train.shape
train.Week = train.Week.astype(int)
plt.figure(figsize=(25,20))
sns.heatmap(train.fillna(0).corr(), annot=True)
plt.show()
fig, axs = plt.subplots(1,2, figsize = (14,5.5))
axs[0].boxplot(train.Unemployment)
axs[0].set_title('Unemployment Boxplot')
axs[1].boxplot(train.CPI)
axs[1].set_title('CPI Boxplot')
plt.show()
good_place = train[train.Unemployment <= 6]
normal_place = train[(train.Unemployment > 6) & (train.Unemployment <= 10)]
bad_place = train[train.Unemployment > 11]

sum_good_place = good_place[['Dept', 'Weekly_Sales']].groupby('Dept').sum().sort_values(by='Weekly_Sales', ascending = False).head(10)
sum_normal_place = normal_place[['Dept', 'Weekly_Sales']].groupby('Dept').sum().sort_values(by='Weekly_Sales', ascending = False).head(10)
sum_bad_place = bad_place[['Dept', 'Weekly_Sales']].groupby('Dept').sum().sort_values(by='Weekly_Sales', ascending = False).head(10)

show_dataframes([sum_good_place, sum_normal_place, sum_bad_place])
median_goodPlace = train[train.Unemployment < 5]['Weekly_Sales'].median()
median_normalPlace = train[(train.Unemployment <= 10) & (train.Unemployment >= 5)]['Weekly_Sales'].median()
median_badPlace = train[train.Unemployment > 10]['Weekly_Sales'].median()
print('Weekly Sales median when unemployment rate is above 10%: U$', median_badPlace)
print('Weekly Sales median when unemployment rate is between 5% and 10%: U$', median_normalPlace)
print('Weekly Sales median when unemployment rate is below 5%: U$', median_goodPlace)
print('''Based on above information, unemployment rate has correlation with target values and must be considered in 
the model''')
median_cpiDeflation = train[train.CPI < 140]['Weekly_Sales'].median()
median_cpiNormal = train[(train.CPI <= 220) & (train.CPI >= 140)]['Weekly_Sales'].median()
median_cpiInflation = train[train.CPI > 220]['Weekly_Sales'].median()
print('Weekly Sales when CPI is above 220 (inflation): U$', median_cpiInflation)
print('Weekly Sales when CPI is between 140 and 220 (Normal): U$', median_cpiNormal)
print('Weekly Sales when CPI is below 140 (deflation): U$', median_cpiDeflation)
print('''Based on above information, CPI has correlation with target values and must be considered in the model''')
treino, teste = train_test_split(train, test_size = 0.2, random_state = 11)

treino = treino.copy()
teste = teste.copy()
X_treino = treino.drop(['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'IsHoliday_y', 'Year', 
                        'Month', 'Weekly_Sales', 'Date', 'Type'], axis = 1)
y_treino = treino.Weekly_Sales
X_teste = teste.drop(['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'IsHoliday_y', 'Year', 
                        'Month', 'Weekly_Sales', 'Date', 'Type'], axis = 1)
y_teste = teste.Weekly_Sales
minmax = MinMaxScaler()

X_treino.loc[:, :] = minmax.fit_transform(X_treino.loc[:,:])
X_teste.loc[:,:] = minmax.transform(X_teste.loc[:,:])
holiday_week_treino = treino.IsHoliday_y.apply(lambda x: True if x else False)
holiday_week_teste = treino.IsHoliday_y.apply(lambda x: True if x else False)

def wmae_train(y_treino, y_pred):
    sumOfWeights = 0
    sumOfErrors = 0
    
    for i in range(0, len(y_pred)):
        weight = 0
        if holiday_week_treino.values[i] == True: 
            weight = 5
        else:
            weight = 1
        
        errors = abs(y_treino.values[i] - y_pred[i])*weight
        sumOfWeights += weight 
        sumOfErrors += errors

    return sumOfErrors/sumOfWeights

def wmae_test(y_teste, y_pred):
    sumOfWeights = 0
    sumOfErrors = 0
    
    for i in range(0, len(y_pred)):
        weight = 0
        if holiday_week_teste.values[i] == True: 
            weight = 5
        else:
            weight = 1
        
        errors = abs(y_teste.values[i] - y_pred[i])*weight
        sumOfWeights += weight 
        sumOfErrors += errors

    return sumOfErrors/sumOfWeights
weight_error = make_scorer(wmae_train, greater_is_better=False)
#Once the values are re-scaled I'll not pass any hyperparameter
lin_reg = LinearRegression().fit(X_treino, y_treino)

y_pred_linreg = lin_reg.predict(X_teste)

WMAE_linreg = wmae_test(y_teste, y_pred_linreg)

r2_linreg = r2_score(y_teste, y_pred_linreg)
Results = {}


Results['Linear Regression'] = [WMAE_linreg, r2_linreg]

print("LinearRegression")
print("---------------------------------------")
print(f'WMAE = {WMAE_linreg:.10} and R-square = {r2_linreg:.4}')
plt.scatter(y_pred_linreg, y_teste.values, alpha = 0.05)
plt.xlabel("y predicted by Linear Regression")
plt.ylabel("y true")
plt.title("y pred vs y true (Linear Regression)")
plt.show()
tree = DecisionTreeRegressor(random_state = 11)

rs_tree = RandomizedSearchCV(estimator = tree,
                            param_distributions = {'max_depth': np.arange(1, 25),
                                                   'min_samples_leaf': np.arange(0.00001, 0.3, 0.00001)},
                            n_iter = 500,
                            scoring = weight_error,
                            cv = 5,
                            random_state = 11)

rs_tree.fit(X_treino, y_treino)
y_pred_rs_tree = rs_tree.predict(X_teste)

wmae_rs_tree = wmae_test(y_teste, y_pred_rs_tree)

r2_rs_tree = r2_score(y_teste, y_pred_rs_tree)

Results['Decision Tree Regressor'] = [wmae_rs_tree, r2_rs_tree]
print("Decision Tree Regressor")
print("------------------------------------------------------------")
print(f'Best parameters: {rs_tree.best_params_}')
print("------------------------------------------------------------")
print(f'WMAE = {wmae_rs_tree:.7} and R-square = {r2_rs_tree:.4}')
plt.scatter(y_pred_rs_tree, y_teste.values, alpha = 0.05)
plt.xlabel("y predicted by Decision Tree")
plt.ylabel("y true")
plt.title("y pred vs y true (Decision Tree)")
plt.show()
rf = RandomForestRegressor(random_state = 11, n_estimators = 30)

rs_rf = RandomizedSearchCV(estimator = rf,
                           param_distributions = {'max_depth': np.arange(1,30),
                                                  'min_samples_leaf': np.arange(0.0001, 0.3, 0.0001)},
                           n_iter = 30,
                           scoring = weight_error,
                           cv = 3,
                           random_state = 11)

rs_rf.fit(X_treino, y_treino)
y_pred_rs_rf = rs_rf.predict(X_teste)

wmae_rs_rf = wmae_test(y_teste, y_pred_rs_rf)

r2_rs_rf = r2_score(y_teste, y_pred_rs_rf)

Results['Random Forest'] = [wmae_rs_rf, r2_rs_rf]
print("Random Forest Regressor")
print("---------------------------------------------------------------")
print(f'Best parameters: {rs_rf.best_params_}')
print("---------------------------------------------------------------")
print(f'WMAE = {wmae_rs_rf:.7} and R-square = {r2_rs_rf:.4}')
plt.scatter(y_pred_rs_rf, y_teste.values, alpha = 0.05)
plt.xlabel("y predicted by Random Forest")
plt.ylabel("y true")
plt.title("y pred vs y true (Random Forest)")
plt.show()
tree = DecisionTreeRegressor(min_samples_leaf = 0.00043, max_depth = 22, random_state = 11)

ada_tree = AdaBoostRegressor(base_estimator = tree,
                             n_estimators = 500,
                             learning_rate = 0.05,
                             random_state = 11)

ada_tree.fit(X_treino, y_treino)
y_pred_ada_tree = ada_tree.predict(X_teste)

wmae_ada_tree = wmae_test(y_teste, y_pred_ada_tree)

r2_ada_tree = r2_score(y_teste, y_pred_ada_tree)

Results['Ada Boosting with Decision Tree'] = [wmae_ada_tree, r2_ada_tree]
print("Ada Boosting with Decision Tree")
print("---------------------------------------")
print(f'WMAE = {wmae_ada_tree:.7} and R-square = {r2_ada_tree:.4}')
plt.scatter(y_teste.values, y_pred_ada_tree, alpha = 0.2)
plt.xlabel("y true")
plt.ylabel("y predicted by Ada Boosting with Decision Tree")
plt.title("y pred vs y true (Ada Boosting)")
plt.show()
grad_boost = GradientBoostingRegressor(n_estimators = 500,
                                       learning_rate = 0.05,
                                       max_depth = 22, 
                                       min_samples_leaf = 0.00043,
                                       random_state = 11)

grad_boost.fit(X_treino, y_treino)
y_pred_grad_boost = grad_boost.predict(X_teste)

wmae_grad_boost = wmae_test(y_teste, y_pred_grad_boost)

r2_grad_boost = r2_score(y_teste, y_pred_grad_boost)

Results['Gradient Boosting'] = [wmae_grad_boost, r2_grad_boost]
print("Gradient Boosting")
print("---------------------------------------")
print(f'WMAE = {wmae_grad_boost:.7} and R-square = {r2_grad_boost:.4}')
plt.scatter(y_teste.values, y_pred_grad_boost, alpha = 0.2)
plt.ylabel("y predicted by Gradient Boosting")
plt.xlabel("y true")
plt.title("y pred vs y true (Gradient Boosting)")
plt.show()
gbm = lightgbm.LGBMRegressor()

param_space = {'num_leaves': Integer(1, 100),
               'max_depth': Integer(1, 40),
               'learning_rate': Real(0.05, 0.25),
               'n_estimators': Integer(50, 500),
               'min_split_gain': Real(0.001, 0.3),
               'min_child_samples': Integer(1, 50)}

bayes_gbm = BayesSearchCV(estimator = gbm,
                          search_spaces = param_space,
                          n_iter = 200,
                          cv = 3,
                          scoring = weight_error, 
                          random_state = 11)

bayes_gbm.fit(X_treino, y_treino)
y_pred_gbm_bs = bayes_gbm.predict(X_teste)

wmae_gbm = wmae_test(y_teste, y_pred_gbm_bs)

r2_gbm = r2_score(y_teste, y_pred_gbm_bs)

Results['LightGBM Regressor'] = [wmae_gbm, r2_gbm]
print("LightGBM Regressor")
print("---------------------------------------")
print(f'WMAE = {wmae_gbm:.7} and R-square = {r2_voting:.4}')
plt.scatter(y_teste.values, y_pred_gbm_bs, alpha = 0.2)
plt.ylabel("y predicted by LightGBM Regressor")
plt.xlabel("y true")
plt.title("y pred vs y true (LightGBM Regressor)")
plt.show()
Results = pd.DataFrame(Results)
Results = Results.T
Results
plt.scatter(y_teste.values, y_pred_ada_tree, alpha = 0.2, color = 'blue', label = 'Ada Boosting')
plt.scatter(y_teste.values, y_pred_grad_boost, alpha = 0.2, color = 'red', label = 'Gradient Boosting')
plt.scatter(y_teste.values, y_pred_gbm_bs, alpha = 0.2, color = 'yellow', label = 'LightGBM Regressor')
plt.ylabel("y predicted")
plt.xlabel("y true")
plt.title("y pred vs y true")
plt.legend()
plt.show()
def prepare_data(dataframe_1, dataframe_2, dataframe_3, var, var2, how):
    '''This function prepare the dataframe to proceed direct to hold out
    dataframe_1: Must be train dataframe
    dataframe_2: Must be stores dataframe
    dataframe_3: Must be features dataframe
    var: Feature to be used to merge first and second dataframe (string or list of strings)
    var2: Feature to be used to merge first and third dataframe (string or list of strings)
    how: How you want to join dataframes'''

    def merge(dataframe_1, dataframe_2, dataframe_3, var, var2, how):
        dataframe_1 = pd.merge(dataframe_1, dataframe_2, on = var, how = how)
        dataframe_1 = pd.merge(dataframe_1, dataframe_3, on = var2, how = how).drop('IsHoliday_x', axis = 1)
        dataframe_1.columns = ['Store', 'Dept', 'Date', 'Weekly_Sales', 'Type', 'Size', 'Temperature',
       'Fuel_Price', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4',
       'MarkDown5', 'CPI', 'Unemployment', 'IsHoliday']
        return dataframe_1

    def split_date(dataframe_1):
        dataframe_1['Date'] = pd.to_datetime(dataframe_1['Date'])
        dataframe_1['Year'], dataframe_1['Month'], dataframe_1['Week'] = dataframe_1['Date'].dt.year, dataframe_1['Date'].dt.month, dataframe_1['Date'].dt.strftime('%V')
        return dataframe_1

    def break_down_holiday(dataframe_1):
        dataframe_1['IsChristmas'] = dataframe_1['IsHoliday'].apply(lambda x: 1 if (x == '51' or x == '52') else 0)
        dataframe_1['IsSuperBowl'] = dataframe_1['IsHoliday'].apply(lambda x: 1 if x == '06' else 0)
        dataframe_1['IsLaborDay'] = dataframe_1['IsHoliday'].apply(lambda x: 1 if x == '36' else 0)
        dataframe_1['IsThanksGiving'] = dataframe_1['IsHoliday'].apply(lambda x: 1 if x == '47' else 0)
        return dataframe_1
    
    def create_easter(dataframe_1):
        dataframe_1.loc[(dataframe_1.Year==2010) & (dataframe_1.Week=='13'), 'IsEaster'] = True
        dataframe_1.loc[(dataframe_1.Year==2011) & (dataframe_1.Week=='16'), 'IsEaster'] = True
        dataframe_1.loc[(dataframe_1.Year==2012) & (dataframe_1.Week=='14'), 'IsEaster'] = True
        dataframe_1['IsEaster'] = dataframe_1.IsEaster.apply(lambda x: 1 if x == True else 0)
        return dataframe_1
        
    dataframe_1 = merge(dataframe_1, dataframe_2, dataframe_3, var, var2, how)
    dataframe_1 = split_date(dataframe_1)
    dataframe_1 = break_down_holiday(dataframe_1)
    dataframe_1 = create_easter(dataframe_1)
    return dataframe_1

def wmae_test(y_teste, y_pred):
    sumOfWeights = 0
    sumOfErrors = 0
    
    for i in range(0, len(y_pred)):
        weight = 0
        if holiday_week_teste.values[i] == True: 
            weight = 5
        else:
            weight = 1
        
        errors = abs(y_teste.values[i] - y_pred[i])*weight
        sumOfWeights += weight 
        sumOfErrors += errors

    return sumOfErrors/sumOfWeights
train_pipe = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/train.csv.zip')
features_pipe = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/features.csv.zip')
stores_pipe = pd.read_csv('/kaggle/input/walmart-recruiting-store-sales-forecasting/stores.csv')
train_pipe = prepare_data(train_pipe, stores_pipe, features_pipe, 'Store', ['Store', 'Date'], 'left')
treino_pipe, teste_pipe = train_test_split(train_pipe, test_size = 0.2, random_state = 11)

treino_pipe = treino_pipe.copy()
teste_pipe = teste_pipe.copy()

X_treino_pipe = treino_pipe.drop(['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'IsHoliday', 'Year', 
                        'Month', 'Weekly_Sales', 'Date', 'Fuel_Price', 'Type'], axis = 1)
y_treino_pipe = treino_pipe.Weekly_Sales
X_teste_pipe = teste_pipe.drop(['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5', 'IsHoliday', 'Year', 
                        'Month', 'Weekly_Sales', 'Date', 'Fuel_Price', 'Type'], axis = 1)
y_teste_pipe = teste_pipe.Weekly_Sales

holiday_week_teste = treino_pipe.IsHoliday.apply(lambda x: True if x else False)
X_treino_pipe.head()
pipe_gbm = Pipeline(steps = [
    ('rescale', MinMaxScaler()),
    ('modelo', lightgbm.LGBMRegressor(num_leaves = 100,
                                      max_depth = 14,
                                      learning_rate = 0.2396915023230759,
                                      n_estimators = 482,
                                      min_split_gain = 0.17147656915295872,
                                      min_child_samples = 12))
])

pipe_gbm.fit(X_treino_pipe, y_treino_pipe)
pd.to_pickle(pipe_gbm, 'pipe_gbm.pkl')
