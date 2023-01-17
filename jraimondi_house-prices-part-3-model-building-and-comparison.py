import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

house_data = pd.read_csv('../input/train.csv')
def drop_unneeded_columns(house_data):
    # drop unbalanced variables
    dropped_data = house_data.drop(['Street', 'Alley', 'LandContour', 'Utilities', 'Condition2', 'RoofMatl', 'Heating', 'LowQualFinSF', 'GarageCond', '3SsnPorch', 'PoolArea', 'PoolQC', 'MiscFeature', 'MiscVal'], axis=1)
    # drop variables without much prediction value
    dropped_data = dropped_data.drop(['BsmtFinType2', 'Functional', 'MoSold', 'YrSold'], axis=1)
    # drop Id Column: it's already the index
    dropped_data = dropped_data.drop(['Id'], axis=1)
    return dropped_data

dropped_data = drop_unneeded_columns(house_data)
print("{} → {}".format(house_data.shape, dropped_data.shape))
# Removing outliers -- training set only

rows_to_drop = (dropped_data['LotFrontage'] > 150)
rows_to_drop = rows_to_drop | (dropped_data['LotArea'] > 50000)
rows_to_drop = rows_to_drop | ((dropped_data['MSZoning'] == 'RL') & (dropped_data['SalePrice'] > 400000))
rows_to_drop = rows_to_drop | (((dropped_data['Neighborhood'] == 'NoRidge') | (dropped_data['Neighborhood'] == 'NridgHt')) & (dropped_data['SalePrice'] > 575000))
rows_to_drop = rows_to_drop | (((dropped_data['OverallCond'] == 5) | (dropped_data['OverallCond'] == 6)) & (dropped_data['SalePrice']) > 500000)
rows_to_drop = rows_to_drop | ((dropped_data['OverallQual'] == 10) & (dropped_data['SalePrice'] < 200000))
rows_to_drop = rows_to_drop | ((dropped_data['YearBuilt'] < 2000) & (dropped_data['SalePrice'] > 600000))
rows_to_drop = rows_to_drop | ((dropped_data['YearRemodAdd'] < 2000) & (dropped_data['SalePrice'] > 600000))
rows_to_drop = rows_to_drop | (dropped_data['TotalBsmtSF'] > 3000)
rows_to_drop = rows_to_drop | (dropped_data['1stFlrSF'] > 3000)
rows_to_drop = rows_to_drop | (dropped_data['2ndFlrSF'] > 1575)
rows_to_drop = rows_to_drop | (dropped_data['GrLivArea'] > 4000)
rows_to_drop = rows_to_drop | (dropped_data['BedroomAbvGr'] > 5)
rows_to_drop = rows_to_drop | (dropped_data['KitchenAbvGr'] == 3)
rows_to_drop = rows_to_drop | (dropped_data['TotRmsAbvGrd'] > 10)
rows_to_drop = rows_to_drop | (dropped_data['Fireplaces'] > 2)
rows_to_drop = rows_to_drop | ((dropped_data['GarageYrBlt'] > 1980) & (dropped_data['SalePrice'] > 600000))
rows_to_drop = rows_to_drop | (dropped_data['GarageCars'] > 3)
rows_to_drop = rows_to_drop | (dropped_data['GarageArea'] > 1200)
rows_to_drop = rows_to_drop | (dropped_data['OpenPorchSF'] > 400)

filtered_data = dropped_data.copy()
filtered_data.drop(filtered_data[rows_to_drop].index, inplace=True)

print("{} → {}".format(dropped_data.shape, filtered_data.shape))
def set_value_to_na(column, dataframe, current_value=0, log=True):
    if (log):
        num_affected_rows = dataframe[column].value_counts()[current_value]
        print('Setting {:,} values to NaN (were all {} for column {}).'.format(num_affected_rows, current_value, column))
    dataframe[column].replace(to_replace=current_value, value=np.nan, inplace=True)

def set_dummy_values_to_na(dataframe, log=True):
    set_value_to_na('YearRemodAdd', dataframe, current_value=1950, log=log)
    set_value_to_na('MasVnrArea', dataframe, log=log)
    set_value_to_na('BsmtFinSF1', dataframe, log=log)
    set_value_to_na('BsmtFinSF2', dataframe, log=log)
    set_value_to_na('BsmtUnfSF', dataframe, log=log)
    set_value_to_na('TotalBsmtSF', dataframe, log=log)
    set_value_to_na('2ndFlrSF', dataframe, log=log)
    set_value_to_na('GarageArea', dataframe, log=log)
    set_value_to_na('WoodDeckSF', dataframe, log=log)
    set_value_to_na('OpenPorchSF', dataframe, log=log)
    set_value_to_na('EnclosedPorch', dataframe, log=log)
    set_value_to_na('ScreenPorch', dataframe, log=log)
# remove the values that are meaningless, let's set them to np.nan so we can later on deal with them in the right way
data_with_nas = filtered_data.copy()

set_dummy_values_to_na(data_with_nas)

print("{} → {}".format(filtered_data.shape, data_with_nas.shape))
from sklearn.preprocessing import PolynomialFeatures

def detect_polynomial_generators(dataframe):
    ordinal_variables = ['LandSlope', 'Condition1', 'BldgType', 'HouseStyle', 'RoofStyle', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'PavedDrive', 'Fence', 'SaleType', 'SaleCondition']
    polynomial_generators = {}
    for ordinal_variable in ordinal_variables:
        #print('Processing {}'.format(ordinal_variable))
        polynomial_generators[ordinal_variable] = {}

        # identify mapping values from string to number
        # TODO Replace this with known possible values from the description of each categorical variable
        unique_values = dataframe[ordinal_variable].unique()
        unique_values = np.append(unique_values, np.nan)
        mapped_values = { value: index + 1 for (index, value) in enumerate(unique_values) }
        polynomial_generators[ordinal_variable]['values'] = mapped_values

        degree = unique_values.size - 1
        replaced_dataframe = dataframe.replace({ ordinal_variable: mapped_values })
        pf = PolynomialFeatures(degree=degree, include_bias = False)
        pf = pf.fit(replaced_dataframe[ordinal_variable].values.reshape(-1, 1))
        polynomial_generators[ordinal_variable]['generator'] = pf

    return polynomial_generators

def generate_polynomials(dataframe, polynomial_generators):
    data_with_polynomials = dataframe.copy()
    for ordinal_variable in polynomial_generators:
        mapped_values = polynomial_generators[ordinal_variable]['values']
        pf = polynomial_generators[ordinal_variable]['generator']
        
        # replace categorical values with numeric values
        data_with_polynomials.replace({ ordinal_variable: mapped_values }, inplace=True)

        # transform them into polynomial features
        new_features = pf.transform(data_with_polynomials[ordinal_variable].values.reshape(-1, 1))
        new_columns = [ '{}_deg{}'.format(ordinal_variable, d) for d in range(pf.degree) ]
        new_dataframe = pd.DataFrame(new_features, columns=new_columns, index=data_with_polynomials.index)
        data_with_polynomials = pd.concat([data_with_polynomials, new_dataframe], axis=1, join_axes=[data_with_polynomials.index])
        data_with_polynomials.drop(columns=[ordinal_variable], inplace=True)
    return data_with_polynomials
data_with_polynomials = data_with_nas.copy()

generators = detect_polynomial_generators(data_with_polynomials)
data_with_polynomials = generate_polynomials(data_with_polynomials, generators)

print("{} → {}".format(data_with_nas.shape, data_with_polynomials.shape))
# Generate columns for categorical variables
def generate_dummies(dataframe):
    return pd.get_dummies(dataframe, drop_first=True)
data_with_dummies = generate_dummies(data_with_polynomials)

print("{} → {}".format(data_with_polynomials.shape, data_with_dummies.shape))
# Missing data: replace with mean but create a "was_missing" column alongside
columns_with_nas = list(col for col in data_with_dummies.columns if data_with_dummies[col].isnull().any())
print('Columns to imput: {}'.format(columns_with_nas))
from sklearn.preprocessing import Imputer

def imput_data(dataframe):
    imputed_data = dataframe.copy()
    for col in columns_with_nas:
        imputed_data[col + '_was_missing'] = imputed_data[col].isnull()

    imputer = Imputer()
    imputed_np = imputer.fit_transform(imputed_data)
    imputed_data = pd.DataFrame(imputed_np, columns=imputed_data.columns)
    
    return imputed_data
imputed_data = imput_data(data_with_dummies)

print("{} → {}".format(data_with_dummies.shape, imputed_data.shape))
model_data = imputed_data.copy()

model_x = model_data.drop(['SalePrice'], axis=1)
model_y = model_data[['SalePrice']]
model_x.head()
model_y.head()
# Scaling
from sklearn.preprocessing import StandardScaler

x_scaler = StandardScaler()
scaled_model = x_scaler.fit_transform(model_x)
scaled_model_x = pd.DataFrame(scaled_model, columns=model_x.columns)

y_scaler = StandardScaler()
scaled_model = y_scaler.fit_transform(model_y)
scaled_model_y = pd.DataFrame(scaled_model, columns=model_y.columns)
def unscale(scaled_predictions_or_targets):
    in_range_predictions_or_targets = np.maximum(scaled_predictions_or_targets, -1)
    return y_scaler.inverse_transform(in_range_predictions_or_targets)
def kaggle_evaluation(scaled_targets, scaled_predictions):
    # this reverse-scales the values, performs logarithms on the results and returns the RMSE
    
    # reverse-scaling
    targets = unscale(scaled_targets.flatten())
    predictions = unscale(scaled_predictions.flatten())
    
    predictions = np.log(predictions)
    targets = np.log(targets)
    return np.sqrt(((predictions - targets) ** 2).mean())
import scipy.stats as stats

def pearson_evaluation(scaled_targets, scaled_predictions):
    targets = unscale(scaled_targets.flatten())
    predictions = unscale(scaled_predictions.flatten())
    correlation, p_value = stats.pearsonr(targets, predictions)
    return (correlation, p_value)
from sklearn.model_selection import KFold

def fit_and_test(model, model_x=scaled_model_x, model_y=scaled_model_y):
    n_splits = 15
    kf = KFold(n_splits=n_splits, shuffle=True)
    split_number = 0
    for train_idx, test_idx in kf.split(scaled_model_x):
        train_x = model_x.iloc[train_idx]
        train_y = model_y.iloc[train_idx].values
        test_x = model_x.iloc[test_idx]
        test_y = model_y.iloc[test_idx].values

        model.fit(train_x, train_y)
        predictions = model.predict(test_x)
        all_predictions = model.predict(model_x)
        split_number = split_number + 1
        print('{:02d}/{:02d} (Partial) RMSE (Logs): {:,.5f} / (Total) RMSE (Logs): {:,.5f}'.format(
            split_number,
            n_splits,
            kaggle_evaluation(test_y, predictions),
            kaggle_evaluation(model_y['SalePrice'].values, all_predictions)
        ))

    predictions = model.predict(model_x)
    print('General RMSE (Logs): {:,.5f}'.format(kaggle_evaluation(model_y['SalePrice'].values, predictions)))
    correlation, p_value = pearson_evaluation(model_y['SalePrice'].values, predictions)
    print('Pearson\'s-R = {:.5f}, p-value = {}'.format(correlation, p_value))
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

lm = linear_model.LinearRegression(normalize=True)
lm_name = 'Linear Regression'
fit_and_test(lm)
from sklearn import tree

dt = tree.DecisionTreeRegressor()
dt_name = 'Decision Trees'
fit_and_test(dt)
import warnings
warnings.filterwarnings('ignore')
from sklearn import ensemble

rf = ensemble.RandomForestRegressor()
rf_name = 'Random Forests'
fit_and_test(rf)
from sklearn import svm

svm_rbf = svm.SVR(kernel='rbf')
svm_rbf_name = 'SVM: RBF kernel'
fit_and_test(svm_rbf)
svm_linear = svm.SVR(kernel='linear')
svm_linear_name = 'SVM: Linear kernel'
fit_and_test(svm_linear)
svm_poly = svm.SVR(kernel='poly')
svm_poly_name = 'SVM: Polynomial kernel'
fit_and_test(svm_poly)
svm_sigmoid = svm.SVR(kernel='sigmoid')
svm_sigmoid_name = 'SVM: Sigmoid kernel'
fit_and_test(svm_sigmoid)
from sklearn import ensemble
gb = ensemble.GradientBoostingRegressor(n_estimators=1000, learning_rate=0.5)
gb_name = 'Gradient Boosting'
fit_and_test(gb)
from sklearn.neural_network import MLPRegressor
nn = MLPRegressor(learning_rate='adaptive', learning_rate_init=0.01, max_iter=500, shuffle=True, hidden_layer_sizes=(100, 50))
nn_name = 'Neural Networks'
fit_and_test(nn)
models = [lm, dt, rf, svm_rbf, svm_linear, svm_poly, svm_sigmoid, gb, nn]
models_names = [lm_name, dt_name, rf_name, svm_rbf_name, svm_linear_name, svm_poly_name, svm_sigmoid_name, gb_name, nn_name]
target_values = scaled_model_y['SalePrice'].values
models_predictions = [model.predict(scaled_model_x) for model in models]
models_rmse = [kaggle_evaluation(target_values, prediction) for prediction in models_predictions]

models_performance = {'Model': models_names, 'Performance': models_rmse}
models_performance = pd.DataFrame(models_performance, columns=['Model', 'Performance'])
models_performance.sort_values(by='Performance')
class StackedModel:
    def __init__(self, models):
        self.models = models
        
    def predict(self, X):
        predictions = [model.predict(X) for model in self.models]
        return np.mean(predictions, axis=0)
best_3_models = models_performance.sort_values(by='Performance')[0:3]
best_3_models = [models[i] for i in best_3_models.index]

best_3 = StackedModel(best_3_models)
models.append(best_3)
models_names.append('Best 3 Averaged')
target_values = scaled_model_y['SalePrice'].values
models_predictions = [model.predict(scaled_model_x) for model in models]
models_rmse = [kaggle_evaluation(target_values, prediction) for prediction in models_predictions]

models_performance = {'Model': models_names, 'Performance': models_rmse}
models_performance = pd.DataFrame(models_performance, columns=['Model', 'Performance'])
models_performance.sort_values(by='Performance')
selected_model_index = models_performance.Performance.idxmin()
selected_model = models[selected_model_index]
print('The final selected model was {}.'.format(models_names[selected_model_index]))
predictions = selected_model.predict(scaled_model_x)
predictions = unscale(predictions).flatten()

unscaled_targets = unscale(target_values)

plt.figure(figsize=(20, 20))
plt.scatter(unscaled_targets, unscaled_targets - predictions, marker='.')
plt.plot(target_values, np.zeros(len(target_values)), c='grey')
plt.figure(figsize=(20, 20))
stats.probplot(target_values - predictions, plot=plt)
# load our test dataset to predict values, and perform the same transformations we did with train data set
data_to_predict = pd.read_csv('../input/test.csv')

data_to_predict.shape
#data_to_predict.set_index('Id')

dropped_data = drop_unneeded_columns(data_to_predict)
print("{} → {}".format(data_to_predict.shape, dropped_data.shape))

# Can't drop rows! We want to predict them all!
#rows_to_drop = ... 

filtered_data = dropped_data.copy()
data_with_nas = filtered_data.copy()
set_dummy_values_to_na(data_with_nas, log=False)
data_with_polynomials = generate_polynomials(data_with_nas.copy(), generators)
print("{} → {}".format(data_with_nas.shape, data_with_polynomials.shape))

data_with_dummies = generate_dummies(data_with_polynomials)
print("{} → {}".format(data_with_polynomials.shape, data_with_dummies.shape))

imputed_data = imput_data(data_with_dummies)
print("{} → {}".format(data_with_dummies.shape, imputed_data.shape))

model_data = imputed_data.copy()

model_x = model_data # no SalePrice 

x_scaler = StandardScaler()
scaled_model = x_scaler.fit_transform(model_x)
scaled_model_x = pd.DataFrame(scaled_model, columns=model_x.columns)
results = selected_model.predict(scaled_model_x)
results = unscale(results)
output_dataframe = pd.concat([data_to_predict['Id'], pd.Series(results)], axis=1, keys=['Id', 'SalePrice'])
output_dataframe.columns
output_dataframe.tail()
output_dataframe.to_csv('predicted_results.csv', index=False)