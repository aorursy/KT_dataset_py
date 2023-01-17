## load libraries

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from sklearn.feature_selection import SelectKBest, f_regression

from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge

from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.neural_network import MLPRegressor

from sklearn.model_selection import KFold, cross_val_score, GridSearchCV

from sklearn.pipeline import Pipeline

import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

plt.rcParams["figure.figsize"] = (20,10)



## load dataset

url = '../input/avocado-prices/avocado.csv'

df = pd.read_csv(url, delimiter=',')

df.head()
# Descriptive statistics

## Dimensions

print("Dataset {0} loaded with {1} rows and {2} columns \n".format('avocado', df.shape[0], df.shape[1]))



## Indentification NaN

isNan = df.isnull().values.any()

print("Dataset containing NaN values: {0} \n".format(isNan))



## Identification duplicates

isDuplicate = df.duplicated(subset=['Date','year', 'region', 'type']).any() 

print("Dataset containing duplicate values: {0} \n".format(isDuplicate))



## Attributes datatype

print("Dataset column initial datatype: \n {0} \n".format(df.dtypes))

list_index_col_type_object = df.select_dtypes('object').columns



## Statistical summary (numerical attribute)

print("Dataset numerical attributes statistical distribution: \n {0} \n".format(df.describe()))



## Classes distribution (categorical attributes)

print("Dataset categorical attributes distribution: \n {0} {1} \n".format(df.groupby('region').size(), df.groupby('type').size()))
# Data visualization

## Univariate plots

df.plot(kind='density', subplots=True, layout=(4,4), sharex=False)

df.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)

plt.show()
## Multivariate plot

sns.heatmap(df.corr(), cmap='viridis')

plt.show()
# Data Cleaning

## Remove attribute Unnamed: 0

df = df.drop(['Unnamed: 0'], axis=1)



## Label categorical data

def convert_string_into_numeric(dataframe, columns):

    lookup = dict()

    for column in columns:

        index_column = df.columns.get_loc(column)

        string_values = dataframe.iloc[:,index_column].unique()

        lookup_column = dict()

        for i, string_value in enumerate(string_values):

            lookup_column[string_value] = i

            index_string_values = np.where(dataframe.iloc[:,index_column] == string_value)[0]

            dataframe.iloc[index_string_values,index_column] = i

        lookup[column] = lookup_column

    return lookup



lookup = convert_string_into_numeric(df, ['type', 'region'])



# Feature Selection

## Add attributes from date and remove this column

df['month'] = pd.to_datetime(df["Date"]).dt.month

df['day'] = pd.to_datetime(df["Date"]).dt.day

del df['Date']



## Uniformize datatype

df = df.apply(pd.to_numeric, downcast='float')



## Apply feature selection

X = df.drop(['AveragePrice'], 1)

y = df['AveragePrice']

selector = SelectKBest(f_regression, k=11).fit(X,y)

selected_columns = selector.get_support(indices=True)

X = X.iloc[:,selected_columns]
# Split dataset

X_train, X_validation, y_train, y_validation = train_test_split(X, y.values, test_size=0.33, random_state=42, shuffle=True)



# Check algorithms

## Testing options

number_folds = 5

seed = 42

scoring = 'neg_mean_squared_error'



## Define testing workflow and select model

models = []

models.append(("ScaledLR", Pipeline([('Scaler', MinMaxScaler()), ('LR', LinearRegression())])))

models.append(("ScaledRIDGE", Pipeline([('Scaler', MinMaxScaler()), ('RIDGE', Ridge())])))

models.append(("ScaledLASSO", Pipeline([('Scaler', MinMaxScaler()), ('LASSO', Lasso())])))

models.append(("ScaledEN", Pipeline([('Scaler', MinMaxScaler()), ('EN', ElasticNet())])))

models.append(("ScaledSVR", Pipeline([('Scaler', MinMaxScaler()), ('SVR', SVR())])))

models.append(("ScaledKNN", Pipeline([('Scaler', MinMaxScaler()), ('KNN', KNeighborsRegressor())])))

models.append(("ScaledCART", Pipeline([('Scaler', MinMaxScaler()), ('CART', DecisionTreeRegressor())])))

models.append(("ScaledNN", Pipeline([('Scaler', MinMaxScaler()), ('NN', MLPRegressor(random_state=1, max_iter=500))])))





results = []

names = []

for name, model in models:

    kfold = KFold(n_splits = number_folds, random_state = seed, shuffle=True)

    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    print("{0}: {1} ({2})".format(name, cv_results.mean(), cv_results.std()))



# Distribution performance

fig, ax = plt.subplots()

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
# Model tuning 

## Normalize

scaler = MinMaxScaler().fit(X_train)

X_normalized = scaler.transform(X_train)



## Select tuning grid parameters

n_neighbors_parameter = np.array([1,2,3,5,7,8,10,15,20])

p_parameter = np.array([1,2])

weights_parameter = np.array(['uniform', 'distance'])

grid_parameters = dict(n_neighbors=n_neighbors_parameter,p=p_parameter, weights=weights_parameter)



## Apply on the model

model = KNeighborsRegressor()

kfold = KFold(n_splits = number_folds, random_state = seed, shuffle=True)

grid = GridSearchCV(estimator=model, param_grid=grid_parameters, scoring=scoring, cv=kfold)

grid_results = grid.fit(X_normalized, y_train)

print("Best score {0} obtain with {1}".format(grid_results.best_score_, grid_results.best_params_))
# Ensemble methods

ensembles = []

ensembles.append(("ScaledRFR", Pipeline([('Scaler', MinMaxScaler()), ('RFR', RandomForestRegressor())])))

ensembles.append(("ScaledABR", Pipeline([('Scaler', MinMaxScaler()), ('ABR', AdaBoostRegressor())])))

ensembles.append(("ScaledGBR", Pipeline([('Scaler', MinMaxScaler()), ('GBR', GradientBoostingRegressor())])))

ensembles.append(("ScaledETR", Pipeline([('Scaler', MinMaxScaler()), ('ETR', ExtraTreesRegressor())])))



results = []

names = []

for name, ensemble in ensembles:

    kfold = KFold(n_splits = number_folds, random_state = seed, shuffle=True)

    cv_results = cross_val_score(ensemble, X_train, y_train, cv=kfold, scoring=scoring)

    results.append(cv_results)

    names.append(name)

    print("{0}: {1} ({2})".format(name, cv_results.mean(), cv_results.std()))



# Distribution performance

fig, ax = plt.subplots()

plt.boxplot(results)

ax.set_xticklabels(names)

plt.show()
# Tuning emsemble method

## Normalize

scaler = MinMaxScaler().fit(X_train)

X_normalized = scaler.transform(X_train)



## Select tuning grid parameters

n_estimators_parameter = np.array([100, 200, 300])

grid_parameters = dict(n_estimators=n_estimators_parameter)



## Apply on the model

ensemble = ExtraTreesRegressor()

kfold = KFold(n_splits = number_folds, random_state = seed, shuffle=True)

grid = GridSearchCV(estimator=ensemble, param_grid=grid_parameters, scoring=scoring, cv=kfold)

grid_results = grid.fit(X_normalized, y_train)

print("Best score {0} obtain with {1}".format(grid_results.best_score_, grid_results.best_params_))
# fit optimal model

scaler = MinMaxScaler().fit(X_train)

X_normalized = scaler.transform(X_train)

model = ExtraTreesRegressor(n_estimators=200)

model.fit(X_normalized, y_train)



# evaluate performance on validation dataset

X_validation_normalized = scaler.transform(X_validation)

y_prediction = model.predict(X_validation_normalized)

print("RMSE model: {0}".format(mean_squared_error(y_validation, y_prediction)))
# If needed save the model

#from pickle import dump

#from pickle import load

# save

#filename = 'model.sav'

#dump(model, open(filename, 'wb'))

# load

#load_model = load(open(filename, 'rb'))

#X_validation_normalized = scaler.transform(X_validation)

#y_prediction = load_model.predict(X_validation_normalized)

#print(mean_squared_error(y_validation, y_prediction))