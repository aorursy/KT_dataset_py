import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
housing = pd.read_csv("../input/housing-california/housing.csv")

housing.head()
housing.info()
housing['ocean_proximity'].value_counts()
housing.describe()
housing.hist(bins=50, figsize=(20,15))

plt.show()
# Splitting train and test datasets

from sklearn.model_selection import train_test_split



train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
#Checking median icnome as category



f = plt.figure(figsize=(10,3))

ax1 = f.add_subplot(121)

ax2 = f.add_subplot(122)



housing["median_income"].hist(ax=ax1, bins=30)

ax1.title.set_text('Median income')

ax1.set_xlabel('Median income')

ax1.set_ylabel('Median income count')





#Creating category - income_cat - with values of median_income to 5 



housing['income_cat'] = np.ceil(housing['median_income']/1.5)

housing['income_cat'].where(housing['income_cat'] <5, 5.0, inplace=True)



housing['income_cat'].hist(bins=20, ax=ax2)

ax2.title.set_text('Median income')

ax2.set_xlabel('Median income')

ax2.set_ylabel('Median income count')
# Stratified sampling on median_income



from sklearn.model_selection import StratifiedShuffleSplit



split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)



for train_index, test_index in split.split(housing, housing['income_cat']):

    strat_train_set = housing.loc[train_index]

    strat_test_set = housing.loc[test_index]
# Propotions of income_cat



housing['income_cat'].value_counts() /len(housing)
# Droppping column income_cat in strain



for set_ in (strat_train_set, strat_test_set):

    set_.drop('income_cat', axis=1, inplace=True)
# Geographical plot



housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1,

            s=housing['population']/100, label='Population', figsize=(10,7),

            c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)



plt.legend()
# Checking correlations - Pearson



corr_matrix = housing.corr()
housing.head()
corr_matrix['median_house_value'].sort_values(ascending=False)



# The biggest correlation is between median_house_value and median_income
housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1, figsize=(10,7))



# We can see several points for horizontal median house values. We should clean this to protect further algorithm.
#Creating additional atributes



housing['rooms_per_family'] = housing['total_rooms'] / housing['households']

housing['bedrooms_per_rooms'] = housing['total_bedrooms'] / housing['total_rooms']

housing['populations_per_family'] = housing['population'] / housing['households']
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)
# Preparing data into machine learning model





housing = train_set.drop('median_house_value', axis=1)

housing_labels = train_set['median_house_value'].copy()
housing_labels.head() # only median house value
housing.isnull().sum()
from sklearn.impute import SimpleImputer 



# Dealing with missing values - median values

imputer = SimpleImputer(strategy='median')



imputer.fit(housing.iloc[:, 4:5])

housing.iloc[:,4:5] = imputer.transform(housing.iloc[:,4:5])
# Testing for missing values - total_bedrooms



housing.isnull().sum()
# Dealing with missing values further. Ocean_proximity is string column



missing_value = [housing.latitude, housing.housing_median_age, housing.total_rooms,

                housing.population, housing.households, housing.median_income,

                housing.median_house_value]
def missing_values(column):

    mean = column.mean()

    column.fillna(mean, inplace=True)

    column.isnull().sum()

    

for col in missing_value:

    missing_values(col)
housing.isnull().sum()
housing.ocean_proximity.fillna('<1H OCEAN', inplace=True)
# Because in ocean_proximity values are distributed in this way i decide to fill missing value with 1H OCEAN

housing.ocean_proximity.hist()



housing.isnull().sum()
# Changing string values into numbers - ocean_proximity

#Changing string values into cathegorical numbers



from sklearn.preprocessing import LabelEncoder



label_encoder=LabelEncoder()

housing['ocean_proximity'] = label_encoder.fit_transform(housing['ocean_proximity'])

housing['ocean_proximity'].value_counts()
# The aim of the problem is to predit median_house_value



# Preparing predicting columns and rest columns

housing_ind = housing.drop('median_house_value', axis=1)

housing_dep = housing['median_house_value']
# Splitting train and test datasets



X_train, X_test, y_train, y_test = train_test_split(housing_ind, housing_dep,

                                                   test_size=0.2, random_state=42)
# Standarizing data - standard scaler



from sklearn.preprocessing import StandardScaler



scaler = StandardScaler()



X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()



# Fitting model into train set - X and y

lin_reg.fit(X_train, y_train)



print('Intercept: ' + str(lin_reg.intercept_))

print('Coefficients: ' +str(lin_reg.coef_))
# Predicting y on test data



y_pred = lin_reg.predict(X_test)
# We can compare test values and predicted ones



print(y_pred[:5])

print(y_test[:5])
# RMSE for linear model



from sklearn.metrics import mean_squared_error

lin_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(lin_rmse)



# Value of RMSE = 69 597$ is very high. The reason is that probably model was not trained enough

# Features are not giving apropriate information or model is not much good
# Additiotnally we can see this on plot



pred = pd.DataFrame({'Predicted':y_pred,'Actual':y_test})

fig= plt.figure(figsize=(16,8))

pred = pred.reset_index()

pred = pred.drop(['index'],axis=1)

plt.plot(pred[:50])

plt.legend(['Actual','Predicted'])
from sklearn.tree import DecisionTreeRegressor



t_reg = DecisionTreeRegressor()

t_reg.fit(X_train, y_train)



treg_y_pred = t_reg.predict(X_test)
# We can compare test values and predicted ones



print(treg_y_pred[:5])

print(y_test[:5])
tree_rmse = np.sqrt(mean_squared_error(y_test, treg_y_pred))

print(tree_rmse)
# Model id better than linear regression but still not satisfying

# K-fold cross-validation



from sklearn.model_selection import cross_val_score



scores = cross_val_score(t_reg, X_train, y_train, 

                        scoring='neg_mean_squared_error', cv=10)



tree_rmse_scores = np.sqrt(-scores)
# Checking results



def display_scores(scores):

    print('Results: ', scores)

    print('Mean: ', scores.mean())

    print('Std variation: ', scores.std())

    

display_scores(tree_rmse_scores)
# Tree model is also not good idea to predict values
# Cross validation for linear regression



lin_scores = cross_val_score(lin_reg, X_train, y_train,

                            scoring='neg_mean_squared_error', cv=10)



lin_rmse_scores = np.sqrt(-lin_scores)

display_scores(lin_rmse_scores)
from sklearn.ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor()

forest_reg.fit(X_train, y_train)



forest_reg_y_pred = forest_reg.predict(X_test)



print(forest_reg_y_pred[:5])

print(y_test[:5])
forest_rmse = np.sqrt(mean_squared_error(y_test, forest_reg_y_pred))

print(forest_rmse)
forest_scores = cross_val_score(forest_reg, X_train, y_train, 

                        scoring='neg_mean_squared_error', cv=10)



forest_rmse_scores = np.sqrt(-forest_scores)



display_scores(forest_rmse_scores)



# Model is over-trained
#  GridSearchCV



from sklearn.model_selection import GridSearchCV



param_grid = [

    {'n_estimators': [3,10,30], 'max_features': [2,4,6,8]},

    {'bootstrap': [False], 'n_estimators': [3,10], 'max_features': [2,3,4]},

]



forest_reg=RandomForestRegressor()



grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')



grid_search.fit(X_train, y_train)
grid_search.best_params_