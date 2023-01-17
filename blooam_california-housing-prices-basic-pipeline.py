housing = pd.read_csv('/kaggle/input/housing/housing.csv')
housing.head()
housing.info()
housing['ocean_proximity'].value_counts()
housing.describe()
%matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
plt.show()
import numpy as np
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
housing['income_cat'] = np.ceil(housing['median_income']/1.5)
housing['income_cat'].where(housing['income_cat']<5, 5.0, inplace=True)
housing['income_cat'].hist()
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
compare_props = pd.DataFrame({
    "All_data": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Error - random (%)"] = 100 * compare_props["Random"] / compare_props["All_data"] - 100
compare_props["Error - stratified (%)"] = 100 * compare_props["Stratified"] / compare_props["All_data"] - 100
compare_props
for set_ in (strat_test_set, strat_train_set):
    set_.drop('income_cat', axis=1, inplace=True)
housing = strat_train_set.copy()
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=housing['population']/100, label='Population',
            c=housing['median_house_value'], figsize=(10,7), cmap=plt.get_cmap('jet'), colorbar=True)
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix
attributes = ['median_house_value','median_income','total_rooms','housing_median_age']
scatter_matrix(housing[attributes], figsize=(12,8))
housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)
housing['Rooms_per_family'] = housing['total_rooms']/housing['households']
housing['Bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
housing['Population_per_family'] = housing['population']/housing['households']

corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)
housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
housing_num = housing.drop('ocean_proximity', axis=1)
imputer.fit(housing_num)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing['ocean_proximity']
housing_cat_encoded = encoder.fit_transform(housing_cat)
print(encoder.classes_)
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(np.array(housing_cat).reshape(-1,1)) #Direct OneHotEncoder on categorical data
housing_cat_1hot
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, houshold_ix = 3,4,5,6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self #Dummy function
    def transform(self, X, y=None):
        Rooms_per_family = X[:, rooms_ix] / X[:, houshold_ix]
        Population_per_family = X[:, population_ix] / X[:, houshold_ix]
        if self.add_bedrooms_per_room:
            Bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, Rooms_per_family, Population_per_family, Bedrooms_per_room]
        else:
            return np.c_[X, Rooms_per_family, Population_per_family]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

housing_num_tr = num_pipeline.fit_transform(housing_num)
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', OneHotEncoder(sparse=False)),
])
from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion([
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline),
])

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print('Predictions: ', lin_reg.predict(some_data_prepared))
print('Labels: ', list(some_labels))
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels, 
                         scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-scores)
def display_scores(scores):
    print('Results: ', scores)
    print('Mean: ', scores.mean())
    print('Std: ', scores.std())

display_scores(tree_rmse_scores)
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=10)
forest_reg.fit(housing_prepared, housing_labels)
forest_pred = forest_reg.predict(housing_prepared)
forest_rmse = np.sqrt(mean_squared_error(housing_labels,forest_pred))
forest_rmse
scores = cross_val_score(forest_reg, housing_prepared, housing_labels, 
                                  scoring='neg_mean_squared_error', cv=10)
forest_rmse_scores = np.sqrt(-scores)
display_scores(forest_rmse_scores)
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3,10,30], 'max_features': [2,4,6,8]},
    {'bootstrap': [False], 'n_estimators': [3,10], 'max_features': [2,3,4]}
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)
print('Best parameters: ', grid_search.best_params_)
print('Best model: ',grid_search.best_estimator_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score),params)
feature_importance = grid_search.best_estimator_.feature_importances_
feature_importance
extra_attribs = ['Rooms_per_family', 'Population_per_family', 'Bedroom_per_rooms']
cat_one_hot_attribs = list(encoder.categories_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importance, attributes), reverse=True)
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set['median_house_value'].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test,final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse
from sklearn.svm import SVR

model = SVR(gamma='auto') #this value will be the default from future versions
svr_scores = cross_val_score(model, housing_prepared, housing_labels, 
                scoring='neg_mean_squared_error', cv=10)
svr_rmse_scores = np.sqrt(-svr_scores)
display_scores(svr_rmse_scores)
display_scores(forest_rmse_scores)
param_grid_svr = [
    {'kernel': ['linear'], 'gamma': ['scale', 'auto'], 'C': [0.1, 0.5, 1.0, 2.0, 5.0]},
    {'kernel': ['rbf'], 'gamma': ['scale', 'auto'], 'C': [0.1, 0.5, 1.0, 2.0, 5.0]}
]
grid_search_svr = GridSearchCV(model, param_grid_svr, cv=5, scoring='neg_mean_squared_error')
grid_search_svr.fit(housing_prepared, housing_labels)
print('Best parameters: ', grid_search_svr.best_params_)
print('Best model: ',grid_search_svr.best_estimator_)
cvres = grid_search_svr.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score),params)
param_grid_C = [
    {'C': [5.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]}
]
grid_search_C = GridSearchCV(SVR(kernel='linear', gamma='scale'), param_grid_C, cv=5, scoring='neg_mean_squared_error')
grid_search_C.fit(housing_prepared, housing_labels)
print('Best parameters: ', grid_search_C.best_params_)
print('Best model: ',grid_search_C.best_estimator_)
cvres = grid_search_C.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score),params)
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal

params = {'kernel': ['linear', 'rbf'], 'C': reciprocal(20,200000)}
random_grid_search = RandomizedSearchCV(model, params, cv=5, scoring='neg_mean_squared_error', verbose=2, random_state=42)
random_grid_search.fit(housing_prepared, housing_labels)
print('Best parameters: ', random_grid_search.best_params_)
print('Best model: ',random_grid_search.best_estimator_)
cvres = random_grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score),params)
reciprocal_distrib = reciprocal(20, 200000)
samples = reciprocal_distrib.rvs(10000, random_state=42)
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title("Reverse distribution")
plt.hist(samples, bins=50)
plt.subplot(122)
plt.title("Log of reverse distribution")
plt.hist(np.log(samples), bins=50)
plt.show()
feature_importance = feature_importance[:-1]
def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]
k = 5
top_k_feature_indices = indices_of_top_k(feature_importance, k)
np.array(attributes)[:5]
sorted(zip(feature_importance, attributes), reverse=True)[:k]
full_pipeline_feature_selection = Pipeline([
    ('full_pipeline', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importance, k))
])
housing_prepared_top_k_features = full_pipeline_feature_selection.fit_transform(housing)
np.all(housing_prepared_top_k_features[0:3] == housing_prepared[0:3, top_k_feature_indices])
prepare_select_predict_pipeline = Pipeline([
    ('data_preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importance, k)),
    ('prediction', SVR(**random_grid_search.best_params_))
])

prepare_select_predict_pipeline.fit(housing, housing_labels)
some_data = housing[1:4]
some_labels = housing_labels[1:4]
print("Predictions:\t", prepare_select_predict_pipeline.predict(some_data))
print("Lables:\t\t", list(some_labels))
grid_param = [
     {'feature_selection__k': list(range(1, len(feature_importance) + 1)),
     'data_preparation__num_pipeline__imputer__strategy': ['mean', 'median', 'most_frequent'],
     'prediction__gamma': ['auto', 'scale']}
]
grid_search_prep = GridSearchCV(prepare_select_predict_pipeline, grid_param, cv=5,
                               scoring='neg_mean_squared_error', error_score=np.nan)
grid_search_prep.fit(housing, housing_labels)
print(grid_search_prep.best_params_)
print(np.sqrt(-grid_search_prep.best_score_))