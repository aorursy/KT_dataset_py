import os 
import tarfile
from six.moves import urllib 
download_root = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
housing_path = 'datasets/housing'
housing_url = download_root + housing_path + '/housing.tgz'
def fetch_housing_data(housing_url = housing_url, housing_path = housing_path):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url,tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path = housing_path)
    housing_tgz.close()
fetch_housing_data()
import pandas as pd
%matplotlib inline
import matplotlib.pyplot as plt
data_original = pd.read_csv('datasets/housing/housing.csv')
data_original.head()
data_original.info()
data_original.describe()
data_original.hist(bins=50, figsize= (20,15))
import numpy as np 
import hashlib
np.random.seed(42)
def train_test_split(data, test_ratio):
    index = np.random.permutation(len(data))
    size = int(len(index)*test_ratio)
    test_index = index[:size]
    train_index = index[size:]
    return data.iloc[train_index],data.iloc[test_index]
train_set, test_set = train_test_split(data_original, 0.2)
train_set
test_set
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data_original, test_size = 0.2,random_state = 42)
train_set.head()
test_set.head()
def check_id(id_, test_ratio, hash):
    return hash(np.int64(id_)).digest()[-1] < (255* test_ratio)
def hash_test_train_split(data, test_ratio,id_column, hash = hashlib.md5):
    id_ = data[id_column]
    in_test_set = id_.apply(lambda id__ : check_id(id__, test_ratio, hash))
    return data[~in_test_set], data[in_test_set]

id_= data_original.reset_index(drop=False)
train_hash_set, test_hash_set= hash_test_train_split(id_, 0.2 ,'index' , hash = hashlib.md5)
id_.head()
train_hash_set
test_hash_set
from sklearn.model_selection import StratifiedShuffleSplit
data= data_original.copy()
#1
data['income_cat'] = np.ceil(data['median_income'] / 1.5)
data['income_cat'].where(data['income_cat'] < 5, 5.0, inplace = True) 
# The above line replaces the false row with the value
#mentioned in the second parameter
data['income_cat']
#2
pd.cut(data['median_income'],bins=[0., 1.5, 3.0, 4.5, 6.0, np.inf],labels = [1,2,3,4,5])
data[['income_cat','median_income']].hist(bins = 20)
data['income_cat'].value_counts() / len(data) *100
split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 42)
for train_index,test_index in split.split(data, data['income_cat']):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]
strat_train_set.head()
strat_test_set.head()
print('Train set',len(strat_train_set),'Test set',len(strat_test_set))
def income_proportion(data):
    '''Function that calculates the ratio of the values in the income_cat column'''
    return data['income_cat'].value_counts() /len(data['income_cat'])

train_set,test_set = train_test_split(data,test_size = 0.2,random_state= 42)

result_table= pd.DataFrame({
    'Overall': income_proportion(data),
    'Random' : income_proportion(test_set),
    'Stratified' : income_proportion(strat_test_set),
}).sort_index()
result_table['% random error'] = (result_table['Random']-result_table['Overall'])/result_table['Overall']*100 
result_table['% Statified Error'] = (result_table['Stratified']-result_table['Overall'])/result_table['Overall']*100
result_table
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
strat_test_set
data.plot(kind = 'scatter',x= 'longitude',y = 'latitude',s=data['population']/100,c=data['median_house_value'],colorbar=True,
        cmap= plt.get_cmap('jet'),label='Population',figsize=(12,7),alpha =0.5)
housing= strat_train_set.copy()
corr_matrix= housing.corr()
corr_matrix
corr_matrix['median_house_value'].sort_values(ascending = False)
from pandas.plotting import scatter_matrix
scatter_matrix(housing,figsize=(20,20))
housing['bedrooms_per_room']= housing['total_bedrooms']/housing['total_rooms']
housing['rooms_per_household'] = housing['total_rooms']/ housing['households']
housing['population_per_household'] = housing['population']/housing['households']
corr_matrix= housing.corr()
corr_matrix['median_house_value'].sort_values(ascending= False)
housing.columns
housing = strat_train_set.drop(['median_house_value'],axis = 1)
housing_label = strat_train_set['median_house_value'].copy()
housing.info()
# Returns the missing data rows
sample_incomplete_rows = housing[housing.isnull().any(axis=1)] 
sample_incomplete_rows
# 1. Removing the entire total_bedrooms column
housing.drop('total_bedrooms',axis = 1)
#2. Removing the missing valued rows
housing.dropna(subset=['total_bedrooms'])
#3. Subtituting a value(e.g., mean, median etc) instead of the missing value
bed_median=housing['total_bedrooms'].median()
housing['total_bedrooms'].fillna(bed_median)
housing
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median') # Strategy available ['mean','median','most_frequent','constant']
housing_num = housing.select_dtypes(include=[np.number])
imputer.fit(housing_num)
imputer.statistics_
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns = housing_num.columns.values,index = housing_num.index)
housing_tr
housing_tr.loc[sample_incomplete_rows.index]
#1
from sklearn.preprocessing import LabelEncoder
encoder_lab = LabelEncoder()
housing_cat_encoded = encoder_lab.fit_transform(housing['ocean_proximity'])
housing_cat_encoded
encoder_lab.classes_ 
housing_cat_encoded.shape
housing_cat_encoded.reshape(-1,1)
# 2
from sklearn.preprocessing import OneHotEncoder
encoder_hot = OneHotEncoder()
housing_cat_hot = encoder_hot.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_hot
encoder_hot.categories_
housing_cat_hot.toarray()
# 3
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat_binarier = encoder.fit_transform(housing['ocean_proximity'])
housing_cat_binarier
#1 Using class
from sklearn.base import BaseEstimator, TransformerMixin

class AddAttribute(BaseEstimator, TransformerMixin):
    def __init__(self,add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self,X):
        return
    def transform(self,X):
        rooms_per_household = X.loc[:,'total_rooms']/X.loc[:,'households']
        population_per_household = X.loc[:,'population']/X.loc[:,'households']
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X.loc[:,'total_bedrooms']/X.loc[:,'total_rooms']
            return np.c_[X,rooms_per_household, population_per_household,bedrooms_per_room]
        else:
            return np.c_[X,rooms_per_household, population_per_household]
Add_Atri =AddAttribute(add_bedrooms_per_room = False)
housing_extra_attribute = Add_Atri.transform(housing)
housing_extra_attribute
# 2 using function and FunctionTransformer
from sklearn.preprocessing import FunctionTransformer

def Add_Attribute(X, add_bedrooms_per_room = True):
        rooms_per_household = X.loc[:,'total_rooms']/X.loc[:,'households']
        population_per_household = X.loc[:,'population']/X.loc[:,'households']
        if add_bedrooms_per_room:
            bedrooms_per_room = X.loc[:,'total_bedrooms']/X.loc[:,'total_rooms']
            return np.c_[X,rooms_per_household, population_per_household,bedrooms_per_room]
        else:
            return np.c_[X,rooms_per_household, population_per_household]
Add_Atri =FunctionTransformer(Add_Attribute, validate= False,kw_args={'add_bedrooms_per_room' : False})
housing_extra_attribute = Add_Atri.transform(housing)
housing_extra_attribute
def dataframeTransformer(X):
    return pd.DataFrame(X, columns = list(housing_tr.select_dtypes(include = [np.number]).columns))
#transform = FunctionTransformer(dataframeTransformer,validate = False)
#transform.fit_transform(housing_extra_attribute)
housing_extra_attribute = pd.DataFrame(housing_extra_attribute, columns = list(housing.columns)
                                       +['rooms_per_household','population_per_household'] ,index = housing.index)
housing_extra_attribute
from sklearn.preprocessing import  StandardScaler
from sklearn.pipeline import Pipeline
num_pipeline = Pipeline([
                  ('Imp',SimpleImputer(strategy= 'median')), # Fills missing value
                  ('transform',FunctionTransformer(dataframeTransformer,validate = False)), # Change the array from Imputer to df
                  ('Attribs_adder',FunctionTransformer(Add_Attribute, validate = False, kw_args={'add_bedrooms_per_room' : True})), # add extra attributes
                  ('std_scaler',StandardScaler()) # Scales the value
                ])
housing_pipe = num_pipeline.fit_transform(housing_num) 
housing_pipe
cat_pipeline=Pipeline([
    ('hot_cat',OneHotEncoder(sparse = False)) 
])
cat_output = cat_pipeline.fit_transform(np.array(housing['ocean_proximity']).reshape(-1,1))
cat_output
from sklearn.compose import ColumnTransformer
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared
housing_prepared = pd.DataFrame(housing_prepared, index = housing.index, columns = list(housing_num.columns)
                                       + ['rooms_per_household','population_per_household','beedrooms_per_room'] + list(full_pipeline.transformers_[1][1].categories_[0]))
housing_prepared.head()
from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(housing_prepared, housing_label)
reg.coef_
some_data_in_train_set = housing.iloc[:5]
some_data_in_train_set
some_data_prepared = full_pipeline.transform(some_data_in_train_set)
some_data_prepared
reg_prediction = reg.predict(some_data_prepared)
reg_prediction
some_data_actual_label=housing_label[:5]
some_data_actual_label
from sklearn.metrics import mean_squared_error

house_prediction = reg.predict(housing_prepared)
lin_mse = mean_squared_error(house_prediction, housing_label)
lin_rmse = np.sqrt(lin_mse)
lin_mse,lin_rmse
from sklearn.metrics import mean_absolute_error
lin_mae = mean_absolute_error(house_prediction,housing_label)
lin_mae
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state = 42)
tree_reg.fit(housing_prepared,housing_label)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_label, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
from sklearn.svm import SVR

svr_reg = SVR(kernel='linear')
svr_reg.fit(housing_prepared, housing_label)
housing_prediction = svr_reg.predict(housing_prepared)
svr_mse = mean_squared_error(housing_label,housing_prediction)
svr_rmse = np.sqrt(svr_mse)
svr_rmse
# Score of the regression model
from sklearn.model_selection import cross_val_score

score = cross_val_score(reg, housing_prepared, housing_label, scoring = 'neg_mean_squared_error', cv =10)
reg_rmse = np.sqrt(-score)
reg_rmse
# Decision tree score
score = cross_val_score(tree_reg, housing_prepared, housing_label, scoring = 'neg_mean_squared_error', cv =10)
tree_rmse = np.sqrt(-score)
tree_rmse
# ramdomforest regressor 
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators = 10, random_state=42)
forest_reg.fit(housing_prepared, housing_label)

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_label, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
score = cross_val_score(forest_reg, housing_prepared, housing_label, scoring = 'neg_mean_squared_error', cv =10)
forest_rmse = np.sqrt(-score)
forest_rmse
score = cross_val_score(svr_reg, housing_prepared, housing_label, scoring = 'neg_mean_squared_error', cv =10)
svr_rmse = np.sqrt(-score)
svr_rmse
from sklearn.model_selection import GridSearchCV

paras = [{'n_estimators':[30,40,45 ], 'max_features' : [2 ,4, 6, 8]},
        {'bootstrap' : [False],'n_estimators':[3, 10], 'max_features' : [2,3,4]}]
forest_reg = RandomForestRegressor(random_state = 42)
grid_search = GridSearchCV(forest_reg, paras, cv=5, scoring = 'neg_mean_squared_error',
                          return_train_score= True)
grid_search.fit(housing_prepared, housing_label)
grid_search.best_params_
grid_search.best_estimator_
feature_importance = grid_search.best_estimator_.feature_importances_
cv = grid_search.cv_results_
for mean_scores, params in zip(cv['mean_test_score'],cv['params']):
    print(np.sqrt(-mean_scores), params)
pd.DataFrame(grid_search.cv_results_)
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_distribs ={
    'n_estimators' : randint(low = 1, high = 200),
    'max_features' : randint(low= 1, high = 8)
}
forest_reg = RandomForestRegressor(random_state = 42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions = param_distribs, cv = 5, n_iter= 10, scoring= 'neg_mean_squared_error', random_state = 42)
rnd_search.fit(housing_prepared, housing_label)
rnd_search.best_estimator_
rnd_search.best_params_
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
final_rmse
