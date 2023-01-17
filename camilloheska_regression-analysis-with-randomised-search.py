import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from sklearn.preprocessing import FunctionTransformer

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

import os

print(os.listdir("../input/california-housing-prices"))
housing = pd.read_csv('../input/california-housing-prices/housing.csv')

housing.head()
print('Number of entries in the dataset: {}.'.format(len(housing)))

print('There are {} features in the dataset.'.format(len(housing.columns)))

print('--------------------')

print('List of categorical features: \n{}'.format([x for x in housing.select_dtypes(include='O').columns]))

print('List of continuous features: \n{}'.format([x for x in housing.select_dtypes(exclude='O').columns]))

print('------------------')

print('Features with missing values include:')

_ = housing.isnull().sum()

for x,y in zip(_.index,_):

    if y>0:

        print('{} with {} missing values.'.format(x,y))

print('------------------')

print('Cardinality of the categorical feature:')

_ = housing.ocean_proximity.value_counts()

for x,y in zip(_.index,_):

    print('{} has {} labels.'.format(x,y))
housing.describe()
housing.hist(bins=50,figsize=(20,15))

plt.show()
housing['income_cat'] = np.ceil(housing['median_income'] /1.5)

print('Cardinality of median_income before discretization {} and after {} .'.format(len(housing.median_income.value_counts()),len(housing.income_cat.value_counts())))

print('After discretization:\n',housing.income_cat.value_counts())
housing['income_cat'] = np.where(housing['income_cat']>5,5.0,housing['income_cat'])

housing.income_cat.plot(kind='hist')
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

for train_index, test_index in split.split(housing,housing['income_cat']):

    strat_train_set = housing.loc[train_index]

    strat_test_set = housing.loc[test_index]
original = pd.Series(housing['income_cat'].value_counts() / len(housing), name='Original')

strat = pd.Series(strat_test_set['income_cat'].value_counts() / len(strat_test_set),name='Stratified')

train_set, test_set = train_test_split(housing,test_size=0.2,random_state=42)

random = pd.Series(test_set['income_cat'].value_counts() / len(test_set), name='Random')

test_sets_comparisons = pd.DataFrame([original,strat,random]).T.sort_index()

test_sets_comparisons['% Error Strat'] = 100 * (test_sets_comparisons['Stratified'] / test_sets_comparisons['Original']) - 100

test_sets_comparisons['% Error Random'] = 100 * (test_sets_comparisons['Random'] / test_sets_comparisons['Original']) - 100

test_sets_comparisons
for _ in (strat_train_set,strat_test_set):

    _.drop(['income_cat'],axis=1,inplace=True)
housing = strat_train_set.copy()
housing.columns
plt.figure(figsize=(20,20))

housing.plot.scatter(x='longitude',y='latitude', alpha=0.1)

plt.show()
fig, ax = plt.subplots(figsize=(15,10))

housing.plot.scatter(x='longitude',y='latitude'

                     ,alpha=0.3,s=housing['population']/100,label='population'

                     ,c='median_house_value',cmap=plt.get_cmap('jet'),colorbar=True, legend=True, ax=ax)

plt.show()
fig, ax = plt.subplots(figsize=(10,5))

housing[housing['median_house_value']>400000].plot.scatter(x='longitude',y='latitude'

                     ,alpha=0.3,s=housing['population']/100,label='population'

                     ,c='median_house_value',cmap=plt.get_cmap('jet'),colorbar=True, legend=True, ax=ax)

plt.show()
def coordinate_transformer(latitude,longitude):

    """

    This method takes the latitude and longitude coordinates, adding the number of missing zeros needed for the geolocator 

    request. The outputs are then used to find the county name.

    """

    number_rounder_lat,number_rounder_long = (9 - len(str(latitude))),(9 - len(str(longitude)))

    latitude = str(latitude) + str(0)*(number_rounder_lat)

    longitude = str(longitude) + str(0)*(number_rounder_long)

    from geopy.geocoders import Nominatim

    geolocator = Nominatim(user_agent="california_median_housing_price")

    try:

        location = geolocator.reverse(latitude+", "+longitude)

        return location.raw['address']['county']

    except:

        return 'Not Found'
print('There are {} instances of lat/long in the dataset.'.format(housing.shape[0]))
_ = housing.groupby(['latitude','longitude'])['housing_median_age'].count().reset_index().drop(['housing_median_age'],axis=1)

print('There are {} unique combinations of lat/long in the dataset.'.format(_.shape[0]))
_['latitude'],_['longitude'] = np.round(_['latitude'],1),np.round(_['longitude'],1)

_ = _.groupby(['latitude','longitude']).count().reset_index()

print('If rounded to 1 decimal point, we have {} unique combinations.'.format(_.shape[0]))
_['longitude'] = np.round(_['longitude'])

_ = _.groupby(['latitude','longitude']).count().reset_index()

print('If rounded to 0 decimal points the longitude, we have {} unique combinations.'.format(_.shape[0]))
#from timeit import default_timer as timer

county_list = []

#start = timer()

for lat, long in zip(_.latitude,_.longitude):

    county_list.append(coordinate_transformer(lat,long))

#end = timer()

#county_list = pd.Series(county_list)

#print(end - start)
_['county'] = county_list

housing['latitude_join'] = np.round(housing['latitude'],1)

housing['longitude_join'] = np.round(np.round(housing['longitude'],1))

housing = pd.merge(housing,_,how='left',left_on=['latitude_join','longitude_join'], right_on=['latitude','longitude']).drop(['latitude_join',

       'longitude_join', 'latitude_y', 'longitude_y'],axis=1)

housing.rename(columns={'longitude_x':'longitude','latitude_x':'latitude'},inplace=True)

housing
threshold = 400000

plt.axhline(y=threshold,linewidth=4, color='red')

housing[(housing.median_house_value>350000)].groupby(['county'])['median_house_value'].mean().plot(kind='bar',legend=True,figsize=(10,7),cmap=plt.get_cmap('jet'))

plt.legend(loc='best')

print('List of Counties that exceed the threshold:')

high_valued_houses_counties = []

for x in housing[(housing.median_house_value>400000)].groupby(['county'])['median_house_value'].mean().index:

    high_valued_houses_counties.append(x)

    print(x)
print('The percentage of districts in highly valued counties (£400k and above) is {:.2%}.'.format(housing[housing.county.isin(high_valued_houses_counties)].shape[0]/housing.shape[0]))

(housing[housing.county.isin(high_valued_houses_counties)]['ocean_proximity'].value_counts()/len(housing))
corr_matrix = housing.corr()

corr_matrix['median_house_value'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix

best_f = corr_matrix['median_house_value'].sort_values(ascending=False).head(4).index.to_list()

scatter_matrix(housing[best_f],figsize=(12,12))

plt.show()
housing.plot(kind='scatter',x='median_income',y='median_house_value',alpha=0.1)
# here we want to remove the ones that appear in the scatter plot - the capping values.

housing[housing.median_house_value==350000].shape

housing[housing.median_house_value==450000].shape

housing[housing.median_house_value==500000].shape
for f in housing.columns[2:]:

    print(f,housing[f].isnull().sum())
import seaborn as sns

sns.distplot((housing.total_bedrooms.dropna()))
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy='median')

imputer.fit(housing['total_bedrooms'].values.reshape(-1,1))

imputer.statistics_
housing.total_bedrooms.median()
median_t = imputer.statistics_ 

housing['total_bedrooms'] = imputer.transform(housing['total_bedrooms'].values.reshape(-1,1))
housing.isnull().sum()
start =1 

end = 3

cols = ['population','median_income','households','total_bedrooms','total_rooms']

ax, fig = plt.subplots(nrows=5,ncols=2,figsize=(20,20))

for col in cols:

    for i in range(start,end):

        plt.subplot(5,2,i)

        sns.distplot(housing[col], label = col)

        plt.legend()

        try:

            plt.subplot(5,2,i+1)

        except:

            plt.subplot(5,2,i)

        sns.distplot(np.log(housing[col]), label= [str(col)+'_log  base'])

        plt.legend()

        break

    start=end

    end=end+2

        
cols = ['population','median_income','households','total_bedrooms','total_rooms']

for col in cols:

    housing[col] = np.log(housing[col])
housing['rooms_per_household'] = housing['total_rooms'] / housing['households']

housing['bedrooms_per_room'] = housing['total_bedrooms'] / housing['total_rooms']

housing['population_per_household'] = housing['population'] / housing['households']

housing['bedrooms_per_household'] = housing['total_bedrooms'] / housing['households']
housing_corr = housing.corr()

housing_corr['median_house_value'].sort_values(ascending=False)
housing.select_dtypes(include=['O']).columns
housing.drop(['county'],axis=1,inplace=True)
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder()

encoded_f = OneHotEncoder(handle_unknown='ignore').fit_transform(housing.ocean_proximity.values.reshape(-1,1)).toarray()

n = housing['ocean_proximity'].unique()

cols = ['{}_{}'.format('Ocean_prox_',n)for n in n]

encoded_f = pd.DataFrame(encoded_f,index=housing.index,columns=cols)

housing = pd.concat([housing,encoded_f],axis=1)

housing.drop(['ocean_proximity'],axis=1,inplace=True)

housing.head()
housing = strat_train_set.drop(['median_house_value'],axis=1).copy()

housing_labels = strat_train_set['median_house_value'].copy()
print(housing.shape, housing_labels.shape)

housing.columns
from sklearn.preprocessing import FunctionTransformer



def add_extra_features(X, add_bedrooms_per_room=True):

    # here I take the col index of each feature of interest

    rooms_ix, bedrooms_ix, population_ix, household_ix, median_income_ix = [

    list(housing.columns).index(col) for col in ("total_rooms", "total_bedrooms", "population", "households",'median_income')]

    

    # here I replicate the calculations I did before but this time I am using directly the col indexes

    rooms_per_household = X[:, rooms_ix] / X[:, household_ix]

    population_per_household = X[:, population_ix] / X[:, household_ix]

    bedrooms_per_household = X[:,bedrooms_ix] / X[:,household_ix]

    median_income_per_household = X[:,median_income_ix] / X[:,household_ix]

    #I let the user decide if return bedrooms_per_room additional to the above calculate

    if add_bedrooms_per_room:

        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]

        return np.c_[X, rooms_per_household, population_per_household,bedrooms_per_household,

                     median_income_per_household,bedrooms_per_room]

    else:

        return np.c_[X, rooms_per_household, population_per_household,bedrooms_per_household,median_income_per_household]





def log_transformation(X):

    import numpy as np

    # get index cols

    population_ix,median_income_ix,household_ix,bedrooms_ix,rooms_ix =[

        list(housing.columns).index(col) for col in ('population','median_income','households','total_bedrooms','total_rooms')

    ]

    # log tranformation

    population_log = np.log(X[:,population_ix].astype('float64'))

    median_income_log = np.log(X[:,median_income_ix].astype('float64'))

    household_log = np.log(X[:,household_ix].astype('float64'))

    bedrooms_log = np.log(X[:,bedrooms_ix].astype('float64'))

    rooms_log = np.log(X[:,rooms_ix].astype('float64'))

    # return results

    return np.c_[X,population_log,median_income_log,household_log,bedrooms_log,rooms_log]



attr_adder = FunctionTransformer(add_extra_features, validate=False,

                                 kw_args={"add_bedrooms_per_room": True})

log_transformed = FunctionTransformer(log_transformation,validate=False)

housing_extra_attribs = attr_adder.fit_transform(housing.values)

housing_log_transformed = log_transformed.fit_transform(housing.values)
housing_extra_attribs = pd.DataFrame(

    housing_extra_attribs,

    columns=list(housing.columns)+['rooms_per_household', 'population_per_household','bedrooms_per_household',

                     'median_income_per_household','bedrooms_per_room'],

    index=housing.index)

housing_extra_attribs.head()
housing_log_transformed = pd.DataFrame(

    housing_log_transformed

    ,columns=list(housing.columns) + ['population_log','median_income_log','household_log','bedrooms_log','rooms_log']

    ,index=housing.index

)

housing_log_transformed.head()
housing.columns
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler



numerical_pipeline= Pipeline([

    ('imputer',SimpleImputer(strategy='median',missing_values=np.nan))

    ,('log_transform',FunctionTransformer(log_transformation,validate=False))

    ,('add_features',FunctionTransformer(add_extra_features,validate=False))

    ,('std_scaler',StandardScaler())

])



housing_numerical_transformed = numerical_pipeline.fit_transform(housing.drop(['ocean_proximity'],axis=1))

housing_numerical_transformed
from sklearn.compose import ColumnTransformer

numerical_f = list(housing.drop(['ocean_proximity'],axis=1))

categorical_f = ["ocean_proximity"]



full_pipeline = ColumnTransformer([

    ('numericals',numerical_pipeline,numerical_f)

    ,('categorical',OneHotEncoder(),categorical_f)

])

housing_completed = full_pipeline.fit_transform(housing)
housing_completed
housing_completed_df = pd.DataFrame(housing_completed, columns=list(housing.drop(['ocean_proximity'],axis=1).columns) +['population_log','median_income_log','household_log','bedrooms_log','rooms_log','rooms_per_household', 'population_per_household','bedrooms_per_household','median_income_per_household','bedrooms_per_room']+['Ocean_prox__<1H OCEAN',

 'Ocean_prox__NEAR OCEAN','Ocean_prox__INLAND','Ocean_prox__NEAR BAY','Ocean_prox__ISLAND'],index=housing.index)

housing_completed_df.head()
from sklearn.linear_model import LinearRegression



lin_reg = LinearRegression()

lin_reg.fit(housing_completed_df,housing_labels)
some_data = housing.iloc[:5]

some_labels = housing_labels.iloc[:5]

some_data_prepared = full_pipeline.transform(some_data)

print('Predictions', np.round(lin_reg.predict(some_data_prepared),1))

print('Labels:',list(some_labels))
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_completed_df)

lin_mse = mean_squared_error(housing_labels,housing_predictions)

lin_rmse = np.sqrt(lin_mse)

lin_rmse
print('Target Summary Statistics:\nMean: {:.2f}\nMedian: {:.2f}\nStandard Deviation: {:.2f}'.format(housing_labels.mean(),housing_labels.median(),housing_labels.std()))
from sklearn.metrics import r2_score

r_score = r2_score(housing_labels,housing_predictions)

r_score
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()

tree_reg.fit(housing_completed_df,housing_labels)
housing_predictions = tree_reg.predict(housing_completed_df)

tree_mse = mean_squared_error(housing_labels,housing_predictions)

tree_rmse = np.sqrt(tree_mse)

tree_rmse
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg,housing_completed_df,housing_labels,scoring='neg_mean_squared_error',cv=10)

rmse_score = np.sqrt(-scores)
print('Scores:\n{}\nMean: {}\nStandard Deviation:{} '.format(rmse_score,rmse_score.mean(),rmse_score.std()))
lin_score = cross_val_score(lin_reg,housing_completed_df,housing_labels,scoring='neg_mean_squared_error',cv=10)

lin_rmse_score = np.sqrt(-lin_score)

print('Scores:\n{}\nMean: {}\nStandard Deviation:{} '.format(lin_rmse_score,lin_rmse_score.mean(),lin_rmse_score.std()))
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=10,random_state=0)

forest_reg.fit(housing_completed_df,housing_labels)

housing_predictions = forest_reg.predict(housing_completed_df)

forest_mse = mean_squared_error(housing_labels,housing_predictions)

forest_rmse = np.sqrt(forest_mse)

forest_rmse
forest_scores = cross_val_score(forest_reg,housing_completed_df,housing_labels,scoring='neg_mean_squared_error',cv=10)

forest_rmse_scores = np.sqrt(-forest_scores)

print('Scores:\n{}\nMean: {}\nStandard Deviation:{} '.format(forest_rmse_scores,forest_rmse_scores.mean(),forest_rmse_scores.std()))
import joblib

joblib.dump(forest_reg,'forest_reg.pkl')
from sklearn.model_selection import GridSearchCV



param_grid = [

    {'n_estimators': [3, 10, 30,40,50], 'max_features': [2, 4, 6, 8]},

    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},

  ]

forest_reg = RandomForestRegressor(random_state=0)

grid_search = GridSearchCV(forest_reg,param_grid,cv=5,scoring='neg_mean_squared_error', return_train_score=True)

grid_search.fit(housing_completed_df,housing_labels)
grid_search.best_params_
grid_search.best_estimator_
grid_results = grid_search.cv_results_

grid_results.keys()
for mean_score,params in zip(grid_results['mean_test_score'],grid_results['params']):

    print(np.sqrt(-mean_score),params)
from sklearn.model_selection import RandomizedSearchCV

from scipy.stats import randint

param_distributions = {

    'n_estimators': randint(low=1, high=200)

    ,'max_features': randint(low=1, high=8)

    }



forest_reg = RandomForestRegressor(random_state=0)

random_search = RandomizedSearchCV(forest_reg,param_distributions=param_distributions,n_iter=20,cv=5,scoring='neg_mean_squared_error', return_train_score=True)

random_search.fit(housing_completed_df,housing_labels)
random_search.best_params_
for mean_scores,params in zip(random_search.cv_results_['mean_test_score'],random_search.cv_results_['params']):

    print(np.sqrt(-mean_scores),params)
model = random_search.best_estimator_

model.feature_importances_
feature_names = housing_completed_df.columns

sorted(zip(model.feature_importances_,feature_names),reverse=True)
housing_completed_less_features = housing_completed_df.drop(['Ocean_prox__<1H OCEAN','Ocean_prox__INLAND', 'Ocean_prox__NEAR BAY', 'Ocean_prox__ISLAND','population','total_rooms','household_log','household_log'],axis=1)



forest_reg_2 = RandomForestRegressor(n_estimators=151,random_state=0,max_features=7)

forest_reg_2.fit(housing_completed_less_features,housing_labels)

housing_predictions = forest_reg_2.predict(housing_completed_less_features)

forest_mse = mean_squared_error(housing_labels,housing_predictions)

forest_rmse = np.sqrt(forest_mse)

forest_rmse
forest_scores = cross_val_score(forest_reg_2,housing_completed_less_features,housing_labels,scoring='neg_mean_squared_error',cv=10)

forest_rmse_scores = np.sqrt(-forest_scores)

print('Scores:\n{}\nMean: {}\nStandard Deviation:{} '.format(forest_rmse_scores,forest_rmse_scores.mean(),forest_rmse_scores.std()))
forest_r2_scores = cross_val_score(forest_reg_2,housing_completed_less_features,housing_labels,scoring='r2',cv=10)

print('Scores:\n{}\nMean: {}\nStandard Deviation:{} '.format(forest_r2_scores,forest_r2_scores.mean(),forest_r2_scores.std()))
final_model = forest_reg_2

final_model
X_test = strat_test_set.drop(['median_house_value'],axis=1)

y_test = strat_test_set['median_house_value'].copy()



X_test.shape,y_test.shape
X_test_preprocessed = full_pipeline.transform(X_test)



# we have to remove the features that we are not using anymore - the original pipeline does not reflect the latest changes

X_test_preprocessed = pd.DataFrame(X_test_preprocessed,columns=housing_completed_df.columns,index=strat_test_set.index).drop(['Ocean_prox__<1H OCEAN','Ocean_prox__INLAND', 'Ocean_prox__NEAR BAY', 'Ocean_prox__ISLAND','population','total_rooms','household_log','household_log'],axis=1)



final_predictions = final_model.predict(X_test_preprocessed)



final_mse = mean_squared_error(y_test,final_predictions)

final_rmse = np.sqrt(final_mse)

final_r2_score = r2_score(y_test,final_predictions)

print('RMSE Score: {}\nR2 Score: {}'.format(final_rmse,final_r2_score))
from scipy import stats

confidence = 0.95

squared_errors = (final_predictions - y_test) ** 2

mean = squared_errors.mean()

m = len(squared_errors)



zscore = stats.norm.ppf((1 + confidence) / 2)

zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)

np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)
