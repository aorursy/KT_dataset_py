import os

import tarfile
open= tarfile.open('/kaggle/input/housing.tgz')

open.getnames()
open.extractall()
import pandas as pd



housing= pd.read_csv('housing.csv')

housing.head()
housing.describe()
housing.info()
housing['ocean_proximity'].value_counts()
import matplotlib.pyplot as plt



%matplotlib inline

housing.hist(bins=50, figsize= (20,20))



plt.show()
correlation= housing.corr()

correlation['median_house_value'].sort_values(ascending= False)
import numpy as np



housing['income_cat']=np.ceil(housing['median_income']/1.5)

housing['income_cat'].where(housing['income_cat']<5, 5.0, inplace=True)

housing['income_cat'].value_counts()/len(housing)
housing['income_cat'].hist(bins=20, figsize=(5,5))
from sklearn.model_selection import StratifiedShuffleSplit



split= StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['income_cat']):

    strat_train_set= housing.loc[train_index]

    strat_test_set= housing.loc[test_index]
for set_ in (strat_train_set, strat_test_set):

    set_.drop('income_cat', axis=1, inplace=True)
housing= strat_train_set.copy()
corr_matrix= housing.corr()

corr_matrix['median_house_value'].sort_values(ascending= False)
from pandas.plotting import scatter_matrix



attributes= ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']

scatter_matrix(housing[attributes], figsize=(12,8))
housing.plot(kind="scatter", x='median_income', y='median_house_value', alpha=0.1)
list(housing)
housing['rooms_per_household']= housing['total_rooms'] / housing['households']

housing['bedrooms_per_room']= housing['total_bedrooms'] / housing['total_rooms']

housing['population_per_household']= housing['population'] / housing['households']
corr_matrix= housing.corr()

corr_matrix['median_house_value'].sort_values(ascending= False)
housing = strat_train_set.drop("median_house_value", axis=1)

housing_labels= strat_train_set["median_house_value"].copy()
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix= 3,4,5,6



class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room= True):

        self.add_bedrooms_per_room= add_bedrooms_per_room

    def fit(self,X, y= None):

        return self

    def transform(self, X, y= None):

        rooms_per_household= X[:, rooms_ix]/X[:,household_ix]

        population_per_household= X[:,population_ix]/X[:,household_ix]

        if self.add_bedrooms_per_room:

            bedrooms_per_room= X[:,bedrooms_ix]/X[:,rooms_ix]

            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]

        else:

            return np.c_[X, rooms_per_household, population_per_household]



attr_adder= CombinedAttributesAdder(add_bedrooms_per_room= False)

housing_extra_attribs= attr_adder.transform(housing.values)    
housing_num= housing.drop("ocean_proximity", axis=1)
from sklearn.pipeline import Pipeline



from sklearn.preprocessing import StandardScaler

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder



num_pipeline= Pipeline([

        ('imputer', SimpleImputer(strategy="median")),

        ('attribs_adder', CombinedAttributesAdder()),

        ('std_scaler', StandardScaler()),

    ])

housing_num_tr= num_pipeline.fit_transform(housing_num)
from sklearn.compose import ColumnTransformer

num_attribs= list(housing_num)

cat_attribs= ['ocean_proximity']



full_pipeline= ColumnTransformer([

    ('num_encoded', num_pipeline, num_attribs),

    ('cat_encoded', OneHotEncoder(), cat_attribs),

])



final_train_data= full_pipeline.fit_transform(housing)
final_train_data.shape
from sklearn.linear_model import LinearRegression



lr= LinearRegression()

lr.fit(final_train_data, housing_labels)
from sklearn.metrics import mean_squared_error as mse



housing_predictions= lr.predict(final_train_data)

lr_mse= mse(housing_labels, housing_predictions)

rmse= np.sqrt(lr_mse)

rmse
from sklearn.model_selection import cross_val_score



lr_score= cross_val_score(lr, final_train_data, housing_labels, scoring= 'neg_mean_squared_error', cv=10)

lr_rmse_score= np.sqrt(-lr_score)
def display_score(score):

    print("scores:", score)

    print("mean:", score.mean())

    print("standard deviation:", score.std())



display_score(lr_rmse_score)
from sklearn.tree import DecisionTreeRegressor



dt= DecisionTreeRegressor()

dt.fit(final_train_data, housing_labels)
housing_predictions= dt.predict(final_train_data)

dt_mse= mse(housing_labels, housing_predictions)

rmse= np.sqrt(dt_mse)

rmse
dt_score= cross_val_score(dt, final_train_data, housing_labels, scoring= "neg_mean_squared_error", cv=10)

dt_rmse_score= np.sqrt(-dt_score)

display_score(dt_rmse_score)
from sklearn.ensemble import RandomForestRegressor



rf= RandomForestRegressor(n_estimators=10, random_state=42)

rf.fit(final_train_data, housing_labels)
housing_predictions= rf.predict(final_train_data)

rf_mse= mse(housing_labels, housing_predictions)

rmse= np.sqrt(rf_mse)

rmse
rf_score= cross_val_score(rf, final_train_data, housing_labels, scoring="neg_mean_squared_error", cv=10)

rf_rmse= np.sqrt(-rf_score)

display_score(rf_rmse)
from sklearn.model_selection import GridSearchCV



param_grid = [

    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},

    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},

  ]



forest_reg = RandomForestRegressor(random_state=42)

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,n_jobs=-1,

                           scoring='neg_mean_squared_error', return_train_score=True)

grid_search.fit(final_train_data, housing_labels)
grid_search.best_params_
final_model= grid_search.best_estimator_
from sklearn.externals import joblib

joblib.dump(final_model, "final_model" )

joblib.dump(housing_labels, " housing_labels" )
cvres= grid_search.cv_results_

for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), params)
feature_importance= grid_search.best_estimator_.feature_importances_



extra_attribs= ["rooms_per_household", "population_per_household", "bedrooms_per_room"]

cat_encoder = full_pipeline.named_transformers_["cat_encoded"]

cat_one_hot_attribs = list(cat_encoder.categories_[0])

attributes= num_attribs + extra_attribs + cat_one_hot_attribs

sorted(zip(feature_importance, attributes), reverse= True)
final_model= grid_search.best_estimator_



X_test= strat_test_set.drop("median_house_value", axis=1)

y_test= strat_test_set['median_house_value'].copy()



X_test_data= full_pipeline.transform(X_test)

final_house_predictions= final_model.predict(X_test_data)



final_mse= mse(y_test, final_house_predictions)

final_rmse= np.sqrt(final_mse)

final_rmse
Final_house_predictions = pd.DataFrame(final_house_predictions, columns=['Housing Prices'])

Final_house_predictions= Final_house_predictions.to_csv(index=False)

joblib.dump(Final_house_predictions, "Predicted Housing Prices.csv")