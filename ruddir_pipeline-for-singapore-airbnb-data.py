# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.


PROJECT_ROOT_DIR ='/kaggle/input/singapore-airbnb'

HOUSING_PATH = os.path.join(PROJECT_ROOT_DIR,"listings.csv")

def load_housing_data(housing_path=HOUSING_PATH):

    csv_path = HOUSING_PATH

    return pd.read_csv(csv_path)

housing = load_housing_data()

housing.head()
housing.info()
import seaborn as sns

corr = housing[['price',	'minimum_nights',	'number_of_reviews', 'calculated_host_listings_count',	'availability_365']].corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(240, 10, n=9),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
corr_matrix = housing.corr()

corr_matrix["price"].sort_values(ascending=False)
from sklearn.model_selection import train_test_split

def getting_working_vars (df,column_target):

    X_train = df.drop(column_target, axis=1)

    y_train = df[column_target].values

   

    return X_train,y_train

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

X_train,Y_train = getting_working_vars (train_set,['price'])

X_test,Y_test = getting_working_vars (test_set,['price'])
from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression

from sklearn.linear_model import SGDRegressor

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

from sklearn import metrics

from sklearn.decomposition import TruncatedSVD

from sklearn.metrics import mean_squared_error
numeric_features = ['minimum_nights', 'number_of_reviews','calculated_host_listings_count']

numeric_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer()),

    ('scaler', StandardScaler())])



categorical_features = ['neighbourhood_group', 'neighbourhood', 'room_type']

categorical_transformer = Pipeline(steps=[

    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),

    ('onehot', OneHotEncoder(handle_unknown='ignore'))])





preprocessor = ColumnTransformer(

    transformers=[

        ('num', numeric_transformer, numeric_features),

        ('cat', categorical_transformer, categorical_features)])



from sklearn.tree import DecisionTreeRegressor



full_pipeline_m = Pipeline(steps=[('preprocessor', preprocessor),

                                  #('pca', TruncatedSVD()),

                                  ('model', LinearRegression())])









param_grid = [  

        {'preprocessor__num__imputer__strategy': ['mean','median','most_frequent'],

         #'pca__n_components':[1,2,3,4,5],

         'model': [SGDRegressor()],

         #'model__penalty': ['l1', 'l2'],

         #'model__loss': ['squared_loss', 'huber', 'epsilon_insensitive','squared_epsilon_insensitive']

         'model__alpha': [0.0001 * x for x in range(1, 6)]

         },                

         {'preprocessor__num__imputer__strategy': ['mean','median','most_frequent'],

          'model': [LinearRegression()]

          

          },                 

        {'preprocessor__num__imputer__strategy': ['mean','median','most_frequent'],

         'model': [DecisionTreeRegressor()],

         

         

         },

         {'preprocessor__num__imputer__strategy': ['mean','median','most_frequent'],

         'model': [RandomForestRegressor()],

         

        

         }

         ]   

           

scoring = ['precision_macro', 'recall_macro', 'f1_macro',

               'balanced_accuracy']

grid_search = GridSearchCV(full_pipeline_m, param_grid, cv=10,

                                     n_jobs=-1, verbose=0);

    #

grid_search.fit(X_train,Y_train);

print(grid_search.best_params_)

housing_predictions = grid_search.predict(X_test)

tree_mse = mean_squared_error(Y_test, housing_predictions)

tree_rmse = np.sqrt(tree_mse)

print(tree_rmse)
#

median_price = np.median(Y_test)

num_rows = len(Y_test)

#num_rows

from sklearn import metrics

null_model_predictions = [median_price]*num_rows

#null_model_predictions

np.sqrt(metrics.mean_squared_error(Y_test, null_model_predictions))