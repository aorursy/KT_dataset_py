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
%matplotlib inline

import pandas as pd

import missingno as msno

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sb

from sklearn.preprocessing import Normalizer

from sklearn.model_selection import train_test_split

import statsmodels.api as sm

from statsmodels.formula.api import ols

from scipy.stats import ttest_ind

# from sklearn.ensemble import RandomForestClassifier

# from sklearn.feature_selection import SelectFromModel

import h2o

from h2o.estimators import H2ORandomForestEstimator

from h2o.grid.grid_search import H2OGridSearch



#sets up pandas table display

pd.set_option('display.width', 800)

pd.set_option('display.max_columns', 100)

pd.set_option('display.notebook_repr_html', True)

#stop scientific notation

pd.options.display.float_format = '{:.2f}'.format
# Making a list of missing value types

missing_values = ["n/a", "na", "--"]
#load the 2016 properties data and target variable

house_2016_df = pd.read_csv('../input/properties_2016.csv', na_values = missing_values, low_memory=False)

house_2017_df = pd.read_csv('../input/properties_2017.csv', na_values = missing_values, low_memory=False)

house_log_2016 = pd.read_csv('../input/train_2016_v2.csv', low_memory=False)

house_log_2017 = pd.read_csv('../input/train_2017.csv', low_memory=False)
#get the size

print(house_2016_df.shape)

print(house_2017_df.shape)

print(house_log_2016.shape)

print(house_log_2017.shape)
house_2017_df.sample(10)
#merge the trasaction dataset fro 2016-2017

house_log_full = pd.concat([house_log_2016,house_log_2017],ignore_index=True)

house_log_full.head(10)
#join the 2017 train set with 2016&2017 target variable

house_2017_full = house_2017_df.merge(house_log_full, on = 'parcelid')

house_2017_full.head(5)
#impute boolean variables 

house_2017_full['fireplaceflag'].replace(True, 1, inplace=True)

house_2017_full['fireplaceflag'].fillna(0, inplace = True)

house_2017_full['hashottuborspa'].replace(True, 1, inplace=True)

house_2017_full['hashottuborspa'].fillna(0, inplace = True) 

house_2017_full['pooltypeid10'].fillna(0, inplace = True) 

house_2017_full['pooltypeid2'].fillna(0, inplace = True)

house_2017_full['pooltypeid7'].fillna(0, inplace = True)

#plot distribution of target variable log error

from scipy.stats import zscore

house_2017_full["logerror_zscore"] = zscore(house_2017_full["logerror"])

house_2017_full["is_outlier"] = house_2017_full["logerror_zscore"].apply(

  lambda x: x <= -2.5 or x >= 2.5

)



plt.figure(figsize=(12,8))

sb.distplot(house_2017_full[~house_2017_full['is_outlier']].logerror.values, bins=50, kde=False)

plt.xlabel('logerror', fontsize=12)

plt.title('logerror distribution')

plt.show()                     
#explore the missing value

missing_df = house_2017_full.isnull().sum(axis=0).reset_index()

missing_df.columns = ['column_name', 'missing_count']

missing_df = missing_df.loc[missing_df['missing_count']>0]

missing_df = missing_df.sort_values(by='missing_count')



ind = np.arange(missing_df.shape[0])

width = 0.9

fig, ax = plt.subplots(figsize=(12,18))

rects = ax.barh(ind, missing_df.missing_count.values, color='lightblue')

ax.set_yticks(ind)

ax.set_yticklabels(missing_df.column_name.values, rotation='horizontal')

ax.set_xlabel("Count of missing values")

ax.set_title("Number of missing values in each column")

plt.show() 
#for continues variables, plot the scatter plot

low_na_num_features = ['lotsizesquarefeet','finishedsquarefeet12','calculatedbathnbr','fullbathcnt','roomcnt','bedroomcnt','bathroomcnt','landtaxvaluedollarcnt', 'taxamount', 'structuretaxvaluedollarcnt', 'taxvaluedollarcnt']
house_2017_full[low_na_num_features].describe()
#explore categorical features with low missing values

low_na_cat_features = ['airconditioningtypeid', 'decktypeid', 'heatingorsystemtypeid', 

                       'hashottuborspa', 'heatingorsystemtypeid','regionidcounty', 'regionidcity',

                       'propertycountylandusecode','propertylandusetypeid','yearbuilt','assessmentyear']
#for continues variables, plot the scatter plot#correlation matrix to measure the corr between continuous variables

num_var = ['basementsqft', 'bathroomcnt', 'bedroomcnt', 'buildingqualitytypeid','buildingclasstypeid','threequarterbathnbr','calculatedfinishedsquarefeet',

           'calculatedbathnbr','finishedsquarefeet6','finishedsquarefeet12', 'finishedsquarefeet13', 'finishedsquarefeet15','finishedsquarefeet50','fireplacecnt',

           'fullbathcnt','garagecarcnt', 'garagetotalsqft','lotsizesquarefeet','poolsizesum', 'poolcnt', 'numberofstories', 'poolcnt', 'poolsizesum','roomcnt', 

           'unitcnt','yardbuildingsqft17','yearbuilt', 'yardbuildingsqft26', 'latitude', 'longitude','taxvaluedollarcnt', 'structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 

           'taxamount','logerror']

corr = house_2017_full[num_var].corr()

ig, ax = plt.subplots(figsize=(25,15)) 

sb.heatmap(corr, cmap="YlGnBu", annot=True, ax = ax)
# Drop Multicollinearity variables

house_2017_full.drop(['taxvaluedollarcnt', 'structuretaxvaluedollarcnt', 'landtaxvaluedollarcnt', 'bathroomcnt',

                      'fullbathcnt','finishedsquarefeet6', 'finishedsquarefeet12', 'finishedsquarefeet13','finishedsquarefeet15',

                      'finishedsquarefeet50'], axis=1, inplace=True)
#one way ANOVA analysis between cat variables has more than two levels and logerror

for i in range(0, len(low_na_cat_features)):

    formular = str('logerror ~ '+low_na_cat_features[i])

    model = ols(formular,data = house_2017_full).fit()

    anova_result = sm.stats.anova_lm(model, typ=2)

    print (anova_result)

    print("\n")
#t test between cat variables has two levels and taxamount

ttest_ind(house_2017_full['logerror'], house_2017_full['fireplaceflag'])
# since the dataset has many categorical variables, and using sklean random forest requires one-hot encoding

# so we use h2o random forest instead

# create h2o object

h2o.init()
#remove logerror outlier and other add-in variables

house_2017_temp = house_2017_full[~house_2017_full['is_outlier']]

house_2017_tree = house_2017_temp.drop(['logerror_zscore','is_outlier','parcelid'], axis =1)

# house_2017_full_hf = h2o.H2OFrame(house_2017_tree)

house_2017_tree.head(5)
#track the data shape

house_2017_tree.shape
#defind the model

# h2o_tree = H2ORandomForestEstimator(ntrees = 50, max_depth = 20, nfolds =10)

#train the model,if x not specify,model will use all x except the y column

# h2o_tree.train(y = 'logerror', training_frame = house_2017_full_hf)

#print variable importance

# h2o_tree_df = h2o_tree._model_json['output']['variable_importances'].as_data_frame()

#visualize the importance

"""

plt.rcdefaults()

fig, ax = plt.subplots(figsize = (10, 10))

variables = h2o_tree._model_json['output']['variable_importances']['variable']

y_pos = np.arange(len(variables))

scaled_importance = h2o_tree._model_json['output']['variable_importances']['scaled_importance']

ax.barh(y_pos, scaled_importance, align='center', color='green', ecolor='black')

ax.set_yticks(y_pos)

ax.set_yticklabels(variables)

ax.invert_yaxis()

ax.set_xlabel('Scaled Importance')

ax.set_title('Variable Importance')

plt.show()



#choose features have importance score >0.2

feature_score = 0.2

selected_features = h2o_tree_df[h2o_tree_df.scaled_importance>=feature_score]['variable']

selected_features

"""
selected_features = ['transactiondate', 'regionidneighborhood','taxamount','calculatedfinishedsquarefeet'

                     ,'yearbuilt','lotsizesquarefeet','propertyzoningdesc','garagetotalsqft'

                     ,'latitude','longitude','bedroomcnt','buildingqualitytypeid'

                     ,'calculatedbathnbr','yardbuildingsqft17']
selected_cols = (pd.Series(selected_features)).append(pd.Series(['logerror']))

#split data to training and test data set

X_train,X_test= train_test_split(house_2017_tree[selected_cols], test_size=0.33, random_state=42)

print(X_train.shape)

print(X_test.shape)
#transfer to h2o dataframe

X_train_h2o = h2o.H2OFrame(X_train)

X_test_h2o = h2o.H2OFrame(X_test)
param = {

      "ntrees" : 100

    , "learn_rate" : 0.02

    , "max_depth" : 10

    , "sample_rate" : 0.7

    , "col_sample_rate_per_tree" : 0.9

    , "min_rows" : 5

    , "seed": 4241

    , "score_tree_interval": 100

    ,  'nfolds': 10

    , "stopping_metric" : "MSE"

}

from h2o.estimators import H2OXGBoostEstimator

model = H2OXGBoostEstimator(**param)

model.train(y = 'logerror', training_frame = X_train_h2o)
#print training model summary

print(model.summary)
# create the new test set metrics

my_metrics = model.model_performance(test_data=X_test_h2o) 

my_metrics
hyper_params = {'max_depth' : [4,6,8,12,16,20]

               ,"learn_rate" : [0.1, 0.01, 0.0001]

               }

param_grid = {

      "ntrees" : 50

    , "sample_rate" : 0.7

    , "col_sample_rate_per_tree" : 0.9

    , "min_rows" : 5

    , "seed": 4241

    , "score_tree_interval": 100

    ,  'nfolds': 10

    , "stopping_metric" : "MSE"

}

model_grid = H2OXGBoostEstimator(**param_grid)
#Build grid search with previously made GBM and hyper parameters

#In a cartesian grid search, users specify a set of values for each hyperparamter that they want to search over, and H2O will train a model for every combination of the hyperparameter values. This means that if you have three hyperparameters and you specify 5, 10 and 2 values for each, your grid will contain a total of 5*10*2 = 100 models.

"""

grid = H2OGridSearch(model_grid,hyper_params,

                         grid_id = 'depth_grid',

                         search_criteria = {'strategy': "Cartesian"})





#Train grid search

grid.train(y='logerror',

           training_frame = X_train_h2o)

"""
#print the grid search results

# Get the grid results, sorted by validation AUC

"""

xgb_gridperf = grid.get_grid(sort_by='mse', decreasing=True)

xgb_gridperf



# Grab the top xgb model, chosen by validation mse

best_xgb = xgb_gridperf.models[17]



# Now let's evaluate the model performance on a test set

# so we get an honest estimate of top model performance

best_xgb_model = best_xgb.model_performance(test_data=X_test_h2o)

#0.1         6   depth_grid_model_4  0.006894977436261656



best_xgb_model.mse()

"""
#build the best model based on previous training

best_param = {

      "ntrees" : 100

    , "learn_rate" : 0.1

    , "max_depth" : 6

    , "sample_rate" : 0.7

    , "col_sample_rate_per_tree" : 0.9

    , "min_rows" : 5

    , "seed": 4241

    , "score_tree_interval": 100

    ,  'nfolds': 10

    , "stopping_metric" : "MSE"

}



best_model = H2OXGBoostEstimator(**best_param)

best_model.train(y = 'logerror', training_frame = X_train_h2o)
# create the test set metrics for the best model

best_metrics = best_model.model_performance(test_data=X_test_h2o) 

best_metrics