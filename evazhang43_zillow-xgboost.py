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

house_2016_df = pd.read_csv('../input/zillow-prize-1/properties_2016.csv', na_values = missing_values, low_memory=False)

house_2017_df = pd.read_csv('../input/zillow-prize-1/properties_2017.csv', na_values = missing_values, low_memory=False)

house_log_2016 = pd.read_csv('../input/zillow-prize-1/train_2016_v2.csv', low_memory=False)

house_log_2017 = pd.read_csv('../input/zillow-prize-1/train_2017.csv', low_memory=False)
from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
print(house_2016_df.shape)

print(house_2017_df.shape)

print(house_log_2016.shape)

print(house_log_2017.shape)
#merge the trasaction dataset fro 2016-2017

#I drop the overlapped parcelid id 

house_log_full = pd.concat([house_log_2016,house_log_2017],ignore_index=True)

house_log_full2= house_log_full.drop_duplicates(subset= 'parcelid' )
#check if 2016 and 2017 is overlap 

len(set(house_2016_df['parcelid']).intersection(house_2017_df['parcelid']))
#so why use both year to join in 2017 house? 

#there are overlaps of parcelid id, 167888-165210= 2678



house_2017_full = house_2017_df.merge(house_log_full2, on = 'parcelid')
house_2017_full.shape
#house_2017_full has all th data that has log, so we need to find the test set 

#house_2017_full_2 = house_2017_df.merge(house_log_full, on = 'parcelid', how = "left")
#two method: 1 use the house_2017_full as the predict set, 2 use the unused as the predict set. 

#house_2017_full.shape
#house_2017_full_2[house_2017_full_2['logerror'].isnull()].shape
#submission= pd.read_csv("../input/zillow-prize-1/sample_submission.csv")
#submission.shape
#house_2017_full[house_2017_full['parcelid']==13850164]
#So there is a process of impute the boolean variables, but we didn't do that for the house_2017_full 

#impute boolean variables 

house_2017_full['fireplaceflag'].replace(True, 1, inplace=True)

house_2017_full['fireplaceflag'].fillna(0, inplace = True)

house_2017_full['hashottuborspa'].replace(True, 1, inplace=True)

house_2017_full['hashottuborspa'].fillna(0, inplace = True) 

house_2017_full['pooltypeid10'].fillna(0, inplace = True) 

house_2017_full['pooltypeid2'].fillna(0, inplace = True)

house_2017_full['pooltypeid7'].fillna(0, inplace = True)
#plot distribution of target variable log error

#1.9% outliner. 

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
h2o.init()
house_2017_temp = house_2017_full[~house_2017_full['is_outlier']]

house_2017_tree = house_2017_temp.drop(['logerror_zscore','is_outlier','parcelid'], axis =1)

house_2017_full_hf = h2o.H2OFrame(house_2017_tree)

house_2017_tree.head(5)



#house_2017_tree dropped the outlier, the parcelid id and logerror_zscore
#defind the model

h2o_tree = H2ORandomForestEstimator(ntrees = 50, max_depth = 20, nfolds =10)

#train the model,if x not specify,model will use all x except the y column

h2o_tree.train(y = 'logerror', training_frame = house_2017_full_hf)

#print variable importance

h2o_tree_df = h2o_tree._model_json['output']['variable_importances'].as_data_frame()

#visualize the importance



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

feature_score = 0.1

selected_features = h2o_tree_df[h2o_tree_df.scaled_importance>=feature_score]['variable']

selected_features

selected_features = ['regionidneighborhood','taxamount','calculatedfinishedsquarefeet'

                     ,'yearbuilt','lotsizesquarefeet','propertyzoningdesc','garagetotalsqft','bedroomcnt','buildingqualitytypeid'

                     ,'calculatedbathnbr','yardbuildingsqft17']

selected_cols = (pd.Series(selected_features)).append(pd.Series(['logerror']))

#split data to training and test data set

X_train,X_test= train_test_split(house_2017_tree[selected_cols], test_size=0.33, random_state=42)

print(X_train.shape)

print(X_test.shape)
len(X_train.columns)
X_test_pred= house_2017_df[selected_features]
X_test_pred.columns
X_test_pred.shape
X_train_h2o = h2o.H2OFrame(X_train)

X_test_h2o = h2o.H2OFrame(X_test)
X_test_pred_h2o = h2o.H2OFrame(X_test_pred)
from h2o.estimators import H2OXGBoostEstimator
X_test_h2o.columns
"""

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

"""
#print(model.summary)
"""

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

"""
"""

#grid = H2OGridSearch(model_grid,hyper_params,

                         grid_id = 'depth_grid',

                         search_criteria = {'strategy': "Cartesian"})





#Train grid search

#grid.train(y='logerror',

       #    training_frame = X_train_h2o)

       """
#xgb_gridperf = grid.get_grid(sort_by='mse', decreasing=True)

#xgb_gridperf
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
result=best_model.predict(X_test_pred_h2o)
#h2o.h2o.download_csv(result, "Predcited.csv")
result_list = h2o.h2o.as_list(result)
len(result_list)
house_2017_df["logerror"]=result_list

Submit = house_2017_df[["parcelid","logerror"]]
Submit["201610"]=result_list

Submit["201611"]=result_list

Submit["201612"]=result_list

Submit["201710"]=result_list

Submit["201711"]=result_list

Submit["201712"]=result_list

Submit = Submit.drop("logerror",axis=1)

Submit.columns = ["ParcelId", "201610","201611","201612","201710","201711","201712"]

Submit.to_csv("Submission.csv", index = False )