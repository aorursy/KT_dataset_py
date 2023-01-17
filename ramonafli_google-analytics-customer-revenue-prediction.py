# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# Importing libraries
import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime


# Function in order to flatten the JSON filds
def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df
%%time
train_df = load_df()
test_df = load_df("../input/test.csv")
train_df_copy = train_df
train_df_copy.head()
train_df_copy.info()
#sns.countplot(train_df_copy['geoNetwork.city'])
#plt.rcParams["figure.figsize"] = (25, 10)

train_df_copy['trafficSource.adwordsClickInfo.criteriaParameters'].value_counts()



# Having a look at unique values of geoNetwork.city, geoNetwork.metro, geoNetwork.region, 
# and trafficSource.campaign and trafficSource.adwordsClickInfo.criteriaParameters
# these fields have more than half missing values
# Dropping these columns
#train_df_copy['geoNetwork.city'].describe()
train_df_copy.drop(['trafficSource.adwordsClickInfo.criteriaParameters'],axis=1,inplace=True)
test_df.drop(['trafficSource.adwordsClickInfo.criteriaParameters'],axis=1,inplace=True)


train_df_copy.drop(['fullVisitorId','sessionId',
                    'visitId'],axis=1,inplace=True)
test_df.drop(['sessionId', 'visitId'],axis=1,inplace=True)
train_df_copy['trafficSource.adwordsClickInfo.gclId'].describe()
train_df_copy.drop(['trafficSource.adwordsClickInfo.gclId'],axis=1,inplace=True)
test_df.drop(['trafficSource.adwordsClickInfo.gclId'],axis=1,inplace=True)
for col in ['totals.bounces','totals.newVisits','totals.visits','totals.transactionRevenue']:
    train_df_copy[col].fillna(0,inplace = True)
train_df_copy['trafficSource.isTrueDirect'].fillna(False,inplace=True)    


for col in ['totals.bounces','totals.newVisits','totals.visits']:
    test_df[col].fillna(0,inplace = True)
test_df['trafficSource.isTrueDirect'].fillna(False,inplace=True)
# Check missing values
train_df_copy.isnull().sum().sort_values()
train_df_copy['totals.pageviews'].describe()
train_df_copy['trafficSource.adwordsClickInfo.page'].describe()

# We fill in the missing values with the most frequent occurence of these columns --> 1
train_df_copy['totals.pageviews'].fillna(1, inplace=True)
train_df_copy['trafficSource.adwordsClickInfo.page'].fillna(1, inplace=True)

test_df['totals.pageviews'].fillna(1, inplace=True)
test_df['trafficSource.adwordsClickInfo.page'].fillna(1, inplace=True)

# Converting totals.pageviews and trafficSource.adwordsClickInfo.page from object to integer
train_df_copy['totals.pageviews'] = train_df_copy['totals.pageviews'].astype(int)
train_df_copy['trafficSource.adwordsClickInfo.page'] = train_df_copy['trafficSource.adwordsClickInfo.page'].astype(int)

test_df['totals.pageviews'] = test_df['totals.pageviews'].astype(int)
test_df['trafficSource.adwordsClickInfo.page'] = test_df['trafficSource.adwordsClickInfo.page'].astype(int)

#trafficSource.campaignCode    
# Only one non-missing. Drop the column
train_df_copy.drop('trafficSource.campaignCode', axis = 1, inplace = True)

#test_df.drop('trafficSource.campaignCode', axis = 1, inplace = True)   #Doesnt exist in test data

train_df_copy['geoNetwork.city'].replace('not available in demo dataset',np.NaN,inplace=True)
train_df_copy['geoNetwork.metro'].replace('not available in demo dataset',np.NaN, inplace=True)
train_df_copy['geoNetwork.region'].replace('not available in demo dataset',np.NaN, inplace=True)
train_df_copy['trafficSource.campaign'].replace('(not set)',np.NaN,inplace=True)

test_df['geoNetwork.city'].replace('not available in demo dataset',np.NaN,inplace=True)
test_df['geoNetwork.metro'].replace('not available in demo dataset',np.NaN, inplace=True)
test_df['geoNetwork.region'].replace('not available in demo dataset',np.NaN, inplace=True)
test_df['trafficSource.campaign'].replace('(not set)',np.NaN,inplace=True)

#geoNetwork.city, geoNetwork.metro, geoNetwork.region, 
# and trafficSource.campaign
## fillna for the other objects
for col in ['trafficSource.keyword',
            'trafficSource.referralPath',
            'trafficSource.adwordsClickInfo.adNetworkType',
            'trafficSource.adwordsClickInfo.isVideoAd',
            'trafficSource.adwordsClickInfo.slot',
            'trafficSource.adContent',
            'geoNetwork.city',
            'geoNetwork.metro',
            'geoNetwork.region',
            'trafficSource.campaign'
            ]:
    
    train_df_copy[col].fillna('unknown', inplace=True)
    test_df[col].fillna('unknown', inplace=True)


# drop constant columns
constant_column = [col for col in train_df_copy.columns if train_df_copy[col].nunique() == 1]

print('drop columns:', constant_column)
train_df_copy.drop(constant_column, axis=1, inplace=True)

test_df.drop(constant_column, axis=1, inplace=True)
train_df_copy.info()
# totals.bounces, totals.hits, totals.newVisits
train_df_copy['totals.bounces'] = train_df_copy['totals.bounces'].astype(int)
train_df_copy['totals.hits'] = train_df_copy['totals.hits'].astype(int)
train_df_copy['totals.newVisits'] = train_df_copy['totals.newVisits'].astype(int)

train_df_copy['trafficSource.adwordsClickInfo.isVideoAd'] = train_df_copy['trafficSource.adwordsClickInfo.isVideoAd'].astype(bool)

test_df['totals.bounces'] = test_df['totals.bounces'].astype(int)
test_df['totals.hits'] = test_df['totals.hits'].astype(int)
test_df['totals.newVisits'] = test_df['totals.newVisits'].astype(int)

test_df['trafficSource.adwordsClickInfo.isVideoAd'] = test_df['trafficSource.adwordsClickInfo.isVideoAd'].astype(bool)
# Parsing the date. 
format_str = '%Y%m%d'
train_df_copy['formated_date'] = train_df_copy['date'].apply(lambda x: datetime.strptime(str(x), format_str))

train_df_copy['day'] = train_df_copy['formated_date'].apply(lambda x:x.day)
train_df_copy['weekday'] = train_df_copy['formated_date'].apply(lambda x:x.weekday())

train_df_copy.drop(['formated_date','date'], axis=1, inplace=True)

##TEST
test_df['formated_date'] = test_df['date'].apply(lambda x: datetime.strptime(str(x), format_str))

test_df['day'] = test_df['formated_date'].apply(lambda x:x.day)
test_df['weekday'] = test_df['formated_date'].apply(lambda x:x.weekday())

test_df.drop(['formated_date','date'], axis=1, inplace=True)



train_df_copy['day'] = train_df_copy['day'].astype('category')
train_df_copy['weekday'] = train_df_copy['weekday'].astype('category')

##TEST
test_df['day'] = test_df['day'].astype('category')
test_df['weekday'] = test_df['weekday'].astype('category')
train_df_copy.drop(['visitStartTime'], axis=1, inplace=True)
test_df.drop(['visitStartTime'], axis=1, inplace=True)

train_df_copy.info()
train_df_copy['totals.transactionRevenue'] = train_df_copy['totals.transactionRevenue'].astype(int)
def mean_transactionRevenue_plot(feature):
    ax = train_df_copy.groupby(feature)['totals.transactionRevenue'].mean().plot.bar()
    ax.set_ylabel('mean transaction revenue')



mean_transactionRevenue_plot('channelGrouping')
mean_transactionRevenue_plot('device.browser')
mean_transactionRevenue_plot('device.deviceCategory')
mean_transactionRevenue_plot('device.operatingSystem')
mean_transactionRevenue_plot('geoNetwork.continent')
mean_transactionRevenue_plot('geoNetwork.country')
mean_transactionRevenue_plot('geoNetwork.subContinent')
train_df_copy.groupby('totals.bounces')['totals.transactionRevenue'].mean()
mean_transactionRevenue_plot('totals.bounces')
mean_transactionRevenue_plot('totals.hits')
mean_transactionRevenue_plot('totals.newVisits')
mean_transactionRevenue_plot('totals.pageviews')
mean_transactionRevenue_plot('trafficSource.adContent')
train_df_copy['trafficSource.adwordsClickInfo.isVideoAd'].describe()
mean_transactionRevenue_plot('trafficSource.adwordsClickInfo.isVideoAd')
train_df_copy['trafficSource.adwordsClickInfo.page'].describe()
mean_transactionRevenue_plot('trafficSource.adwordsClickInfo.page')
train_df_copy['trafficSource.adwordsClickInfo.slot'].describe()
mean_transactionRevenue_plot('trafficSource.adwordsClickInfo.slot')
train_df_copy['trafficSource.isTrueDirect'].describe()
mean_transactionRevenue_plot('trafficSource.isTrueDirect')
train_df_copy['trafficSource.keyword'].describe()
train_df_copy.drop('trafficSource.keyword',axis=1,inplace=True)
test_df.drop('trafficSource.keyword',axis=1,inplace=True)
train_df_copy['trafficSource.medium'].describe()
mean_transactionRevenue_plot('trafficSource.medium')
train_df_copy['trafficSource.referralPath'].value_counts()
#train_df_copy.drop('trafficSource.referralPath',axis=1,inplace=True)
#test_df.drop('trafficSource.referralPath',axis=1,inplace=True)
train_df_copy['trafficSource.source'].describe()
mean_transactionRevenue_plot('trafficSource.source')
mean_transactionRevenue_plot('weekday')
mean_transactionRevenue_plot('geoNetwork.continent')
mean_transactionRevenue_plot('geoNetwork.country')
train_df_copy.info()
# %% Categorical columns
# List of categorical columns to recode
catCols = ['channelGrouping', 'device.browser','device.deviceCategory',
           'device.operatingSystem', 'geoNetwork.continent','geoNetwork.country',
           'geoNetwork.networkDomain',
           'geoNetwork.subContinent', 'trafficSource.adContent', 
           'trafficSource.adwordsClickInfo.adNetworkType',
           'trafficSource.adwordsClickInfo.slot',
           'trafficSource.source','trafficSource.medium',
           'day','weekday', 'trafficSource.referralPath',
           'geoNetwork.city', 
           'geoNetwork.metro', 
           'geoNetwork.region', 
           'trafficSource.campaign'
          ]

# Recode
for c in catCols:
    # Convert column to pd.Categotical
    train_df_copy[c] = pd.Categorical(train_df_copy[c])
    test_df[c] = pd.Categorical(test_df[c])

    # Extract the cat.codes and replace the column with these
    train_df_copy[c] = train_df_copy[c].cat.codes
    test_df[c] = test_df[c].cat.codes

    # Convert the cat codes to categotical...
    train_df_copy[c] = pd.Categorical(train_df_copy[c])
    test_df[c] = pd.Categorical(test_df[c])



# Generate a logical index of categorical columns to maybe use with LightGBM later
catCols = [i for i,v in enumerate(train_df_copy.dtypes) if str(v)=='category']
#train_df_copy.info()
train_df_copy.info()
#%% Prepare data
def prepLGB(data,
            classCol='',
            IDCol='',
            fDrop=[]):

        # Drop class column
        if classCol != '':
            labels = data[classCol]
            fDrop = fDrop + [classCol]
        else:
            labels = []

        if IDCol != '':
            IDs = data[IDCol]
        else:
            IDs = []

        if fDrop != []:
           data = data.drop(fDrop,
                            axis=1)

        # Create LGB mats
        lData = lgb.Dataset(data, label=labels,
                            free_raw_data=False,
                            feature_name=list(data.columns),
                            categorical_feature='auto')

        return lData, labels, IDs, data

train_df_copy['totals.transactionRevenue'] = np.log1p(train_df_copy['totals.transactionRevenue'])
import lightgbm as lgb
#reg = lgb.train(params, d_train, 100)
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

from sklearn.metrics import make_scorer, mean_squared_error,roc_auc_score, roc_curve

from sklearn.pipeline import Pipeline
import numpy as np
test_df.head()
# Split training data in to training and validation sets.
# Validation set is used for early stopping.
trainData, validData = train_test_split(train_df_copy,
                                        test_size=0.3)

# Prepare the data sets
trainDataL, trainLabels, trainIDs, trainData = prepLGB(trainData,
                                                 classCol='totals.transactionRevenue',
                                                 IDCol='',
                                                 fDrop=[])

validDataL, validLabels, validIDs, validData = prepLGB(validData,
                                                 classCol='totals.transactionRevenue',
                                                 IDCol='',
                                                 fDrop=[])

testDataL, _, _ , testData = prepLGB(test_df,
                                  classCol='',
                                     IDCol='fullVisitorId',
                                     fDrop=[])

# Prepare data set using all the training data
allTrainDataL, allTrainLabels, _ , allTrainData = prepLGB(train_df_copy,
                                                 classCol='totals.transactionRevenue',
                                                 IDCol='',
                                                 fDrop=[])
#inspecting the target variable
sns.kdeplot(np.log(train_df_copy['totals.transactionRevenue']))
#y = np.log1p(train_df_copy['totals.transactionRevenue'])
#X = train_df_copy.drop('totals.transactionRevenue',axis=1)
#X.info()
#https://www.kaggle.com/garethjns/microsoft-lightgbm-with-parameter-tuning-0-823
#https://lightgbm.readthedocs.io/en/latest/Python-API.html?highlight=fit
#grouped_test = test_df[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
#grouped_test.to_csv('submit.csv',index=False)
lgbm_model.get_params().keys()

import lightgbm as lgb
#reg = lgb.train(params, d_train, 100)
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.externals import joblib
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV

from sklearn.metrics import make_scorer, mean_squared_error,roc_auc_score, roc_curve

from sklearn.pipeline import Pipeline
import numpy as np

    
    #Hiper parameters to explore
params = {"learning_rate":np.arange(0.1,0.2,0.05),
              "metric": ['rmse'],
              "num_leaves": [70,80],
              "min_data_in_leaf": [100,1000],
              "max_depth": [7],
              'objective': ['regression'],
              'boosting': ['gbdt'],
          'bagging_fraction': [0.7,0.8],
        'bagging_freq': [1,3,5],
        'feature_fraction': [0.5,0.7],
        'max_bin': [128, 255],
          "min_child_samples" : [100]
             
                  }
                
#lgbm_model = lgb.train()
lgbm_model = lgb.LGBMRegressor(boosting_type= 'gbdt',
          objective = 'regression',
          verbose = 5,
          max_depth = params['max_depth'],
          learning_rate = params['learning_rate'],
          metric = ['rmse'],
                               num_leaves = params['num_leaves'],
                               min_data_in_leaf = params['min_data_in_leaf'],
                               bagging_fraction= params['bagging_fraction'],
        bagging_freq= params['bagging_freq'],
                               feature_fraction= params['feature_fraction'],
                               max_bin= params['max_bin'],
                               min_child_samples = params['min_child_samples']
                                         
                              ) #Regressor

# To view the default model params:

lgbm_model.get_params().keys()

grid = RandomizedSearchCV(lgbm_model, params, cv=5, n_jobs=4, verbose = 8)
grid.fit(allTrainData, allTrainLabels)



print(grid.best_params_)
print(grid.best_score_)
# Using parameters already set above, replace in the best from the grid search
params['max_depth'] = grid.best_params_['max_depth']
params['learning_rate'] = grid.best_params_['learning_rate']
params['num_leaves'] = grid.best_params_['num_leaves']
params['min_data_in_leaf'] = grid.best_params_['min_data_in_leaf']

params['min_child_samples'] = grid.best_params_['min_child_samples']

params['bagging_fraction'] = grid.best_params_['bagging_fraction']
params['bagging_freq'] = grid.best_params_['bagging_freq']
params['feature_fraction'] = grid.best_params_['feature_fraction']
params['max_bin'] = grid.best_params_['max_bin']


print('Fitting with params: ')
print(params)
k = 4

for i in range(0, k):
    print('Fitting model', k)

    # Prepare the data set for fold
    trainData, validData = train_test_split(train_df_copy,
                                            test_size=0.4)
    trainDataL, trainLabels, trainIDs, trainData = prepLGB(trainData,
                                                     classCol='totals.transactionRevenue',
                                                     IDCol='',
                                                     fDrop=[])
    validDataL, validLabels, validIDs, validData = prepLGB(validData,
                                                     classCol='totals.transactionRevenue',
                                                     IDCol='',
                                                     fDrop=[])
    # Train
    gbm = lgb.train(params,
                    trainDataL,
                    100000,
                    valid_sets=[trainDataL, validDataL],
                    early_stopping_rounds=50,
                    verbose_eval=4)

                    #valid_sets=[trainDataL, validDataL],

lgb.plot_importance(gbm,height=0.5,figsize=(10,10))
plt.show()
testData.head()
predsValid = 0
predsTrain = 0
predsTest = 0

predsValid += gbm.predict(validData,num_iteration=gbm.best_iteration)/k
predsTrain += gbm.predict(trainData,num_iteration=gbm.best_iteration)/k
predsTest += gbm.predict(testData.drop('fullVisitorId',axis=1),num_iteration=gbm.best_iteration)/k

#preds_Test = pd.concat(testData['fullVisitorId'],predsTest,axis=1)

predsTest[predsTest<0] = 0
sub = pd.DataFrame()
sub["PredictedLogRevenue"] = np.expm1(predsTest)
sub['fullVisitorId'] = testData['fullVisitorId']

sub = sub.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
sub.columns = ["fullVisitorId", "PredictedLogRevenue"]
sub["PredictedLogRevenue"] = np.log1p(sub["PredictedLogRevenue"])
sub.to_csv("baseline_lgb1.csv", index=False)
sub.head(100)
sub.head(100)
