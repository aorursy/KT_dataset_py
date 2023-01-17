import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



import os

for dirname, _, filenames in os.walk('/kaggle'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
### Importing Libraries or Packages that are needed throughout the Program ###

import numpy as np

import pandas as pd

import xgboost as xgb

import random

import datetime

import gc

import seaborn as sns 

color = sns.color_palette()



import sys

pd.options.display.max_columns = None

pd.options.mode.chained_assignment = None

pd.options.display.float_format



from sklearn.model_selection import train_test_split

import matplotlib as mpl

import matplotlib.pyplot as plt

import matplotlib.pylab as pylab

%matplotlib inline
train2016_df = pd.read_csv('/kaggle/input/zillow-prize-1/train_2016_v2.csv' , parse_dates=["transactiondate"])

train2017_df = pd.read_csv('/kaggle/input/zillow-prize-1/train_2017.csv' , parse_dates=["transactiondate"])



prop2016_df = pd.read_csv('/kaggle/input/eda-v22-zillow/properties_2016_proc.csv', index_col=0)

prop2017_df = pd.read_csv('/kaggle/input/eda-v22-zillow/properties_2017_proc.csv', index_col=0)

test = pd.read_csv('/kaggle/input/zillow-prize-1/sample_submission.csv')

print("Training 2016 transaction: " + str(train2016_df.shape))

print("Training 2017 transaction: " + str(train2016_df.shape))

print("Number of Property 2016: " + str(prop2016_df.shape))

print("Number of Property 2017: " + str(prop2017_df.shape))

print("Sample Size: " + str(test.shape))
prop2016_df.head()
df_imp=prop2016_df[['latitude','longitude']]

from sklearn import preprocessing



x = df_imp #returns a numpy array

min_max_scaler = preprocessing.MinMaxScaler()

x_scaled = min_max_scaler.fit_transform(x)

df = pd.DataFrame(x_scaled)
df.head()
# Choosing the optimal k

from scipy.spatial.distance import cdist, pdist

from sklearn.cluster import KMeans



k_range = range(1,10)

# Try clustering the data for k values ranging 1 to 10

k_means_var = [KMeans(n_clusters = k).fit(df) for k in k_range]

centroids = [X.cluster_centers_ for X in k_means_var]



k_euclid = [cdist(df, cent, 'euclidean') for cent in centroids]

dist = [np.min(ke, axis=1) for ke in k_euclid]



# Calculate within-cluster sum of squares

wcss = [sum(d**2) for d in dist]



# Visualize the elbow method for determining k

import matplotlib.pyplot as plt

plt.plot(k_range, wcss)

plt.xlabel('Range of k')

plt.ylabel('RSS within cluster')

plt.title('plot of Lattitude V/S Longtitude')

plt.show()
kmeans = KMeans(n_clusters=4, random_state=0).fit(df)

labels = kmeans.labels_

#Glue back to originaal data

df['clusters'] = labels

df2 = df.rename(columns = {0 : 'Lattitude', 1: 'Longtitude'})

#Add the column into our list
sns.lmplot('Lattitude', 'Longtitude', data = df2, fit_reg=False,hue="clusters",  scatter_kws={"marker": "D", "s": 100})

plt.title('Lattitude v/s Longtitude')

plt.xlabel('Lattitude')

plt.ylabel('Longtitude')

plt.show()
del kmeans, df, df2, wcss, dist

gc.collect()

print('Memory usage reduction…')
train_2016 = train2016_df.merge(prop2016_df, how='left', on='parcelid')

train_2017 = train2016_df.merge(prop2017_df, how='left', on='parcelid')

train = pd.concat([train_2016, train_2017], axis=0, ignore_index=True)
catvars = ['airconditioningtypeid','buildingqualitytypeid',

           'decktypeid','fips','hashottuborspa', 'fireplaceflag','heatingorsystemtypeid','yearbuilt',

           'taxdelinquencyflag', 'assessmentyear']



#numvars = [i for i in prop2016_df.columns if i not in catvars]

#print ("Có {} numeric và {} categorical columns".format(len(numvars),len(catvars)))



# Some variables take on very many categorical values. 

# For the sake of this exercise, we'll drop them.

cols_to_drop = [

    'location_1', 'location_2', 'location_3', 'location_4',

    'rawcensustractandblock',

    'propertycountylandusecode',

    'propertylandusetypeid',

    'propertyzoningdesc',

    'regionidzip',

]



# Dropping selected columns

df_known = train.drop(cols_to_drop, axis=1)



# Re-encoding categorical variables

df_known_cat = pd.get_dummies(df_known, columns=catvars)



df_known_cat = df_known_cat.drop('transactiondate', axis=1)
dtype_df = df_known_cat.dtypes.reset_index()

dtype_df.columns = ['Count', 'Column type']

dtype_df.groupby('Column type').aggregate('count').reset_index()
from scipy.cluster import hierarchy as hc

import scipy

corr = np.round(scipy.stats.spearmanr(df_known).correlation, 4)

corr_condensed = hc.distance.squareform(1-corr)

z = hc.linkage(corr_condensed, method='average')

fig = plt.figure(figsize=(16,8))

dendrogram = hc.dendrogram(z, labels=df_known.columns, orientation='left', leaf_font_size=16)

plt.show()
df_known_cat.head()
# Creating our variables and targets

X = df_known_cat.drop(["logerror", "parcelid"], axis=1)

y = df_known_cat["logerror"]



# Randomly splitting into a training and a validation set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
#del properties, train,test

gc.collect()

print('Memory usage reduction…')
from sklearn.preprocessing import LabelEncoder

lbl = LabelEncoder()



# Dropping selected columns

prop_2016_known = prop2016_df.drop(cols_to_drop, axis=1)

# Re-encoding categorical variables

prop_2016_known_cat = pd.get_dummies(prop_2016_known, columns=catvars)



# Dropping selected columns

prop_2017_known = prop2017_df.drop(cols_to_drop, axis=1)

# Re-encoding categorical variables

prop_2017_known_cat = pd.get_dummies(prop_2017_known, columns=catvars)
dtype_df = prop_2016_known_cat.dtypes.reset_index()

dtype_df.columns = ['Count', 'Column type']

dtype_df.groupby('Column type').aggregate('count').reset_index()
prop_2017_known_cat.head()
# Grid search

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import r2_score

from xgboost import XGBRegressor



params = {

    'min_child_weight':[4,5],

    'max_depth': range (2, 10, 1),

    'n_estimators': range(60, 220, 40),

    'learning_rate': [0.1, 0.01, 0.05],

    'objective': ['reg:linear'],

    'eval_metric': ['mae'],

}



X = df_known_cat.drop(["parcelid", "logerror"], axis=1)

Y = df_known_cat["logerror"]

X, X_Val, Y, Y_Val = train_test_split(X, Y)



# Initialize XGB and GridSearch

xgb = XGBRegressor(nthread=-1) 



grid = GridSearchCV(xgb, params, cv = 3)

grid.fit(X, Y)

best_params = grid.best_params_



# Print the r2 score

print(r2_score(Y_Val, grid.best_estimator_.predict(X_Val))) 

print("\n========================================================")

print(" Results from Grid Search " )

print("========================================================")    

print("\n The best estimator across ALL searched params:\n", grid.best_estimator_)

print("\n The best score across ALL searched params:\n", grid.best_score_)

print("\n The best parameters across ALL searched params:\n", grid.best_params_)
%%time



# Maximum number of trees we will collect

num_rounds = 300



# Transforming our data into XGBoost's internal DMatrix structure

dtrain = xgb.DMatrix(X_train, y_train)

dvalid = xgb.DMatrix(X_test, y_test)



# Training

xgb_params = {

    'min_child_weight':best_params['min_child_weight'],

    'max_depth': best_params['max_depth'],

    'n_estimators': best_params['n_estimators'],

    'learning_rate': best_params['learning_rate'],

    'objective': 'reg:linear',

    'eval_metric': 'mae',

}



#xgb = XGBRegressor(nthread=-1)

model = XGBRegressor(xgb_params,   # Training parameters

                     num_rounds    # Max number of trees

                    )

model.fit(X_train,y_train)



# Best score obtained

print("Best score: ", model.best_score)

print("Number iteration: ",model.best_iteration)





# plot the important features #

fig, ax = plt.subplots(figsize=(12,18))

xgb.plot_importance(model, max_num_features=50, height=0.8, ax=ax)

plt.show()
def transform_test_features(features_2016, features_2017):

    return features_2016.drop(['parcelid'], axis=1), features_2017.drop(['parcelid'], axis=1)



def predict_and_export(models, features_2016, features_2017, file_name):

    # Construct DataFrame for prediction results

    submission_2016 = pd.DataFrame()

    submission_2017 = pd.DataFrame()

    submission_2016['ParcelId'] = features_2016.parcelid

    submission_2017['ParcelId'] = features_2017.parcelid

    

    test_features_2016, test_features_2017 = transform_test_features(features_2016, features_2017)

    

    pred_2016, pred_2017 = [], []

    for i, model in enumerate(models):

        print("Start model {} (2016)".format(i))

        pred_2016.append(model.predict(test_features_2016))

        print("Start model {} (2017)".format(i))

        pred_2017.append(model.predict(test_features_2017))

    

    # Take average across all models

    mean_pred_2016 = np.mean(pred_2016, axis=0)

    mean_pred_2017 = np.mean(pred_2017, axis=0)

    

    submission_2016['201610'] = [float(format(x, '.4f')) for x in mean_pred_2016]

    submission_2016['201611'] = submission_2016['201610']

    submission_2016['201612'] = submission_2016['201610']



    submission_2017['201710'] = [float(format(x, '.4f')) for x in mean_pred_2017]

    submission_2017['201711'] = submission_2017['201710']

    submission_2017['201712'] = submission_2017['201710']

    

    submission = submission_2016.merge(how='inner', right=submission_2017, on='ParcelId')

    

    print("Length of submission DataFrame: {}".format(len(submission)))

    print("Submission header:")

    print(submission.head())

    submission.to_csv(file_name, index=False)

    return submission, pred_2016, pred_2017 
file_name = 'v8_xgboost_single.csv'

submission, pred_2016, pred_2017 = predict_and_export([model], prop_2016_known_cat, prop_2017_known_cat, file_name)