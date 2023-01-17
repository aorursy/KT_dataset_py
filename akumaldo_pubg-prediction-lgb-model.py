# Python ≥3.5 is required

import sys

assert sys.version_info >= (3, 5)



# Scikit-Learn ≥0.20 is required

import sklearn

assert sklearn.__version__ >= "0.20"



# Common imports

import numpy as np

import os

import gc



import matplotlib.pyplot as plt

import seaborn as sns

palette = sns.color_palette('Paired', 10)



import numpy as np

import pandas as pd

# Pandas display options

pd.set_option('display.float_format', lambda x: '%.3f' % x)





#setting fontsize and style for all the plots

plt.style.use('fivethirtyeight')

plt.rcParams['font.size'] = 18

plt.rcParams['figure.figsize'] = (16,5)



%matplotlib inline 

#plotting directly without requering the plot()



import warnings

warnings.filterwarnings(action="ignore") #ignoring most of warnings, cleaning up the notebook for better visualization



pd.set_option('display.max_columns', 500) #fixing the number of rows and columns to be displayed

pd.set_option('display.max_rows', 500)



print(os.listdir("../input")) #showing all the files in the ../input directory



# Set random seed 

randomseed = 42



# Any results you write to the current directory are saved as output. Kaggle message :D
train = pd.read_csv('../input/train_V2.csv')

test = pd.read_csv('../input/test_V2.csv')



print('Train dataset shape: {}'.format(train.shape))

print('Test dataset shape: {}'.format(test.shape))
train.head()
train.info()
train.nunique()
train.describe()
fig = plt.figure(figsize=(16,10))

plt.subplot(2,1,1)

sns.countplot(train['kills'])

plt.xlabel('kills',fontsize = 15,color='blue')

plt.ylabel('Count',fontsize = 15,color='blue')

plt.subplot(2,1,2)

sns.countplot(train['headshotKills'])

plt.xlabel('Head Shot Kills',fontsize = 15,color='blue')

plt.ylabel('Count',fontsize = 15,color='blue')
train[train['kills'] >= 8]['kills'].value_counts()
fig = plt.figure(figsize=(16,10))

plt.subplot(4,1,1)

sns.scatterplot(x='kills', y='winPlacePerc', data=train.sample(500000, random_state = randomseed))

plt.xlabel('kills',fontsize = 15,color='blue')

plt.ylabel('Win Percentage',fontsize = 15,color='blue')

plt.subplot(4,1,2)

sns.scatterplot(x='revives', y='winPlacePerc', data=train.sample(500000, random_state = randomseed))

plt.xlabel('revives',fontsize = 15,color='blue')

plt.ylabel('Win Percentage',fontsize = 15,color='blue')

plt.subplot(4,1,3)

sns.scatterplot(x='headshotKills', y='winPlacePerc', data=train.sample(500000, random_state = randomseed))

plt.xlabel('headshot Kills',fontsize = 15,color='blue')

plt.ylabel('Win Percentage',fontsize = 15,color='blue')

plt.subplot(4,1,4)

sns.scatterplot(x='damageDealt', y='winPlacePerc', data=train.sample(500000, random_state = randomseed))

plt.xlabel('Damage Dealt',fontsize = 15,color='blue')

plt.ylabel('Win Percentage',fontsize = 15,color='blue')

plt.tight_layout(h_pad=1.5)
def point_plot(x,target, df):

    fig = plt.figure(figsize=(16,10))

    sns.set_context("notebook", font_scale=1.5)

    num_plots = len(x)

    for i, variable in enumerate(x):

        plt.subplot(num_plots,1,1+i)

        sns.pointplot(x=variable,y=target,data=df,color='#606060',alpha=0.8)

        plt.xlabel('',fontsize = 15,color='blue')

        plt.ylabel('Target variable: {}'.format(target),fontsize = 15,color='blue')

        plt.title(variable + "/" + target,fontsize = 20,color='blue')

    plt.tight_layout(h_pad=1.5)
point_plot(['vehicleDestroys','weaponsAcquired'],'winPlacePerc', train)
point_plot(['heals','boosts'],'winPlacePerc', train)
point_plot(['kills','revives','headshotKills'],'winPlacePerc', train)
train.isnull().sum()
train = train.dropna()
#let's create this function to make it easier and clean to fit the model and use the cross_val_score and obtain results

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler,MinMaxScaler,Imputer, RobustScaler

import time #implementing in this function the time spent on training the model

from sklearn.metrics import mean_squared_error,mean_absolute_error

from sklearn.model_selection import  GridSearchCV,train_test_split

from sklearn.ensemble import RandomForestRegressor

from catboost import CatBoostRegressor

import lightgbm as lgb

import xgboost as xgb

import eli5

from eli5.sklearn import PermutationImportance



import gc



#imputing all NaN value(if any) and scalling

pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")), 

        #('scale', MinMaxScaler(feature_range = (0, 1))),

        ('robustScaler', RobustScaler()),

])







#Generic function for making a classification model and accessing performance:

def fit_model(train, train_labels, test_set, params={},model=None, 

                         GridSearch=False, plot_features_importances=False):

    

    time_start = time.perf_counter() #start counting the time

    #creating our validation set out of the training set and labels provided

    

    X_train, x_val, y_train, y_val = train_test_split(train, train_labels, test_size=0.1, random_state=randomseed)

    X_train = pipeline.fit_transform(X_train) #fiting and transforming the dataset using the pipeline provided

    x_val = pipeline.fit_transform(x_val)

    

    test_sub = np.zeros(test_set.shape[0])

    test_set = pipeline.fit_transform(test_set)

    

    predict_val = np.zeros(train.shape[0])

    score = {}

    

    if model != None: grid_model = GridSearchCV(model, params,verbose=1, cv=3) #initializing the grid search model



    if GridSearch:

        grid_model.fit(X_train, y_train)

        score_grid = grid_model.best_score_

        

        #predicting using the model that has been trained above

        

        predict_val = grid_model.predict(x_val)

        score['MAE'] = mean_absolute_error(y_val, predict_val)

        score['RMSE'] = np.sqrt(mean_squared_error(y_val, predict_val))

        

        print("Model Report")



        print("MAE: "+ str(score["MAE"]))

        print("RMSE: "+ str(score["RMSE"]))

        print('\n')

    

        test_sub = grid_model.predict(test_set) 



    #################### PLOTTING FEATURES IMPORTANCE ####################

    

    # Sort features according to importance

    if plot_features_importances:

        if GridSearch:

            # Extract feature importances

            feature_importances = pd.DataFrame({'feature': list(train.columns), 'importance': grid_model.best_estimator_.feature_importances_})

        else:

            feature_importances = pd.DataFrame({'feature': list(train.columns), 'importance': model.feature_importances_})

        

        feature_importances = feature_importances.sort_values('importance', ascending = False).reset_index()



        # Normalize the feature importances to add up to one

        feature_importances['importance_normalized'] = feature_importances['importance'] / feature_importances['importance'].sum()



        # Make a horizontal bar chart of feature importances

        plt.figure(figsize = (10, 6))

        ax = plt.subplot()



        # Need to reverse the index to plot most important on top

        ax.barh(list(reversed(list(feature_importances.index[:15]))), 

                feature_importances['importance_normalized'].head(15), 

                align = 'center', edgecolor = 'k')



        # Set the yticks and labels

        ax.set_yticks(list(reversed(list(feature_importances.index[:15]))))

        ax.set_yticklabels(feature_importances['feature'].head(15))



        # Plot labeling

        plt.xlabel('Normalized Importance'); plt.title('Feature Importances')



    #perm = PermutationImportance(grid_model, random_state=randomseed).fit(x_val,y_val)

    #eli5.show_weights(perm)

    

    time_end = time.perf_counter() #end of counting the time

    

    total_time = time_end-time_start #total time spent during training and cross_validation

    

    print("Amount of time spent during training the model and cross validation: %4.3f seconds" % (total_time))

    

    

    # Clean up memory

    gc.enable()

    del model, X_train, x_val, y_train, y_val,score, total_time, time_end, time_start,predict_val,test_set

    gc.collect()

                        

    return test_sub
train.columns
# add basic player-level features by combining other features together

def add_player_features(X):

    X['headshot_rate'] = X['headshotKills'] / (X['kills'] + 0.00001)

    X['kill_streak_rate'] = X['killStreaks'] / (X['kills'] + 0.00001)

    X['kills_assists'] = X['kills'] + X['assists']

    X['heals_boosts'] = X['heals'] + X['boosts']

    X['total_distance'] = X['walkDistance'] + X['rideDistance'] + X['swimDistance']

    X['kills_assists_per_heal_boost'] = X['kills_assists'] / (X['heals_boosts'] + 1)

    X['damageDealt_per_heal_boost'] = X['damageDealt'] / (X['heals_boosts'] + 1)

    X['road_kills_per_rideDistance'] = X['roadKills'] / (X['rideDistance'] + 0.01)

    X['maxPlace_per_numGroups'] = X['maxPlace'] / X['numGroups']

    X['assists_per_kill'] = X['assists'] / (X['kills'] + X['assists'] + 0.0001)

    X['killPlace'] = X['killPlace'] - 1

    X['teamwork'] = X['assists'] + X['revives']

    agg = X.groupby(['groupId']).size().to_frame('players_in_team')

    X = X.merge(agg, how='left', on=['groupId'])

    X['headshotKills_over_kills'] = X['headshotKills'] / X['kills']

    X['headshotKills_over_kills'].fillna(0, inplace=True)

    X['killPlace_over_maxPlace'] = X['killPlace'] / X['maxPlace']

    X['killPlace_over_maxPlace'].fillna(0, inplace=True)

    X['killPlace_over_maxPlace'].replace(np.inf, 0, inplace=True)

    return X
corr = train.corr() #Let's take a look at the pearson's corr, just to have an overall view of how the attributes influence the price.

#using this correlation, we can have an idea of the linear correlation, positive and negative.

ax = sns.set(rc={'figure.figsize':(40,25)})

plt.xticks(fontsize=30)

plt.yticks(fontsize=30)

sns.heatmap(corr, annot=True,xticklabels=corr.columns.values,

    yticklabels=corr.columns.values,

    linecolor='white',

    linewidths=0.1,

    cmap="RdBu").set_title('Pearsons Correlation Factors Heat Map', color='blue', size='20')

gc.enable()

del corr

gc.collect()
train = add_player_features(train)

test = add_player_features(test)



cols_to_drop = ['Id','groupId','matchId','matchType',

                'headshotKills', 'killStreaks', 'walkDistance', 'rideDistance', 'swimDistance', 'heals']



dummy_train = pd.get_dummies(train['matchType'])

train = pd.concat([train, dummy_train], axis=1)

dummy_test = pd.get_dummies(test['matchType'])

test = pd.concat([test, dummy_test], axis=1)



print('Training set shape after creating dummies: {}'.format(train.shape))

print('Testing set shape after creating dummies: {}'.format(test.shape))





train_labels = train['winPlacePerc']

train_prepared = train.drop(cols_to_drop + ['winPlacePerc'], axis=1)

test_prepared = test.drop(cols_to_drop, axis=1)



print('Training set shape without ids and the target: {}'.format(train_prepared.shape))

print('Testing set shape without ids: {}'.format(test_prepared.shape))
test_prepared.head()
# Create the random forest

params={}

random_forest = RandomForestRegressor(n_estimators = 20, max_depth = 20,oob_score = True,

                                      bootstrap = True, verbose = 1, n_jobs = -1)



prediction_random = fit_model(train_prepared,train_labels, test_prepared, params=params,model=random_forest,GridSearch=True,

                           plot_features_importances=True)
params_lgb = {

        'boosting_type':'gbdt',

        'objective': 'regression',

        'num_leaves': 31,

        'learning_rate': 0.05,

        'max_depth': -1,

        'subsample': 0.8,

        'subsample_freq': 1,

        'colsample_bytree': 0.6,

        'reg_aplha': 1,

        'reg_lambda': 0.001,

        'metric': 'rmse',

        'min_split_gain': 0.5,

        'min_child_weight': 1,

        'min_child_samples': 10,

        'scale_pos_weight':1

    }



model_lgb = lgb.LGBMRegressor(**params_lgb, n_estimators = 3000, nthread = 4, n_jobs = -1)





prediction_lgb = fit_model(train_prepared,train_labels, test_prepared, params={},model=model_lgb,GridSearch=True,

                           plot_features_importances=True)
sample_sub = pd.read_csv('../input/sample_submission_V2.csv')



sub_lgb = pd.DataFrame({'Id': sample_sub['Id'], 'winPlacePerc': prediction_lgb})

sub_lgb.to_csv('LGB_model_sub.csv', index = False)

sub_lgb.head()