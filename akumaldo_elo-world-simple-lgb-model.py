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

import numpy as np

import pandas as pd



%matplotlib inline 

#plotting directly without requering the plot()



import warnings

warnings.filterwarnings(action="ignore") #ignoring most of warnings, cleaning up the notebook for better visualization



pd.set_option('display.max_columns', 500) #fixing the number of rows and columns to be displayed

pd.set_option('display.max_rows', 500)



print(os.listdir("../input")) #showing all the files in the ../input directory



# Any results you write to the current directory are saved as output. Kaggle message :D
train = pd.read_csv('../input/train.csv', parse_dates=['first_active_month'], low_memory=True)

test = pd.read_csv('../input/test.csv', low_memory=True, parse_dates=['first_active_month'])

df_new_merch = pd.read_csv('../input/new_merchant_transactions.csv', parse_dates=True, low_memory=True)

df_hist_trans = pd.read_csv('../input/historical_transactions.csv',parse_dates=True, low_memory=True)



print('Training set shape: {}'.format(train.shape))

print('Testing set shape: {}'.format(train.shape))

print('Merchants set shape: {}'.format(df_new_merch.shape))

print('Historical transactions set shape: {}'.format(df_hist_trans.shape))
train.head()
test.head()
print('Max date for the training set: {}'.format(train['first_active_month'].max()))

print('Max date for the testing set: {}'.format(test['first_active_month'].max()))
train.describe()
train['year'] = train['first_active_month'].dt.year

train['month'] = train['first_active_month'].dt.month

test['year'] = test['first_active_month'].dt.year

test['month'] = test['first_active_month'].dt.month

plt.subplot(3,1,1)

train['target'].plot.hist(bins=30, edgecolor='black', figsize=(16,10))

plt.subplot(3,1,2)

sns.countplot(train['year'])

plt.subplot(3,1,3)

sns.countplot(train['month'])

plt.xticks(rotation='vertical')

plt.title('Some title')

plt.tight_layout(h_pad=0.5)
train[train['target'] < - 30]['target'].value_counts()
plt.figure(figsize = (16, 12))





train['log_target'] = np.log1p(train['target'])





# iterate through the sources

for i, features in enumerate(['feature_1', 'feature_2', 'feature_3']):

    

    # create a new subplot for each source

    plt.subplot(3, 1, i + 1)

    # plot repaid loans

    sns.kdeplot(train.loc[train['log_target'] >= 0, features], label = 'log_target >= 0')

    # plot loans that were not repaid

    sns.kdeplot(train.loc[train['log_target'] < 0, features], label = 'log_target < 0')

    

    # Label the plots

    plt.title('Distribution of %s by log 1p target Value' % features)

    plt.xlabel('%s' % features); plt.ylabel('Density');

    

plt.tight_layout(h_pad = 1.5)



train.drop('log_target', axis=1, inplace=True)

df_hist_trans.head()
group_purch = df_hist_trans.groupby('card_id')['purchase_amount'].size().reset_index()

group_purch.columns = ["card_id", "count_purchase_amount"]



train = pd.merge(train,

                 group_purch,

                 on='card_id',

                 how='left')



test = pd.merge(test,

                 group_purch,

                 on='card_id',

                 how='left')





print('Training set shape after merging with historical transaction: {}'.format(train.shape))



print('Testing set shape after merging with historical transaction: {}'.format(test.shape))



gc.enable

del df_hist_trans, group_purch

gc.collect()
train.head()
train.isnull().sum()
train.describe()
fig = plt.figure(figsize=(16,5))

plt.subplot(2,1,1)

plt.scatter(train['count_purchase_amount'], train['target'], c='c')

plt.subplot(2,1,2)

train.set_index('first_active_month')['count_purchase_amount'].plot()
df_new_merch.head()
group_card_temp = df_new_merch.groupby('card_id')['purchase_amount'].size().reset_index()



group_card_temp.columns = ['card_id', 'count_new_merch_purchases']



train = pd.merge(train,

                group_card_temp,

                on='card_id',

                how='left')



test = pd.merge(test,

                group_card_temp,

                on='card_id',

                how='left')



print('Training set shape after merging with new merchant transaction: {}'.format(train.shape))



print('Testing set shape after merging with new merchant transaction: {}'.format(test.shape))



train['count_new_merch_purchases'].fillna(0,inplace=True)

test['count_new_merch_purchases'].fillna(0,inplace=True)





del group_card_temp, df_new_merch

gc.collect()
train.head()
train.isnull().sum()
fig = plt.figure(figsize=(16,5))

plt.subplot(2,1,1)

sns.scatterplot(x='count_purchase_amount', y='target', data= train)

plt.title('Relationship between target value and purchase amount')

plt.subplot(2,1,2)

sns.scatterplot(x='count_new_merch_purchases', y='target', data= train)

plt.title('Relationship between target value and new merchants purchase amount')

plt.tight_layout(h_pad=0.5)
fig = plt.figure(figsize=(16,5))

plt.subplot(2,1,1)

train.set_index('first_active_month')['count_purchase_amount'].plot(c='b')

plt.title('Purchase amount through time')

plt.subplot(2,1,2)

train.set_index('first_active_month')['count_new_merch_purchases'].plot(c='r')

plt.title('New merchantes purchase amount through time')

plt.tight_layout(h_pad=0.5)
fig = plt.figure(figsize=(16,8))

corr_train = train.corr()

sns.heatmap(corr_train, cmap = plt.cm.RdYlBu_r, vmin = -0.25, annot = True, vmax = 0.6)

plt.title('Correlation Heatmap');
train_labels = train['target'].copy()

mask = ['target','first_active_month','card_id']

cols = [col for col in train.columns if col not in mask]

train_prepared = train.loc[:, cols]

test_prepared = test.loc[:, cols]



print('Training set: {}'.format(train_prepared.shape))

print('testing set: {}'.format(test_prepared.shape))
 #let's use the Imputer to fill the NAN values with the median value

from sklearn.impute import SimpleImputer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler,MinMaxScaler,Imputer, RobustScaler

from sklearn.metrics import mean_absolute_error, mean_squared_error

import time #implementing in this function the time spent on training the model

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV,cross_val_score,train_test_split, KFold

import lightgbm as lgb

import gc



#imputing all NaN value

pipeline = Pipeline([

        ('imputer', SimpleImputer(strategy="median")), 

        #('scale', MinMaxScaler(feature_range = (0, 1))),

        ('robustScaler', RobustScaler()),

])



nfolds = 10

folds = KFold(n_splits=nfolds, shuffle=True,random_state=42)





#Generic function for making a classification model and accessing performance:

def fit_model(train, train_labels, test_set, params={}, 

                         fold=folds, model=None, 

                         GridSearch=False, plot_features_importances=False):

    

    time_start = time.perf_counter() #start counting the time

    #creating our validation set out of the training set and labels provided

    X_train, x_val, y_train, y_val = train_test_split(train, train_labels, test_size=0.1, random_state=42)

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

        

    else:

        model = lgb.LGBMRegressor(**params, n_estimators = 5000, nthread = 4, n_jobs = -1)



        for n, (index, val_index) in enumerate(folds.split(train)):

            

            print('Starting Fold number: %d' %n)

            X, X_val = train.values[index], train.values[val_index]

            Y, Y_val = train_labels[index], train_labels[val_index]

            X = pipeline.fit_transform(X)

            X_val = pipeline.fit_transform(X_val)

            

            model.fit(X, Y, 

                    eval_set=[(X, Y), (X_val, Y_val)],

                    verbose=1000, early_stopping_rounds=200)

            

            predict_val = model.predict(X_val)

            test_temp = model.predict(test_set, num_iteration=model.best_iteration_)

            test_sub += test_temp

            

            if score == {}:

                score['MAE'] = mean_absolute_error(Y_val, predict_val)

                score['RMSE'] = np.sqrt(mean_squared_error(Y_val, predict_val))            

            else:

                score['MAE'] += mean_absolute_error(Y_val, predict_val)

                score['RMSE'] += np.sqrt(mean_squared_error(Y_val, predict_val))



                        

        test_sub /= nfolds

                        

        print("Model Report")



        print("MAE: "+ str(score["MAE"]/nfolds))

        print("RMSE: "+ str(score["RMSE"]/nfolds))

        

        print('\n')



        

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

    

    time_end = time.perf_counter() #end of counting the time

    

    total_time = time_end-time_start #total time spent during training and cross_validation

    

    print("Amount of time spent during training the model and cross validation: %4.3f seconds" % (total_time))

    # Clean up memory

    gc.enable()

    del model, X_train, x_val, y_train, y_val,score, total_time, time_end, time_start,predict_val,test_set

    gc.collect()

                        

    return test_sub
########## LGB ########

params = {

          'num_leaves': 30,

         'min_data_in_leaf': 20,

         'objective': 'regression',

         'max_depth': 10,

         'learning_rate': 0.01,

         "boosting": "gbdt",

         "feature_fraction": 0.9,

         "bagging_freq": 1,

         "bagging_fraction": 0.9,

         "bagging_seed": 11,

         "metric": 'rmse',

         "lambda_l1": 0.2,

}

prediction_lgb = fit_model(train_prepared,train_labels, test_prepared, params=params, plot_features_importances=True)
sub_df = pd.DataFrame({"card_id":test["card_id"].values})

sub_df["target"] = prediction_lgb

sub_df.to_csv("lgbm.csv", index=False)

sub_df.head()