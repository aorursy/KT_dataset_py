# Load libraries
import numpy as np # linear algebra
import pandas as pd # data processing
from pandas import Series, DataFrame # to deal with time data
import gc # To collect RAM garbage
import time # To get current time, used to calculate model training time
from sklearn.model_selection import train_test_split # To split training and validation datasets
import matplotlib.pyplot as plt # For plotting feature importance
import lightgbm as lgb # Light gbm model
import warnings
warnings.filterwarnings('ignore') # Toignore warnings
# Set debug mode. When debug=1, we'll only be importing a few lines.
# When debug=0, we'll import a much larger dataset to do serious training.
# You'll see how to set this up in a later code block. 
# Make sure to set debug=1 throughout the workshop! 

debug=1
# Define data types
# uint32 is an unsigned integer with 32 bit 
# which means that you can represent 2^32 numbers (0-4294967295)
dtypes = {
            'ip'            : 'uint32',
            'app'           : 'uint16',
            'device'        : 'uint8',
            'os'            : 'uint8',
            'channel'       : 'uint16',
            'is_attributed' : 'uint8',
            'click_id'      : 'uint32',
            }
# Only import columns you need: create a list before you actually read the data
train_cols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed']

# Exercise: create a list called test_cols with the following feature: 
# 'ip','app','device','os', 'channel', 'click_time', 'click_id'



# It takes a long time to load a large dataset, so we print a mark here just to keep track of the process
print("Loading training data...")

# Now reading the training and test data.
# Load only a few lines if debug=1; Load a much larger part of the dataset if debug=0
if debug:
    train = pd.read_csv("../input/train.csv", dtype=dtypes, parse_dates=['click_time'], 
                        nrows=100000, usecols=train_cols)
    test = pd.read_csv("../input/test.csv", dtype=dtypes, parse_dates=['click_time'], 
                       nrows=100000, usecols=test_cols)
else: 
    train =  pd.read_csv("../input/train.csv", dtype=dtypes, parse_dates=['click_time'], 
                         # skiprows=range(1,129903891), this will skip the first n rows
                         nrows=1000000, usecols=train_cols)
    test = pd.read_csv("../input/test.csv", dtype=dtypes, parse_dates=['click_time'], 
                       usecols=test_cols)
print ("Loading finished")
# Exercise: Print a sentence to indicate that we are now "processing data"


# First, we have to get the length of the training data. 
# We'll need this number when we have to split the training and test data again
len_train = len(train)    

# Now append test data to training data and make a new data frame called full
full=train.append(test)   
# Now we have stored both training and test data in a new dataframe called full
# We can delete training and test data because we don't need them anymore and they are very large
del test  
del train 

# Collect any other temp garbage in the RAM. 
# It's a good habit to gc.collect() from time to time when you deal with large datasets
gc.collect() 
# Assign a list for predictors and target. We'll use these two lists very soon
predictors = ["ip","app","device","os","channel"]
target = "is_attributed"

# Create a list with names of categorical variables. LGBM can handle categorical variables smoothly.
categorical = ["ip", "app", "device","os", "channel"]
# Getting time features
# Get "day" from "click_time" and add it to "full" as a new feature
full['day'] = pd.to_datetime(full.click_time).dt.day.astype('int8')
# Append "day" to the predictor list
predictors.append('day')

# Get "hour" from "click_time" and add it to "full" as a new feature
full['hour'] = pd.to_datetime(full.click_time).dt.hour.astype('int8')
# Append "hour" to the predictor list
predictors.append('hour')

# Exercise: get "minute" and "second" from "click_time" and add them to "full" as new features


# Exercise: Append "minute" and "second" to the predictor list

# shift means we are shifting the whole column up by one row, which basically means we're getting the next value in line
# We subtract the time of current click from the time of next click, so we get the time difference between two clicks
# We convert this difference to seconds, and claim that its data type is "float32", which means real number with 32 bit
same=["channel", "app", "os", "device","ip"]
full['next_click'] = (full.groupby(same).click_time.shift(-1) - full.click_time).dt.seconds.astype('float32')

# Append "next_click" to the predictor list
predictors.append('next_click')
# Exercise: using very similar method, create a new feature called "prev_click", 
# indicating the time difference between the current click and the previuos click
# Hint: just change -1 to +1, and switch the two click_time

# Set the group. You can have other combinations
group = ['ip','day','hour']

# group by ip+day+hour, choose one column (click_time) to count the number of rows, 
# fill this number into the click_time variable, then change the column name to ip_day_hour
gp = full.groupby(group)["click_time"].count().reset_index().rename(index=str, columns={'click_time':"ip_day_hour"})

# merge back with full data
full = full.merge(gp, on=group, how='left')

# Append new variable name to the list "prdictors"
predictors.append("ip_day_hour")

# Delete gp and collect garbage
del gp
gc.collect()
# Exercise: Very similarly, create a new variable called ip_app_channel, 
# which calculates the total number of clicks for the same ip + app + channel
# Delete and collect garbage

# Split training and test data
train = full[:len_train]
test = full[len_train:]

# Set X(predictors) and y(target)
X = train[predictors]
y = train[target]
    
# Delete unused parts 
del train
gc.collect()
# Split training and validation data using train_test_split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2, random_state = 30)

# delete X and y since we don't need them anymore
del X, y

    
# lgb.Dataset defines the training and validation dataset
# this is a little bit confusing, but in lgb.Dataset, "label" means y(the target variable), 
# because our prediction is actually "labelling" the target
# We also define the feature names for feature importance plotting afterwards
xgtrain = lgb.Dataset(X_train.values, label=y_train.values, feature_name=predictors, categorical_feature = categorical)                          
xgvalid = lgb.Dataset(X_val.values, label=y_val.values, feature_name=predictor,categorical_feature = categoricals)

# Setting lgb model parameters; not mandatory
lgb_params = {
        'boosting_type': 'gbdt', # Gradient Boosted Decision Trees
        'objective': 'binary', # Because we are predicting 0 and 1
        'metric': 'auc', # Method to evaluate the model, auc means "area under the curve". The lower the better
        'learning_rate': 0.03, #Basically the weight for each boosting iteration. A smaller learning_rate may increase accuracy but lead to slower training speed. 
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit. Too deep may lead to overfitting. 
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.6,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 8, # Number of threads using for training models, better to set it large for large dataset
        'verbose': 0, # Do not affect training, just affect how detailed the information produced during training would be
        'random_state': 42
    }
# Training start! Print a marker for it. 
print("Training...")

# Set start_time as current time
start_time = time.time()

# Create an empty data frame for writing the evaluation results. 
evals_results={}

# This is the real lgb model training process
# There are more parameters to tune! I put those parameters that we are most likely to change here

model_lgb = lgb.train(lgb_params, # The list of parameters that we have already set
                xgtrain,  # The training dataset
                valid_sets= [xgtrain, xgvalid], # We produce evaluation score for both the training and validation dataset
                valid_names=['train','valid'],  # Assign names to the training and validation dataset
                early_stopping_rounds=100, # If there's no improvement after 10 rounds, then stop
                verbose_eval=50,  # Print evaluation scores every 10 rounds
                num_boost_round=5000, # Maximum 200 rounds, even if it does not meet the early_stopping_round requirement
                evals_result=evals_results) # Write evalution results into evals_results
                
# Print current time - start_time, this is the time used for training the model    
print('Model training time: {} seconds'.format(time.time() - start_time))

gc.collect()

# Exercise: change early_Stopping_round, verbose_eval, and num_boost_round, and train the model again
# Print our how much time the training took
# Note: you may need to run the revious block to reset xgtrain and xgvalid because lgbm has already processed categorical data
# List feature importance for all features
print("Features importance...")
gain = model_lgb.feature_importance('gain')
ft = pd.DataFrame({'feature':model_lgb.feature_name(), 
                   'split':model_lgb.feature_importance('split'), 
                   'gain':100 * gain / gain.sum()}).sort_values('gain', ascending=False)
print(ft)
# Plot feature importance using the "split" method
split = lgb.plot_importance(model_lgb, importance_type="split")
plt.show(split) # Show the plot
plt.savefig("feature_importance_split.png") # Save the plot in the output
# Exercise: Plot feature importance using the "gain" method

# Print a mark here
print ("Predicting test data...")

# Creat X_test, which includes all features in the test dataset
X_test = test[predictors]

# Feed X_test to our trained "model_lgb" to predict the target variable (y) in the test dataset
# Store our prediction in ypred
ypred = model_lgb.predict(X_test,num_iteration=model_lgb.best_iteration)

gc.collect()
# Print a mark
print ("Writing submission file...")

# Read the sample submission file
submission = pd.read_csv("../input/sample_submission.csv")

# Change the value in the prediction column into our prediction "ypred"
submission["is_attributed"] = ypred

# Write it into a csv file
submission.to_csv("submission.csv", index = False)

# Print a final mark
print ("Mission Completed")