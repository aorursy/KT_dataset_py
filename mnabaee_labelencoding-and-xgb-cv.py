import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import sklearn

print('The scikit-learn version is {}.'.format(sklearn.__version__))
#Load the training and test files

df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')

print('training: ', df_train.shape)

print('test: ', df_test.shape)
#Convert to Numpy arrays and separate features/targets

training_samples = df_train.as_matrix()

training_targets = training_samples[:,-1]

training_samples = training_samples[:,1:-1]



test_samples = df_test.as_matrix()

test_samples = test_samples[:,1:]
plt.hist(training_targets[np.where(training_targets < 15000)], bins = 200, color='r', normed=True)

plt.grid(True)

plt.xlabel('Target Value')

plt.ylabel('Normalized Frequency')

plt.title('Distribution of target values from training set.')

plt.show()
#Encode the Labels of the categorical data

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

# [0:116]

allLabels = np.concatenate( ( training_samples[:, 0:116].flat , test_samples[:, 0:116].flat ) )

le.fit( allLabels )

del allLabels

print(le.classes_)
#Transform the labels to int values

for colIndex in range(116):

    training_samples[:, colIndex] = le.transform(training_samples[:, colIndex])

    test_samples[:, colIndex] = le.transform( test_samples[:, colIndex] )
training_samples = training_samples.astype(np.float)

test_samples = test_samples.astype(np.float)

print(training_samples.shape)

print(test_samples.shape)
from scipy.stats import skew, boxcox

#Calculate the skew of the features

for featureIdx in range( training_samples.shape[1] ):

    train_test_feature_values = np.concatenate( (training_samples[:,featureIdx], test_samples[:,featureIdx] ), axis=0 )

    skew_ = skew(train_test_feature_values )

    #Transform the numeric features with high skew values

    if abs(skew_) > 0.25 and featureIdx >= 116:

        print(skew_)

        train_test_feature_values = train_test_feature_values + 1

        transformed_feature_values, lm = boxcox( train_test_feature_values )

        training_samples[:,featureIdx] = transformed_feature_values[0:training_samples.shape[0]]

        test_samples[:,featureIdx] = transformed_feature_values[training_samples.shape[0]:]

        
#Train and Cross-Validate using the splits

#We will use xgb for regression with CV for grid search

import xgboost as xgb



#The parameters are taken from these two kernels: 

#    https://www.kaggle.com/mmueller/allstate-claims-severity/stacking-starter, 

#    https://www.kaggle.com/tilii7/allstate-claims-severity/bias-correction-xgboost

# with minor changes

xgb_params = {

    'seed': 0,

    'colsample_bytree': 0.3085,

    'silent': 1,

    'subsample': 0.7,

    'learning_rate': 0.01,

    'objective': 'reg:linear',

    'max_depth': 7,

    'num_parallel_tree': 1,

    'min_child_weight': 4.2922,

    'eval_metric': 'mae',

    'eta':0.1,

    'gamma': 0.5290,

    'subsample':0.9930,

    'max_delta_step':0,

    'booster':'gbtree',

    'nrounds': 1001

}



dtrain = xgb.DMatrix( training_samples, label=training_targets)

xgb_cv_res = xgb.cv(xgb_params, dtrain, num_boost_round=2001, nfold=4, seed = 0, stratified=False,

             early_stopping_rounds=25, verbose_eval=50, show_stdv=True)



print('finished cv.')

xgb_cv_res.plot(y=['test-mae-mean', 'train-mae-mean'], grid=True, logx=True)

plt.xlabel('Round')

plt.ylabel('mae')

plt.show()



best_nrounds = xgb_cv_res.shape[0] - 1

xgb_best = xgb.train(xgb_params, dtrain, best_nrounds)   
#Predit for the test data

dtest = xgb.DMatrix( test_samples)

pred_test = xgb_best.predict(dtest)

print(pred_test)
#Save results to csv file

df_res = pd.DataFrame(df_test, columns=['id'])

df_res['loss'] = pred_test

print(df_res.iloc[0])

df_res.to_csv('result.csv', index=False)