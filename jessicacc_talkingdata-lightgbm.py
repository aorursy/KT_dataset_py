import pandas as pd
import time
import numpy as np
import gc
import matplotlib.pyplot as plt
import seaborn as sns

import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score, roc_curve, auc, classification_report
from sklearn.model_selection import GridSearchCV
np.random.seed(42)
sub_file_name = '03-sub_lgb_balanced99.csv'
#exp_path = '../../experiments/exp_3/'
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }

print('loading train data...')
train_df = pd.read_csv('../input/talkingdata-adtracking-fraud-detection/train.csv', skiprows=range(1,144903891), nrows=40000000, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])

print('loading test data...')
test_df = pd.read_csv('../input/talkingdata-adtracking-fraud-detection/test.csv', dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])

len_train = len(train_df)

# Join the datasets to apply the transformations only one time
train_df=train_df.append(test_df)

train_df.shape, test_df.shape
print('Train Data:')
display(train_df.head())
display(train_df.tail())

print('Test Data:')
display(test_df.head())
display(test_df.tail())
del test_df
gc.collect()
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')
train_df['minute'] = pd.to_datetime(train_df.click_time).dt.minute.astype('uint8')
train_df['second'] = pd.to_datetime(train_df.click_time).dt.second.astype('uint8')

gc.collect()
# Define all the groupby transformations
GROUPBY_AGGREGATIONS = [
    # V1 - GroupBy Features #
    # Variance in day, for ip-app-channel
    {'groupby': ['ip','app','channel'], 'select': 'day', 'agg': 'var'},
    # Variance in hour, for ip-app-os
    {'groupby': ['ip', 'app', 'os'], 'select': 'hour', 'agg': 'var'},
    # Variance in hour, for ip-day-channel
    {'groupby': ['ip','day','channel'], 'select': 'hour', 'agg': 'var'},
    # Count, for ip-day-hour
    {'groupby': ['ip','day','hour'], 'select': 'channel', 'agg': 'count'},
    # Count, for ip-app
    {'groupby': ['ip', 'app'], 'select': 'channel', 'agg': 'count'},        
    # Count, for ip-app-os
    {'groupby': ['ip', 'app', 'os'], 'select': 'channel', 'agg': 'count'},
    # Count, for ip-app-day-hour
    {'groupby': ['ip','app','day','hour'], 'select': 'channel', 'agg': 'count'},
    # Mean hour, for ip-app-channel
    {'groupby': ['ip','app','channel'], 'select': 'hour', 'agg': 'mean'}, 
    # V2 - GroupBy Features #
    # Average clicks on app by distinct users; is it an app they return to?
    {'groupby': ['app'], 
     'select': 'ip', 
     'agg': lambda x: float(len(x)) / len(x.unique()), 
     'agg_name': 'AvgViewPerDistinct'
    },
    # How popular is the app or channel?
    {'groupby': ['app'], 'select': 'channel', 'agg': 'count'},
    {'groupby': ['channel'], 'select': 'app', 'agg': 'count'},
#     # V3 - GroupBy Features                                              #
#     # https://www.kaggle.com/bk0000/non-blending-lightgbm-model-lb-0-977 #
    {'groupby': ['ip'], 'select': 'channel', 'agg': 'nunique'}, 
    {'groupby': ['ip'], 'select': 'app', 'agg': 'nunique'}, 
    {'groupby': ['ip','day'], 'select': 'hour', 'agg': 'nunique'}, 
    {'groupby': ['ip','app'], 'select': 'os', 'agg': 'nunique'}, 
#     {'groupby': ['ip'], 'select': 'device', 'agg': 'nunique'}, 
#     {'groupby': ['app'], 'select': 'channel', 'agg': 'nunique'}, 
#     {'groupby': ['ip', 'device', 'os'], 'select': 'app', 'agg': 'nunique'}, 
#     {'groupby': ['ip','device','os'], 'select': 'app', 'agg': 'cumcount'}, 
#     {'groupby': ['ip'], 'select': 'app', 'agg': 'cumcount'}, 
#     {'groupby': ['ip'], 'select': 'os', 'agg': 'cumcount'}, 
#     {'groupby': ['ip','day','channel'], 'select': 'hour', 'agg': 'var'}    
]

# Apply all the groupby transformations
for spec in GROUPBY_AGGREGATIONS:
    # Name of the aggregation we're applying
    agg_name = spec['agg_name'] if 'agg_name' in spec else spec['agg']
    # Name of new feature
    new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), agg_name, spec['select'])
    # Info
    print("Grouping by {}, and aggregating {} with {}".format(
        spec['groupby'], spec['select'], agg_name))
    # Unique list of features to select
    all_features = list(set(spec['groupby'] + [spec['select']]))
    # Perform the groupby
    gp = train_df[all_features]. \
        groupby(spec['groupby'])[spec['select']]. \
        agg(spec['agg']). \
        reset_index(). \
        rename(index=str, columns={spec['select']: new_feature})
    # Merge back to X_total
    if 'cumcount' == spec['agg']:
        train_df[new_feature] = gp[0].values
    else:
        train_df = train_df.merge(gp, on=spec['groupby'], how='left')
        
    # Clear memory
    del gp
    gc.collect()

train_df.head()
gc.collect()
HISTORY_CLICKS = {
    'identical_clicks': ['ip', 'app', 'device', 'os', 'channel'],
    'app_clicks': ['ip', 'app']
}

# Go through different group-by combinations
for fname, fset in HISTORY_CLICKS.items():
    # Clicks in the past
    train_df['prev_'+fname] = train_df.groupby(fset).cumcount().rename('prev_'+fname)
#     # Clicks in the future
#     train_df['future_'+fname] = train_df.iloc[::-1].groupby(fset).cumcount().rename('future_'+fname).iloc[::-1]

# Count cumulative subsequent clicks
train_df.head()
train_df.head()
# only train data
display(train_df[:len_train].isna().sum())

# only test data
display(train_df[len_train:].isna().sum())
len_train
test_df = train_df[len_train:]
val_df = train_df[(len_train-2500000):len_train]
train_df = train_df[:(len_train-2500000)]

train_df.shape, test_df.shape, val_df.shape
print('Train Data:')
display(train_df.head())
display(train_df.tail())

print('Val Data:')
display(val_df.head())
display(val_df.tail())

print('Test Data:')
display(test_df.head())
display(test_df.tail())
# function
def lgb_modelfit_nocv(params, dtrain, dvalid, predictors, target='target', objective='binary', metrics='auc',
                       feval=None, early_stopping_rounds=20, num_boost_round=3000, verbose_eval=10, 
                      categorical_features=None):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.01,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
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
        'nthread': 4,
        'verbose': 0,
        'metric':metrics
    }

    lgb_params.update(params)
    
    print("preparing validation datasets")
    xgtrain = lgb.Dataset(dtrain[predictors].values, label=dtrain[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features)
    xgvalid = lgb.Dataset(dvalid[predictors].values, label=dvalid[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical_features)
    evals_results = {}
    bst1 = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgtrain, xgvalid], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=feval)
    n_estimators = bst1.best_iteration
    
    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics+":", evals_results['valid'][metrics][n_estimators-1])
    
    print('\nPlot - Feature Importance')
    lgb.plot_importance(bst1)
    plt.show()
    
    return bst1
# defininf the features and target

target = 'is_attributed'

predictors = ['app','device','os', 'channel', 'hour', 'day', 
              'ip_day_hour_count_channel', 'ip_day_channel_var_hour', 
              'ip_app_count_channel',
              'ip_app_os_count_channel', 'ip_app_os_var_hour',
              'ip_app_channel_var_day','ip_app_channel_mean_hour',
             'ip_app_day_hour_count_channel']

print('Total predictors: {}'.format(len(predictors)))


categorical = ['app', 'device', 'os', 'channel', 'hour', 'day']

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')
gc.collect()
print("Training...")
start_time = time.time()

params = {
    'learning_rate': 0.15,
    #'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': 7,  # 2^max_depth - 1
    'max_depth': 3,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': 0.7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.9,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight':99 # because training data is extremely unbalanced 
}
bst = lgb_modelfit_nocv(params, 
                        train_df, 
                        val_df, 
                        predictors, 
                        target, 
                        objective='binary', 
                        metrics='auc',
                        early_stopping_rounds=30, 
                        verbose_eval=True, 
                        num_boost_round=500, 
                        categorical_features=categorical)

print('[{}]: model training time'.format(time.time() - start_time))


# del train_df
# del val_df
# gc.collect()

print("Predicting...")
sub['is_attributed'] = bst.predict(test_df[predictors])

print("writing to file...")
sub.to_csv(sub_file_name,index=False)

print("done...")
print("Predicting in validation dataset...")

predictions_lgbm_valdf_prob = bst.predict(val_df[predictors])

predictions_lgbm_valdf = np.where(predictions_lgbm_valdf_prob > 0.5, 1, 0) #Turn probability to 0-1 binary output
#Print accuracy
acc_lgbm = accuracy_score(val_df['is_attributed'], predictions_lgbm_valdf)
print('Overall accuracy of Light GBM model:', acc_lgbm)
#Print Area Under Curve
plt.figure()
false_positive_rate, recall, thresholds = roc_curve(val_df['is_attributed'], predictions_lgbm_valdf)

roc_auc = auc(false_positive_rate, recall)

plt.title('Receiver Operating Characteristic (ROC)')
plt.plot(false_positive_rate, recall, 'b', label = 'AUC = %0.3f' %roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1], [0,1], 'r--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.ylabel('Recall')
plt.xlabel('Fall-out (1-Specificity)')
plt.show()

print('AUC score:', roc_auc)
#Print Confusion Matrix

plt.figure()
cm = confusion_matrix(val_df['is_attributed'], predictions_lgbm_valdf)

labels = ['App Not Downloaded', 'App Downloaded']
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot = True, fmt='d',vmin = 0.2);
plt.title('Confusion Matrix')
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()
# classification report

report = classification_report(val_df['is_attributed'], predictions_lgbm_valdf)

print(report)
gc.collect()
