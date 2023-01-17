import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import gc # Garbage Collector
df_train = pd.read_csv("../input/train_sample.csv", parse_dates=['click_time', 'attributed_time'])
print('Data frame column types:')
print(df_train.dtypes)
print("\n")
print('Glimpse:')
print(df_train.head())
df_test = pd.read_csv("../input/test.csv", parse_dates=['click_time'])
print(f'Data has no missing value? {df_train.isnull().values.any()}')
print('app download frequency (0 - no, 1 - yes):')
print(df_train['is_attributed'].value_counts())
print('percentage:')
print(df_train['is_attributed'].value_counts(normalize=True))
def handle_time(df_name):
    df_name['click_day'] = df_name['click_time'].dt.day
    df_name['click_hour'] = df_name['click_time'].dt.hour
#     df_train['click_minute'] = df_train['click_time'].dt.minute
#     df_train['click_second'] = df_train['click_time'].dt.second
    return df_name

df_train = handle_time(df_train)
df_test = handle_time(df_test)
from sklearn.model_selection import train_test_split

X_train, X_validation, y_train, y_validation = train_test_split(df_train.drop(['attributed_time', 'click_time'], axis=1), \
                                                                df_train['is_attributed'], test_size=0.33, random_state=0)

del df_train
gc.collect()
feature_combinations = [        
    ['app', 'channel'],
    ['app', 'os'],
    ['app', 'device'],
    ['app', 'device', 'os']
]

for cols in feature_combinations:
    calc_df = X_train.groupby(cols)['is_attributed'].apply(lambda x: x.sum() / float(x.count())) 
    calc_df = calc_df.to_frame()
    calc_df.rename(columns={'is_attributed': '_'.join(cols)+'_conv_rate'}, inplace=True)

    X_train = X_train.join(calc_df, on=cols, how='left', sort=False)
    X_validation = X_validation.join(calc_df, on=cols, how='left', sort=False)
    X_validation.fillna(0, inplace=True)

    df_test = df_test.join(calc_df, on=cols, how='left', sort=False)
    df_test.fillna(0, inplace=True)

    del calc_df

gc.collect()
# https://www.kaggle.com/nanomathias/feature-engineering-importance-testing. Simplified. Only keep some combinations. 
# Define all the groupby transformations
click_aggregations = [
    # Variance in hour, for ip-app-os
    # {'groupby': ['ip','app','os'], 'select': 'click_hour', 'agg': 'var'},
    # Count, for ip-app-os
    {'groupby': ['ip', 'app', 'os'], 'select': 'click_hour', 'agg': 'count'},
    # Count, for ip-app-day-hour
    {'groupby': ['ip','app','click_day','click_hour'], 'select': 'channel', 'agg': 'count'},    
    # How popular is the app or channel?
    {'groupby': ['app'], 'select': 'channel', 'agg': 'count'},
    {'groupby': ['channel'], 'select': 'app', 'agg': 'count'}    
]


# Apply all the groupby transformations
for spec in click_aggregations:

    # Name pattern of new feature
    new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), spec['agg'], spec['select'])

    # Info
#     print("Grouping by {}, and aggregating {} with {}".format(
#         spec['groupby'], spec['select'], spec['agg']
#     ))

    # Unique list of features to select
    all_features = list(set(spec['groupby'] + [spec['select']]))

    # Perform the groupby
    gp = X_train[all_features]. \
        groupby(spec['groupby'])[spec['select']]. \
        agg(spec['agg']). \
        reset_index(). \
        rename(columns={spec['select']: new_feature})

    # Merge back to total data frame
    if 'cumcount' == spec['agg']:
        X_train[new_feature] = gp[0].values
        X_validation[new_feature] = gp[0].values
        X_validation.fillna(0, inplace=True)
        
        df_test[new_feature] = gp[0].values
        df_test.fillna(0, inplace=True)
    else:
        X_train = X_train.merge(gp, on=spec['groupby'], how='left')
        X_validation = X_validation.merge(gp, on=spec['groupby'], how='left')
        X_validation.fillna(0, inplace=True)
        
        df_test = df_test.merge(gp, on=spec['groupby'], how='left')
        df_test.fillna(0, inplace=True)
     # Clear memory
    del gp
    gc.collect()

gc.collect()
for col in ['os', 'app', 'device', 'channel']:
    print(f'Number of unique {col} in training data: {X_train[col].nunique()}')
for col in ['os', 'app', 'device', 'channel']:
    print(f'Number of unique {col} in testing data: {df_test[col].nunique()}')
from sklearn.feature_extraction import FeatureHasher 

FH = FeatureHasher(n_features=6, input_type='string') # device will have hash collision
for col in ['os', 'app', 'device', 'channel']:
    newcolnm = col+'_FH'
    newcolnm = pd.DataFrame(FH.transform(X_train[col].astype('str')).toarray()).add_prefix(col)
    X_train = X_train.join(newcolnm)
    X_validation = X_validation.join(newcolnm)
    X_validation.fillna(0, inplace=True)
    
    df_test = df_test.join(newcolnm)
    df_test.fillna(0, inplace=True)
    del newcolnm
    gc.collect()
del FH
    
gc.collect()

# from h2o.estimators import H2ORandomForestEstimator
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
X_train.drop(['ip', 'app', 'device', 'os', 'channel', 'is_attributed'], axis=1, inplace=True)
X_validation.drop(['ip', 'app', 'device', 'os', 'channel', 'is_attributed'], axis=1, inplace=True)

df_test.drop(['ip', 'app', 'device', 'os', 'channel', 'click_time'], axis=1, inplace=True)
# # Define model
# clf_rf = RandomForestClassifier(random_state=0)
# # Train model
# clf_rf.fit(X_train, y_train)
clf_adab = AdaBoostClassifier(n_estimators=200, random_state=0)
clf_adab.fit(X_train, y_train)
# importances = clf_rf.feature_importances_
 
# print ("Random Forest Sorted Feature Importance:")
# sorted_feature_importance = sorted(zip(importances, list(X_train)), reverse=True)
# print (sorted_feature_importance)
# print ('\n')
# print(f'Random Forest AUC: {roc_auc_score(y_validation, clf_rf.predict_proba(X_validation)[:, 1])}')

# del importances
importances = clf_adab.feature_importances_

print ("AdaBoost Sorted Feature Importance:")
sorted_feature_importance = sorted(zip(importances, list(X_train)), reverse=True)
print (sorted_feature_importance)
print(f'AdaBoost AUC: {roc_auc_score(y_validation, clf_adab.predict_proba(X_validation)[:, 1])}')
del importances
del X_train, y_train, X_validation, y_validation
gc.collect()
split_size = 20
test_df_list = np.array_split(df_test, split_size, axis=0)
submission_df_list = []
for i, test_df_chunk in reversed(list(enumerate(test_df_list))):
    test_df_chunk['is_attributed'] = clf_adab.predict_proba(test_df_chunk.drop('click_id', axis=1))[:, 1]
    submission_df_list.append(test_df_chunk[['click_id', 'is_attributed']])
    del test_df_list[i]
    gc.collect()
del df_test
gc.collect()
result = pd.concat(submission_df_list)
del submission_df_list
gc.collect()

result.sort_values(by='click_id', inplace=True)
result.to_csv('adaboost_submission.csv', header=True, index=False)