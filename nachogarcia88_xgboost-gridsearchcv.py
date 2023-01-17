import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))
from sklearn.metrics import roc_curve, auc
import xgboost as xgb
print('Loading the training/test data using pandas...')
train = pd.read_csv('../input/training.csv')
test = pd.read_csv('../input/test.csv')
check_agreement = pd.read_csv('../input/check_agreement.csv')
check_correlation = pd.read_csv('../input/check_correlation.csv')
def add_features(df):
    # features from phunter
    df['flight_dist_sig'] = df['FlightDistance']/df['FlightDistanceError']
    df['flight_dist_sig2'] = (df['FlightDistance']/df['FlightDistanceError'])**2
    df['NEW_IP_dira'] = df['IP']*df['dira']
    df['p0p2_ip_ratio']=df['IP']/df['IP_p0p2']
    df['p1p2_ip_ratio']=df['IP']/df['IP_p1p2']
    df['DCA_MAX'] = df.loc[:, ['DOCAone', 'DOCAtwo', 'DOCAthree']].max(axis=1)
    df['iso_bdt_min'] = df.loc[:, ['p0_IsoBDT', 'p1_IsoBDT', 'p2_IsoBDT']].min(axis=1)
    df['iso_min'] = df.loc[:, ['isolationa', 'isolationb', 'isolationc','isolationd', 'isolatione', 'isolationf']].min(axis=1)
    df['NEW_FD_LT']=df['FlightDistance']/df['LifeTime']
    return df
train = add_features(train)
test = add_features(test)
check_agreement = add_features(check_agreement)
check_correlation = add_features(check_correlation)
filter_out = ['id','min_ANNmuon', 'production', 'mass', 'signal', 'SPDhits', 'isolationb', 'isolationc', 'DOCAone', 'DOCAtwo', 'DOCAthree','CDF1', 'CDF2', 'CDF3']
#filter_out = ['id', 'min_ANNmuon', 'production', 'mass', 'signal','SPDhits','p0_track_Chi2Dof','CDF1', 'CDF2', 'CDF3','isolationb', 'isolationc','p0_pt', 'p1_pt', 'p2_pt', 'p0_p', 'p1_p', 'p2_p', 'p0_eta', 'p1_eta', 'p2_eta','DOCAone', 'DOCAtwo', 'DOCAthree']
features = list(f for f in train.columns if f not in filter_out)
print('Training XGBoost model...')
model_xgb = xgb.XGBClassifier()
params ={'nthread': 4,
         'objective': 'binary:logistic',
         'max_depth' : 8,
         'min_child_weight': 3,
         'learning_rate' : 0.1, 
         'n_estimators' : 300, 
         'subsample' : 0.9, 
         'colsample_bytree' : 0.5,
         'silent': 1}
model_xgb.fit(train[features],train.signal)
pred_test = model_xgb.predict_proba(test[features])[:,1]
result = pd.DataFrame({'id': test.id})
result['prediction'] = pred_test
result.to_csv('submission_nacho.csv', index=False, sep=',')