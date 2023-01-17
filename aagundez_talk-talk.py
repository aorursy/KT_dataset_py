import random
import datetime
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from subprocess import check_output

random.seed(2010)

print(check_output(['ls', '../input']).decode('utf8'))
def read(file_name):
    return pd.read_csv('../input/{}.csv'.format(file_name))

app_events = read('app_events')
app_labels = read('app_labels')
events = read('events')
gender_age_test = read('gender_age_test')
gender_age_train = read('gender_age_train')
label_categories = read('label_categories')
phone_brand_device_model = read('phone_brand_device_model')
sample_submission = read('sample_submission')

app_le = LabelEncoder()
app_le.fit(app_events['app_id'])

device_model_le = LabelEncoder()
device_model_le.fit(phone_brand_device_model['device_model'])
def app_activity(gender_age):
    active_events = (gender_age
                     .merge(events, how='left')
                     .merge(app_events[app_events['is_active'] == 1], how='left'))
        
    events_grouped = active_events.replace(np.nan, -1).groupby(['device_id', 'app_id']).size()

    data = list(events_grouped)
    device_id_rows, app_id_cols = zip(*events_grouped.index)
        
    device_le = LabelEncoder()
    device_le.fit(device_id_rows)
    
    device_idx_map = device_le.transform(device_id_rows)
    app_idx_map = app_le.transform(app_id_cols)
    
    X = csr_matrix(
        (data, (device_idx_map, app_idx_map)), 
        shape=(len(device_le.classes_), len(app_le.classes_)))
    
    m = (pd.DataFrame({ 'device_id': device_id_rows })
         .merge(phone_brand_device_model, how='left')
         .groupby('device_id')
         .first()['device_model'])

    X = hstack((X, device_model_le.transform(list(m)).reshape(len(m),1)), format='csr')
    
    return X, device_le

def create_submission(test, prediction):
    # Make Submission
    now = datetime.datetime.now()
    sub_file = 'submission_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    print('Writing submission: ', sub_file)
    f = open(sub_file, 'w')
    f.write('device_id,F23-,F24-26,F27-28,F29-32,F33-42,F43+,M22-,M23-26,M27-28,M29-31,M32-38,M39+\n')
    total = 0
    
    for i in range(len(test)):
        str1 = str(test[i])
        for j in range(12):
            str1 += ',' + str(prediction[i][j])
        str1 += '\n'
        total += 1
        f.write(str1)
    f.close()
X, device_le_train = app_activity(gender_age_train)
y = pd.DataFrame(device_le_train.classes_, columns=['device_id']).merge(gender_age_train)['group']
eta = 0.1
max_depth = 3
subsample = 0.7
colsample_bytree = 0.7

params = {
    "objective": "multi:softprob",
    "num_class": 12,
    "booster" : "gbtree",
    "eval_metric": "mlogloss",
    "eta": eta,
    "max_depth": max_depth,
    "subsample": subsample,
    "colsample_bytree": colsample_bytree,
    "silent": 1,
    "seed": 0,
}
num_boost_round = 500
early_stopping_rounds = 50
test_size = 0.3

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=test_size, random_state=0)

group_le = LabelEncoder()
group_le.fit(gender_age_train['group'])

dtrain = xgb.DMatrix(X_train, group_le.transform(y_train))
dvalid = xgb.DMatrix(X_valid, group_le.transform(y_valid))

watchlist = [(dtrain, 'train'), (dvalid, 'eval')]

gbm = xgb.train(params, dtrain, num_boost_round, evals=watchlist, early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
X_test, device_le_test = app_activity(gender_age_test)
test_prediction = gbm.predict(xgb.DMatrix(X_test), ntree_limit=gbm.best_iteration)
create_submission(device_le_test.classes_, test_prediction)
