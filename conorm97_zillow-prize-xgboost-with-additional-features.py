import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
#load training file
train = pd.read_csv("../input/train_2016_v2.csv", parse_dates=["transactiondate"])
print(train.head())
print('---------------------')
print(train.shape)
#load property features/description file
prop = pd.read_csv("../input/properties_2016.csv")
print(prop.head())
print('---------------------')
print(prop.shape)
#replace missing values with -1 
for x in prop.columns:
    prop[x] = prop[x].fillna(-1)
#encode non-numerical data
for c in prop[['hashottuborspa', 'propertycountylandusecode', 'propertyzoningdesc', 'fireplaceflag', 'taxdelinquencyflag']]:
    label = LabelEncoder()
    label.fit(list(prop[c].values))
    prop[c] = label.transform(list(prop[c].values))
#transaction month
#I found the importance of this feature in my kernel: https://www.kaggle.com/conorm97/zillow-prize-exploration
train['transaction_month'] = pd.DatetimeIndex(train['transactiondate']).month

#inclusion of this feature will allow three sets of predictions to be made for submission (October, November, December) 
#The following features were found to be imortant in the kernel: https://www.kaggle.com/nikunjm88/creating-additional-features?scriptVersionId=1379783 

#living area proportions 
prop['living_area_prop'] = prop['calculatedfinishedsquarefeet'] / prop['lotsizesquarefeet']

#tax value ratio
prop['value_ratio'] = prop['taxvaluedollarcnt'] / prop['taxamount']

#tax value proportions
prop['value_prop'] = prop['structuretaxvaluedollarcnt'] / prop['landtaxvaluedollarcnt']

#combination of longitude and latitude 
prop['location'] = prop['latitude'] + prop['longitude']
test = prop.drop(['parcelid'], axis=1)
#many more parcelids in properties file, merge with training file
train = pd.merge(train, prop, on='parcelid', how='left')
print(train.head())
print('---------------------')
print(train.shape)
#presence of outliers identified in my kernel: https://www.kaggle.com/conorm97/zillow-prize-exploration
log_errors = train['logerror']
train = train[train.logerror < np.percentile(log_errors, 99.5)]
train = train[train.logerror > np.percentile(log_errors, 0.5)]

print('upper limit: ', np.percentile(log_errors, 99.5))
print('lower limit: ', np.percentile(log_errors, 0.5))
x_train = train.drop(['parcelid', 'logerror','transactiondate'], axis=1)
y_train = train['logerror']
#for now I will use parameters and cv methods from this kernel: https://www.kaggle.com/danieleewww/xgboost-without-outliers-lb-0-06463?scriptVersionId=1452576
params = {
    'eta': 0.033,
    'max_depth': 6,
    'subsample': 0.80,
    'objective': 'reg:linear',
    'eval_metric': 'mae',
    'base_score': np.mean(y_train),
    'silent': 1
}
dtrain = xgb.DMatrix(x_train, y_train)
#cv_result = xgb.cv(params, 
                   #dtrain, 
                   #nfold=5,
                   #num_boost_round=500,
                   #early_stopping_rounds=5,
                   #verbose_eval=10, 
                   #show_stdv=False
                  #)
#num_boost_rounds = len(cv_result)
#print(num_boost_rounds) #output: 112
num_boost_rounds = 112
#train model
mdl = xgb.train(params, dtrain, num_boost_round=num_boost_rounds)
#use sample submission as a template to overwrite
sub = pd.read_csv('../input/sample_submission.csv')
#create test files for predictions involving different months
oct_test = test.copy()
#nov_test = test.copy()
#dec_test = test.copy()
#set transaction month columns for each prediction 
oct_test['transaction_month'] = np.repeat(10, oct_test.shape[0])
#nov_test['transaction_month'] = np.repeat(11, nov_test.shape[0])
#dec_test['transaction_month'] = np.repeat(12, dec_test.shape[0])
def reorder(df):
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    df = df.loc[:, cols]
    return df
oct_test = reorder(oct_test)
#nov_test = reorder(nov_test)
#dec_test = reorder(dec_test)
#convert test file format to that required by XGBoost algorithm
d_oct_test = xgb.DMatrix(oct_test)
#d_nov_test = xgb.DMatrix(nov_test)
#d_dec_test = xgb.DMatrix(dec_test)
print(oct_test.head())
print('---------------')
print(x_train.head())
oct_pred = mdl.predict(d_oct_test)
#nov_pred = mdl.predict(d_nov_test)
#dec_pred = mdl.predict(d_dec_test)
ss = pd.read_csv('../input/sample_submission.csv')
sub_oct_pred = []
sub_nov_pred = []
sub_dec_pred = []

for i,predict in enumerate(oct_pred):
    sub_oct_pred.append(str(round(predict,4)))
sub_oct_pred=np.array(sub_oct_pred)

#for i,predict in enumerate(nov_pred):
#    sub_nov_pred.append(str(round(predict,4)))
#sub_nov_pred=np.array(sub_nov_pred)

#for i,predict in enumerate(dec_pred):
#    sub_dec_pred.append(str(round(predict,4)))
#sub_dec_pred=np.array(sub_dec_pred)
#sub = pd.DataFrame({'ParcelId': prop['parcelid'].astype(np.int32), '201610':sub_oct_pred, '201611': sub_nov_pred, '201612':sub_dec_pred, '201710':sub_oct_pred, '201711':sub_nov_pred, '201712':sub_dec_pred})
sub = pd.DataFrame({'ParcelId': prop['parcelid'].astype(np.int32), '201610':sub_oct_pred, '201710':sub_oct_pred})
cols = sub.columns.tolist()
cols = cols[-1:] + cols[:-1]
sub = sub[cols]
sub.to_csv('submission.gz', index=False, compression = 'gzip')