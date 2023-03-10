#Python Modules
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import operator
# Loading data
df_train = pd.read_csv('../input/train_users_2.csv')
df_test = pd.read_csv('../input/test_users.csv')
labels = df_train['country_destination'].values
df_train = df_train.drop(['country_destination'], axis=1)
id_test = df_test['id']
piv_train = df_train.shape[0]
# Creating a DataFrame with train+test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
#Removing id and date_first_booking
df_all = df_all.drop(['id', 'date_first_booking'], axis=1)
#Filling nan
df_all = df_all.fillna(-1)
# date_account_created
dac = np.vstack(
    df_all.date_account_created.astype(str).apply(
        lambda x: list(map(int, x.split('-')))
        ).values
    )
df_all['dac_year'] = dac[:,0]
df_all['dac_month'] = dac[:,1]
df_all['dac_day'] = dac[:,2]
df_all = df_all.drop(['date_account_created'], axis=1)
# timestamp_first_active
tfa = np.vstack(
    df_all.timestamp_first_active.astype(str).apply(
        lambda x: list(map(int, [x[:4], x[4:6], x[6:8],
                                 x[8:10], x[10:12],
                                 x[12:14]]))
        ).values
    )
df_all['tfa_year'] = tfa[:,0]
df_all['tfa_month'] = tfa[:,1]
df_all['tfa_day'] = tfa[:,2]
df_all = df_all.drop(['timestamp_first_active'], axis=1)
# One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 
             'affiliate_channel', 'affiliate_provider', 
             'first_affiliate_tracked', 'signup_app', 
             'first_device_type', 'first_browser']
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)
# Splitting train and test
X = df_all.iloc[:piv_train,:]
le = LabelEncoder()
y = le.fit_transform(labels)   
X_test = df_all.iloc[piv_train:,:]
# Classifier
params = {'eta': 0.1,
          'max_depth': 8,
          'nround': 100,
          'subsample': 0.7,
          'colsample_bytree': 0.8,
          'seed': 1,
          'objective': 'multi:softprob',
          'eval_metric':'ndcg',
          'num_class': 12,
          'nthread':3}
num_boost_round = 10
dtrain = xgb.DMatrix(X, y)
clf1 = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round)
# Get feature scores and store in DataFrame
importance = clf1.get_fscore()
importance_df = pd.DataFrame(
    sorted(importance.items(), key=operator.itemgetter(1)), 
    columns=['feature','fscore']
    )
# Plot feature importance of top 20
importance_df.iloc[-20:,:].plot(x='feature',y='fscore',kind='barh')

# Only select features w/ a feature score (can also specify min fscore)
# Retrain model with reduced feature set
df_all = df_all[importance_df.feature.values]
X = df_all.iloc[:piv_train,:]
X_test = df_all.iloc[piv_train:,:]
dtrain = xgb.DMatrix(X, y)
clf2 = xgb.train(params=params, dtrain=dtrain, num_boost_round=num_boost_round)
y_pred = clf2.predict(xgb.DMatrix(X_test)).reshape(df_test.shape[0],12)
# Taking the 5 classes with highest probabilities
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * 5
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:5].tolist()

# Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('sub.csv',index=False)