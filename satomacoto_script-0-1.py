%matplotlib inline
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier
import matplotlib.pyplot as plt
#Loading data
df_train_raw = pd.read_csv('../input/train_users_2.csv')
df_test = pd.read_csv('../input/test_users.csv')
labels = df_train_raw['country_destination'].values
df_train = df_train_raw.drop(['country_destination'], axis=1)
id_test = df_test['id']
piv_train = df_train.shape[0]
#Creating a DataFrame with train+test data
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
#Removing id and date_first_booking
df_all = df_all.drop(['id', 'date_first_booking'], axis=1)
#Filling nan
df_all = df_all.fillna(-1)

#####Feature engineering#######
#date_account_created
df_all.date_account_created = pd.to_datetime(df_all.date_account_created)
df_all['dac_year'] = df_all.date_account_created.apply(lambda x: x.year)
df_all['dac_month'] = df_all.date_account_created.apply(lambda x: x.month)
df_all['dac_day'] = df_all.date_account_created.apply(lambda x: x.day)
df_all['dac_weekday'] = df_all.date_account_created.apply(lambda x: x.weekday())
df_all['dac_week'] = df_all.date_account_created.apply(lambda x: x.week)
df_all['dac_log_elapsed'] = np.log((datetime.date(2016, 1, 1) - df_all.date_account_created).astype('timedelta64[D]'))
df_all = df_all.drop(['date_account_created'], axis=1)

#timestamp_first_active
df_all.timestamp_first_active = pd.to_datetime(df_all.timestamp_first_active, format='%Y%m%d%H%M%S')
df_all['tfa_year'] = df_all.timestamp_first_active.apply(lambda x: x.year)
df_all['tfa_month'] = df_all.timestamp_first_active.apply(lambda x: x.month)
df_all['tfa_day'] = df_all.timestamp_first_active.apply(lambda x: x.day)
df_all['tfa_weekday'] = df_all.timestamp_first_active.apply(lambda x: x.weekday())
df_all['tfa_week'] = df_all.timestamp_first_active.apply(lambda x: x.week)
df_all['tfa_log_elapsed'] = np.log((datetime.date(2016, 1, 1) - df_all.timestamp_first_active).astype('timedelta64[D]'))
df_all = df_all.drop(['timestamp_first_active'], axis=1)

#Age
av = df_all.age.values
df_all['age'] = np.where(np.logical_or(av<14, av>90), -1, av)
df_all['age_year'] = np.where(av > 1900, -1, av)

#One-hot-encoding features
ohe_feats = ['gender', 'signup_method', 'signup_flow', 'language', 'affiliate_channel',
             'affiliate_provider', 'first_affiliate_tracked', 'signup_app', 'first_device_type',
             'first_browser']
for f in ohe_feats:
    df_all_dummy = pd.get_dummies(df_all[f], prefix=f)
    df_all = df_all.drop([f], axis=1)
    df_all = pd.concat((df_all, df_all_dummy), axis=1)    
#Splitting train and test
vals = df_all.values
X_train = vals[:piv_train]
le = LabelEncoder()
y_train = le.fit_transform(labels)
X_test = vals[piv_train:]
# from sklearn.model_selection import cross_val_score
# from sklearn.ensemble import GradientBoostingClassifier

# np.random.seed(42)
# samples = np.random.choice(piv_train, 10000)
# X_train = vals[samples]
# y_train = le.fit_transform(labels)[samples]
# model = GradientBoostingClassifier()
# cross_val_score(model, X_train, y_train, cv=5, n_jobs=-1)
np.random.seed(42)
samples = np.random.choice(piv_train, 50000)
X_train = vals[samples]
y_train = le.fit_transform(labels)[samples]
#Classifier
xgb = XGBClassifier(max_depth=6, learning_rate=0.25, n_estimators=42,
                    objective='multi:softprob', subsample=0.6, colsample_bytree=0.6, seed=42)
xgb.fit(X_train, y_train)
y_pred = xgb.predict_proba(X_test)  

#Taking the n classes with highest probabilities
n_class = 5
ids = []  #list of ids
cts = []  #list of countries
for i in range(len(id_test)):
    idx = id_test[i]
    ids += [idx] * n_class
    cts += le.inverse_transform(np.argsort(y_pred[i])[::-1])[:n_class].tolist()

#Generate submission
sub = pd.DataFrame(np.column_stack((ids, cts)), columns=['id', 'country'])
sub.to_csv('sub.csv',index=False)