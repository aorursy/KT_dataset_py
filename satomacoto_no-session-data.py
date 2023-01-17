%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
!ls ../input
#Loading data
df_train = pd.read_csv('../input/train_users_2.csv')
df_test = pd.read_csv('../input/test_users.csv')
df_sessions = pd.read_csv('../input/sessions.csv')
df_coutries = pd.read_csv('../input/countries.csv')
df_sessions.user_id.values
f = 'signup_method'
pd.get_dummies(df_train[f], prefix=f)
df_train.columns
tmp = df_train.pivot_table(values='id', index='language', columns='country_destination', aggfunc='count')
tmp.fillna(0).div(tmp.sum(1), axis=0).plot(kind='bar', stacked=True, legend=None)