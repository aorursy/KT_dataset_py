import pandas as pd

from sklearn.preprocessing import OneHotEncoder

import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
df_train = pd.read_csv("../input/student-shopee-code-league-marketing-analytics/train.csv")

df_test = pd.read_csv("../input/student-shopee-code-league-marketing-analytics/test.csv")

df_users = pd.read_csv("../input/student-shopee-code-league-marketing-analytics/users.csv")
df_train = df_train.fillna(-1)

df_users = df_users.fillna(-1)

df_test = df_test.fillna(-1)
user_dict = {}

for row in df_users.itertuples():

    user_dict[row.user_id] = (row.attr_1,row.attr_2,row.attr_3,row.age,row.domain)
def get_user_feature(user_id,i):

    if user_id in user_dict:

        return user_dict[user_id][i]

    else:

        return -2
def fill_ints(data):

    if isinstance(data,int):

        return data

    if data.isnumeric():

        return data

    else:

        return -1
def time_to_categorical_series(df,type="hour"):

    if type == "hour":

        return df['date_time'].dt.hour.astype('category')

    elif type == "dayofweek":

        return df['date_time'].dt.dayofweek.astype('category')

    elif type == "month":

        return df['date_time'].dt.month.astype('category')

    else:

        return None

    

def time_to_categorical(df):

    hour_series = time_to_categorical_series(df,type='hour')

    dayofweek_series = time_to_categorical_series(df,type='dayofweek')

    month_series = time_to_categorical_series(df,type='month')



    df['hour'] = hour_series

    df['dayofweek'] = dayofweek_series

    df['month'] = month_series
cat_features = ['country_code','hour','dayofweek','month','domain']

numerical_features = [ 'subject_line_length',

       'last_open_day', 'last_login_day', 'last_checkout_day',

       'open_count_last_10_days', 'open_count_last_30_days',

       'open_count_last_60_days', 'login_count_last_10_days',

       'login_count_last_30_days', 'login_count_last_60_days',

       'checkout_count_last_10_days', 'checkout_count_last_30_days',

       'checkout_count_last_60_days','attr1', 'attr2',

       'attr3', 'age']
def make_df_features(df,train=None,encoder=None):

    df['attr1'] = df['user_id'].apply(lambda x: get_user_feature(x,0))

    df['attr2'] = df['user_id'].apply(lambda x: get_user_feature(x,1))

    df['attr3'] = df['user_id'].apply(lambda x: get_user_feature(x,2))

    df['age'] = df['user_id'].apply(lambda x: get_user_feature(x,3))

    df['domain'] = df['user_id'].apply(lambda x: get_user_feature(x,4))

    df['date_time'] = pd.to_datetime(df['grass_date'])

    df['last_open_day'] = df['last_open_day'].apply(fill_ints)

    df['last_login_day'] = df['last_login_day'].apply(fill_ints)

    df['last_checkout_day'] = df['last_checkout_day'].apply(fill_ints)

    time_to_categorical(df)

    cat = df.loc[:,cat_features].values

    if train:

        encoder = OneHotEncoder(handle_unknown='ignore',sparse=False)

        cat = encoder.fit_transform(cat).astype(np.float64)

    else:

        cat = encoder.transform(cat).astype(np.float64)

    val = df.loc[:,numerical_features].values.astype(np.float64)

    return np.concatenate([cat,val],axis=1),encoder
train_features,encoder = make_df_features(df_train,True)

train_labels = df_train['open_flag'].values



test_features,_ = make_df_features(df_test,False,encoder=encoder)
clf = GradientBoostingClassifier()

clf.fit(train_features,train_labels)

predictions = clf.predict(test_features)
df_test = df_test.drop([col for col in df_test.columns if col!='row_id'],axis=1)

df_test['open_flag'] = predictions

df_test.to_csv('sub.csv',index=False)