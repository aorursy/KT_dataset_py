import pandas as pd



from sklearn.preprocessing import OneHotEncoder

from lightgbm import LGBMClassifier



from sklearn.metrics import roc_auc_score
data_train = pd.read_csv('../input/ozonmasters-ml2-2020-c1/1_data/train_data.csv')

target_train = pd.read_csv('../input/ozonmasters-ml2-2020-c1/1_data/train_target.csv')

data_train['target'] = target_train



data_test = pd.read_csv('../input/ozonmasters-ml2-2020-c1/1_data/test_data.csv')
def add_features(data_train, data_test):

    for data in [data_train, data_test]:

        data['is_dist'] = (data['dist'] == -1).astype(float)

        

        # add tmp features

        data.loc[:,'time_as_str'] = data['due'].apply(lambda x: x[x.find(' ') + 1:-4])

        data.loc[:, 'date_as_str'] = data['due'].apply(lambda x: x[:x.find(' ')])

        

        # add date features

        data_date_structure = pd.to_datetime(data['date_as_str'])

        data.loc[:, 'week_day'] = data_date_structure.dt.dayofweek

        

        # add time features

        data.loc[:, 'time_in_seconds'] = pd.to_timedelta(data['time_as_str']).dt.total_seconds()

        data.loc[:, 'time_in_minutes'] = data['time_in_seconds'] // 60

        data.loc[:, 'time_in_hours'] = data['time_in_minutes'] // 60

        

        # fillna for cat features

        data.fillna({

            'f_class': 'unknown_f',

            's_class': 'unknown_s',

            't_class': 'unknown_t',

        }, inplace=True)

        

        # drop tmp features

        data.drop(['due', 'time_as_str', 'date_as_str'], axis=1, inplace=True)

    

    for column in ['f_class', 's_class', 't_class', 'week_day']:

        data_train[column] = data_train[column].astype('category')

        data_test[column] = data_test[column].astype('category')

        data_test[column] = (

            data_test[column]

            .cat

            .set_categories(data_train[column].cat.categories)

        )

    

    data_train = pd.get_dummies(data_train)

    data_test = pd.get_dummies(data_test)

    

    return data_train, data_test
data_train, data_test = add_features(data_train, data_test)
data_train.head()
data_test.head()
feature_columns = data_train.columns.tolist()

feature_columns.pop(feature_columns.index('target'))



target_column = ['target']
data_train[feature_columns].dtypes
clf = LGBMClassifier(n_estimators=1577, learning_rate=0.05, num_leaves=63, max_depth=5)

clf.fit(

    data_train[feature_columns], data_train[target_column].values.ravel(),

    eval_metric='auc',

    verbose=True,

)



predictions = clf.predict_proba(data_test[feature_columns])
prediction_df = pd.DataFrame(predictions[:, 1], columns=['target'])

prediction_df = prediction_df.reset_index()
prediction_df.to_csv('first_baseline.csv', index=0)