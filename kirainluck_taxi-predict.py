data_train = pd.read_csv('train_data.csv')

target_train = pd.read_csv('train_target.csv')

data_train['target'] = target_train

data_test = pd.read_csv('test_data.csv')
data_train['min_dist'] = -1

data_test['min_dist'] = -1
sorted_train = data_train.sort_values('due')

sorted_test = data_test.sort_values('due')
sorted_train['due'] = pd.to_datetime(sorted_train['due'], format='%Y-%m-%d %H:%M:%S.%f')

sorted_test['due'] = pd.to_datetime(sorted_test['due'], format='%Y-%m-%d %H:%M:%S.%f')
for i in range(data_train.shape[0]):

    r2 = sorted_train.iloc[[i]];



    rows = pd.DataFrame(columns=data_train.columns)

    for j in range(i, data_train.shape[0]):

        cur_row = sorted_train.iloc[[j]]

        if (((cur_row['due'].iloc[0]-r2['due'].iloc[0]).total_seconds() / 60) < 30):

            rows = rows.append(cur_row)

        else:

            break



    for j in range(i, -1, -1):

        cur_row = sorted_train.iloc[[j]]

        if (((r2['due'].iloc[0]-cur_row['due'].iloc[0]).total_seconds() / 60) < 30):

            rows = rows.append(cur_row)

        else:

            break

        

    sorted_train.iat[i,8] = (rows[rows['dist'] >0 ])['dist'].min()    
for i in range(data_test.shape[0]):

    r2 = sorted_test.iloc[[i]];



    rows = pd.DataFrame(columns=data_test.columns)

    for j in range(i, data_test.shape[0]):

        cur_row = sorted_test.iloc[[j]]

        if (((cur_row['due'].iloc[0]-r2['due'].iloc[0]).total_seconds() / 60) < 30):

            rows = rows.append(cur_row)

        else:

            break



    for j in range(i, -1, -1):

        cur_row = sorted_test.iloc[[j]]

        if (((r2['due'].iloc[0]-cur_row['due'].iloc[0]).total_seconds() / 60) < 30):

            rows = rows.append(cur_row)

        else:

            break

        

    sorted_test.iat[i,8] = (rows[rows['dist'] >0 ])['dist'].min() 
for i in range(data_train.shape[0]):

    data_train.loc[i,'min_dist'] = sorted_train.loc[i,'min_dist']
for i in range(data_test.shape[0]):

    data_test.loc[i,'min_dist'] = sorted_test.loc[i,'min_dist']
data_train.head()
data_test.head()
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

data_train['lonlatsum'] = data_train.apply(lambda row: (row.lon+row.lat), axis=1);

data_test['lonlatsum'] = data_test.apply(lambda row: (row.lon+row.lat), axis=1);

data_train['lonlatmul'] = data_train.apply(lambda row: (row.lon*row.lat), axis=1);

data_test['lonlatmul'] = data_test.apply(lambda row: (row.lon*row.lat), axis=1);

data_train['lonlatsqr'] = data_train.apply(lambda row: math.sqrt(row.lon*row.lon+row.lat*row.lat), axis=1);

data_test['lonlatsqr'] = data_test.apply(lambda row: math.sqrt(row.lon*row.lon+row.lat*row.lat), axis=1);
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
predictions = clf.predict_proba(data_test[feature_columns])[:, 1]
submission = pd.DataFrame()

submission['index'] = np.arange(predictions.shape[0])

submission['target'] = predictions

submission.to_csv('submission.csv', index=False)

submission.head()
X_train, X_test, y_train, y_test = train_test_split(data_train[feature_columns], data_train[target_column], test_size=0.33, random_state=42)
clf.fit(X_train[feature_columns], y_train[target_column])
predictions = clf.predict_proba(X_test[feature_columns])

auc_score = roc_auc_score(y_test, predictions[:, 1])

auc_score