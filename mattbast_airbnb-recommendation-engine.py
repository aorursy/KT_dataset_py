import numpy as np 

import pandas as pd 

import matplotlib.pyplot as plt

import tensorflow as tf

import ml_metrics as metrics

from sklearn.preprocessing import MinMaxScaler
train_users = pd.read_csv(

    '/kaggle/input/airbnb-recruiting-new-user-bookings/train_users_2.csv.zip',

    parse_dates = ['timestamp_first_active', 'date_account_created', 'date_first_booking']

)
fig, ax = plt.subplots(3, 3, figsize=(20, 12))



# row one

ax[0,0].set_title('Age')

train_users['age'].hist(ax=ax[0,0])



ax[0,1].set_title('Language')

train_users['language'].value_counts().plot.bar(ax=ax[0,1])



ax[0,2].set_title('Gender')

train_users['gender'].value_counts().plot.bar(ax=ax[0,2])



# row two

ax[1,0].set_title('Signup Method')

train_users['signup_method'].value_counts().plot.bar(ax=ax[1,0])



ax[1,1].set_title('Signup Flow')

train_users['signup_flow'].value_counts().plot.bar(ax=ax[1,1])



ax[1,2].set_title('Affliate Channel')

train_users['affiliate_channel'].value_counts().plot.bar(ax=ax[1,2])



# row three

ax[2,0].set_title('Affliate Provider')

train_users['affiliate_provider'].value_counts().plot.bar(ax=ax[2,0])



ax[2,1].set_title('Signup App')

train_users['signup_app'].value_counts().plot.bar(ax=ax[2,1])



ax[2,2].set_title('First Browser')

train_users['first_browser'].value_counts().nlargest(8).plot.bar(ax=ax[2,2])
train_users['age'] = train_users['age'].fillna(train_users['age'].mean())

train_users.loc[train_users['age'] > 1920, 'age'] = train_users['age'].mean()
fig, ax = plt.subplots(3, 6, figsize=(20, 14))



# row one

ax[0,0].set_title('Age')

train_users.plot.scatter(x='country_destination', y='age', ax=ax[0,0])



ax[0,1].set_title('Gender - UNKNOWN')

train_users[train_users['gender'] == '-unknown-']['country_destination'].value_counts().plot.bar(ax=ax[0,1], color='firebrick')



ax[0,2].set_title('Gender - FEMALE')

train_users[train_users['gender'] == 'FEMALE']['country_destination'].value_counts().plot.bar(ax=ax[0,2], color='firebrick')



ax[0,3].set_title('Gender - MALE')

train_users[train_users['gender'] == 'MALE']['country_destination'].value_counts().plot.bar(ax=ax[0,3], color='firebrick')



ax[0,4].set_title('Gender - OTHER')

train_users[train_users['gender'] == 'OTHER']['country_destination'].value_counts().plot.bar(ax=ax[0,4], color='firebrick')



ax[0,5].set_title('Signup Flow')

train_users.plot.scatter(x='country_destination', y='signup_flow', ax=ax[0,5], color='forestgreen')



# row two

ax[1,0].set_title('Lang - en')

train_users[train_users['language'] == 'en']['country_destination'].value_counts().plot.bar(ax=ax[1,0], color='gold')



ax[1,1].set_title('Lang - fr')

train_users[train_users['language'] == 'fr']['country_destination'].value_counts().plot.bar(ax=ax[1,1], color='gold')



ax[1,2].set_title('Lang - zh')

train_users[train_users['language'] == 'zh']['country_destination'].value_counts().plot.bar(ax=ax[1,2], color='gold')



ax[1,3].set_title('Lang - es')

train_users[train_users['language'] == 'es']['country_destination'].value_counts().plot.bar(ax=ax[1,3], color='gold')



ax[1,4].set_title('Lang - it')

train_users[train_users['language'] == 'it']['country_destination'].value_counts().plot.bar(ax=ax[1,4], color='gold')



ax[1,5].set_title('Lang - de')

train_users[train_users['language'] == 'de']['country_destination'].value_counts().plot.bar(ax=ax[1,5], color='gold')



# row three

ax[2,0].set_title('Signup method - basic')

train_users[train_users['signup_method'] == 'basic']['country_destination'].value_counts().plot.bar(ax=ax[2,0], color='orchid')



ax[2,1].set_title('Signup method - facebook')

train_users[train_users['signup_method'] == 'facebook']['country_destination'].value_counts().plot.bar(ax=ax[2,1], color='orchid')



ax[2,2].set_title('Signup App - web')

train_users[train_users['signup_app'] == 'Web']['country_destination'].value_counts().plot.bar(ax=ax[2,2], color='pink')



ax[2,3].set_title('Signup App - ios')

train_users[train_users['signup_app'] == 'iOS']['country_destination'].value_counts().plot.bar(ax=ax[2,3], color='pink')



ax[2,4].set_title('Signup App - android')

train_users[train_users['signup_app'] == 'Android']['country_destination'].value_counts().plot.bar(ax=ax[2,4], color='pink')



ax[2,5].set_title('Signup App - moweb')

train_users[train_users['signup_app'] == 'Moweb']['country_destination'].value_counts().plot.bar(ax=ax[2,5], color='pink')
fig, ax = plt.subplots(3, 6, figsize=(20, 14))



# row one

ax[0,0].set_title('Aff Channel - direct')

train_users[train_users['affiliate_channel'] == 'direct']['country_destination'].value_counts().plot.bar(ax=ax[0,0])



ax[0,1].set_title('Aff Channel - sem-brand')

train_users[train_users['affiliate_channel'] == 'sem-brand']['country_destination'].value_counts().plot.bar(ax=ax[0,1])



ax[0,2].set_title('Aff Channel - sem-non-brand')

train_users[train_users['affiliate_channel'] == 'sem-non-brand']['country_destination'].value_counts().plot.bar(ax=ax[0,2])



ax[0,3].set_title('Aff Channel - other')

train_users[train_users['affiliate_channel'] == 'other']['country_destination'].value_counts().plot.bar(ax=ax[0,3])



ax[0,4].set_title('Aff Channel - api')

train_users[train_users['affiliate_channel'] == 'api']['country_destination'].value_counts().plot.bar(ax=ax[0,4])



ax[0,5].set_title('Aff Channel - seo')

train_users[train_users['affiliate_channel'] == 'seo']['country_destination'].value_counts().plot.bar(ax=ax[0,5])



# row two

ax[1,0].set_title('Aff Channel - content')

train_users[train_users['affiliate_channel'] == 'content']['country_destination'].value_counts().plot.bar(ax=ax[1,0])



ax[1,1].set_title('Aff Channel - remarketing')

train_users[train_users['affiliate_channel'] == 'remarketing']['country_destination'].value_counts().plot.bar(ax=ax[1,1])



ax[1,2].set_title('Browser - Chrome')

train_users[train_users['first_browser'] == 'Chrome']['country_destination'].value_counts().plot.bar(ax=ax[1,2], color='firebrick')



ax[1,3].set_title('Browser - Safari')

train_users[train_users['first_browser'] == 'Safari']['country_destination'].value_counts().plot.bar(ax=ax[1,3], color='firebrick')



ax[1,4].set_title('Browser - Firefox')

train_users[train_users['first_browser'] == 'Firefox']['country_destination'].value_counts().plot.bar(ax=ax[1,4], color='firebrick')



ax[1,5].set_title('Browser - IE')

train_users[train_users['first_browser'] == 'IE']['country_destination'].value_counts().plot.bar(ax=ax[1,5], color='firebrick')



# row three

ax[2,0].set_title('Aff Provider - direct')

train_users[train_users['affiliate_provider'] == 'direct']['country_destination'].value_counts().plot.bar(ax=ax[2,0], color='forestgreen')



ax[2,1].set_title('Aff Provider - google')

train_users[train_users['affiliate_provider'] == 'google']['country_destination'].value_counts().plot.bar(ax=ax[2,1], color='forestgreen')



ax[2,2].set_title('Aff Provider - other')

train_users[train_users['affiliate_provider'] == 'other']['country_destination'].value_counts().plot.bar(ax=ax[2,2], color='forestgreen')



ax[2,3].set_title('Aff Provider - craigslist')

train_users[train_users['affiliate_provider'] == 'craigslist']['country_destination'].value_counts().plot.bar(ax=ax[2,3], color='forestgreen')



ax[2,4].set_title('Aff Provider - facebook')

train_users[train_users['affiliate_provider'] == 'facebook']['country_destination'].value_counts().plot.bar(ax=ax[2,4], color='forestgreen')



ax[2,5].set_title('Aff Provider - bing')

train_users[train_users['affiliate_provider'] == 'bing']['country_destination'].value_counts().plot.bar(ax=ax[2,5], color='forestgreen')
train_users['dac_month'] = train_users['date_account_created'].dt.month

train_users['tfa_month'] = train_users['timestamp_first_active'].dt.month
fig, ax = plt.subplots(4, 3, figsize=(20, 14))



# row one

ax[0,0].set_title('Month Account Created - Jan')

train_users[train_users['dac_month'] == 1]['country_destination'].value_counts().plot.bar(ax=ax[0,0])



ax[0,1].set_title('Year Account Created - Feb')

train_users[train_users['dac_month'] == 2]['country_destination'].value_counts().plot.bar(ax=ax[0,1])



ax[0,2].set_title('Year Account Created - Mar')

train_users[train_users['dac_month'] == 3]['country_destination'].value_counts().plot.bar(ax=ax[0,2])



# row two

ax[1,0].set_title('Month Account Created - Apr')

train_users[train_users['dac_month'] == 4]['country_destination'].value_counts().plot.bar(ax=ax[1,0])



ax[1,1].set_title('Year Account Created - May')

train_users[train_users['dac_month'] == 5]['country_destination'].value_counts().plot.bar(ax=ax[1,1])



ax[1,2].set_title('Year Account Created - Jun')

train_users[train_users['dac_month'] == 6]['country_destination'].value_counts().plot.bar(ax=ax[1,2])



# row three

ax[2,0].set_title('Month Account Created - Jul')

train_users[train_users['dac_month'] == 7]['country_destination'].value_counts().plot.bar(ax=ax[2,0])



ax[2,1].set_title('Year Account Created - Aug')

train_users[train_users['dac_month'] == 8]['country_destination'].value_counts().plot.bar(ax=ax[2,1])



ax[2,2].set_title('Year Account Created - Sep')

train_users[train_users['dac_month'] == 9]['country_destination'].value_counts().plot.bar(ax=ax[2,2])



# row four

ax[3,0].set_title('Month Account Created - Oct')

train_users[train_users['dac_month'] == 10]['country_destination'].value_counts().plot.bar(ax=ax[3,0])



ax[3,1].set_title('Year Account Created - Nov')

train_users[train_users['dac_month'] == 11]['country_destination'].value_counts().plot.bar(ax=ax[3,1])



ax[3,2].set_title('Year Account Created - Dec')

train_users[train_users['dac_month'] == 12]['country_destination'].value_counts().plot.bar(ax=ax[3,2])
def get_batches():

    return pd.read_csv(

        '/kaggle/input/airbnb-recruiting-new-user-bookings/sessions.csv.zip',

        usecols=['user_id', 'action_type', 'action_detail', 'secs_elapsed'],

        chunksize=10000

    )
def load_and_group(group_column, agg_method):

    groups_list = []



    for batch in get_batches():

        groups_list.append(

            batch.groupby(['user_id', group_column])['secs_elapsed'].agg([agg_method])

        )

        

    groups = pd.concat(groups_list)

    groups = groups.groupby(['user_id', group_column]).sum()

    

    return groups 
action_counts = load_and_group(group_column='action_type', agg_method='count')

action_counts.head()
def pivot_log_groups(log_group, group_column, agg_method):

    log_group = log_group.reset_index().pivot(index='user_id', columns=group_column, values=[agg_method]).fillna(0)

    

    log_group.columns = log_group.columns.get_level_values(1)

    log_group = log_group.rename(columns={'-unknown-': 'unknown'})

    log_group = log_group.add_suffix('_' + group_column + '_' + agg_method)

    log_group_cols = log_group.columns.values

    log_group = log_group.reset_index()

    log_group = log_group.rename(columns={'user_id': 'id'})

    

    return log_group, log_group_cols
action_counts, action_counts_cols = pivot_log_groups(action_counts, group_column='action_type', agg_method='count')

action_counts.head()
train_users = train_users.merge(action_counts, how='left', on=['id']).fillna(0)

train_users.head()
val_users = train_users[train_users['timestamp_first_active'] >= '2014-06-01']

train_users = train_users[train_users['timestamp_first_active'] < '2014-06-01']



print('Count of training users: ' + str(len(train_users)))

print('Count of validation users: ' + str(len(val_users)))
train_labels = train_users.pop('country_destination')

val_labels = val_users.pop('country_destination')
train_labels = pd.Categorical(train_labels)

val_labels = pd.Categorical(val_labels)



categories = {i:category for i, category in enumerate(train_labels.categories.values)}
train_labels = train_labels.codes

val_labels = val_labels.codes
train_users = train_users.drop(columns=['timestamp_first_active', 'date_account_created', 'date_first_booking', 'first_affiliate_tracked'])

val_users = val_users.drop(columns=['timestamp_first_active', 'date_account_created', 'date_first_booking', 'first_affiliate_tracked'])
tf_train_data = tf.data.Dataset.from_tensor_slices(

    (dict(train_users), train_labels)

)



tf_val_data = tf.data.Dataset.from_tensor_slices(

    (dict(val_users), val_labels)

)
tf_train_data = tf_train_data.shuffle(100).batch(32)

tf_val_data = tf_val_data.batch(len(val_users))
language_vocab = tf.feature_column.categorical_column_with_vocabulary_list(

    'language', 

    train_users['language'].unique()

)



gender_vocab = tf.feature_column.categorical_column_with_vocabulary_list(

    'gender', 

    train_users['gender'].unique()

)



channel_vocab = tf.feature_column.categorical_column_with_vocabulary_list(

    'affiliate_channel', 

    train_users['affiliate_channel'].unique()

)



dac_month_vocab = tf.feature_column.categorical_column_with_vocabulary_list(

    'dac_month', 

    train_users['dac_month'].unique()

)



tfa_month_vocab = tf.feature_column.categorical_column_with_vocabulary_list(

    'tfa_month', 

    train_users['tfa_month'].unique()

)
features = []



for col_name in action_counts_cols:

        features.append(tf.feature_column.numeric_column(col_name))
features.extend([

    tf.feature_column.numeric_column('age'),

    tf.feature_column.indicator_column(language_vocab),

    tf.feature_column.indicator_column(gender_vocab),

    tf.feature_column.indicator_column(channel_vocab),

    tf.feature_column.indicator_column(dac_month_vocab),

    tf.feature_column.indicator_column(tfa_month_vocab),

])



feature_layer = tf.keras.layers.DenseFeatures(features)
model = tf.keras.Sequential([

    feature_layer,

    tf.keras.layers.Dense(64, activation='relu'),

    tf.keras.layers.Dense(12, activation='softmax'),

])
optimiser = tf.keras.optimizers.Ftrl(learning_rate=0.001)



model.compile(

    optimizer=optimiser, 

    loss='sparse_categorical_crossentropy', 

    metrics=['accuracy'])
callbacks = [

    tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=2, verbose=1),

    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose=1),

]
train_log = model.fit(

    tf_train_data,

    validation_data=tf_val_data,

    epochs=100,

    callbacks=callbacks

)
probabilities = model.predict(tf_val_data)
predictions = np.argpartition(-probabilities, 4, axis=1)[:,0:5:]
formatted_val_labels = np.array([[label] for label in val_labels])
metrics.mapk(formatted_val_labels.astype(str), predictions.astype(str), 5)
test_users = pd.read_csv(

    '/kaggle/input/airbnb-recruiting-new-user-bookings/test_users.csv.zip',

    parse_dates = ['timestamp_first_active', 'date_account_created']

)
test_users['age'] = test_users['age'].fillna(train_users['age'].mean())

test_users.loc[test_users['age'] > 100, 'age'] = train_users['age'].mean()



test_users['dac_month'] = test_users['date_account_created'].dt.month

test_users['tfa_month'] = test_users['timestamp_first_active'].dt.month



test_users = test_users.drop(columns=['timestamp_first_active', 'date_account_created', 'date_first_booking', 'first_affiliate_tracked'])



test_users = test_users.merge(action_counts, how='left', on=['id']).fillna(0)
tf_test_data = tf.data.Dataset.from_tensor_slices(

    (dict(test_users))

)



tf_test_data = tf_test_data.batch(len(test_users))
probabilities = model.predict(tf_test_data)

predictions = np.argpartition(-probabilities, 4, axis=1)[:,0:5:]
submission = pd.DataFrame(data=predictions, index=test_users['id'])



submission = pd.melt(

    submission.reset_index(), 

    id_vars=['id'], 

    value_vars=[0, 1, 2, 3, 4],

    value_name='country'

)
submission = submission.sort_values(by=['id', 'variable'])
submission = submission.set_index('id')

submission = submission.drop(columns=['variable'])
submission['country'] = submission['country'].replace(categories)
submission.to_csv('submisison.csv')