# This value stores the path where all the input data is stored. 

# This is in case you run the notebook on your local computer.

INPUT_DATA_PATH = '../input/'
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# SKLEARN

from sklearn.model_selection import KFold

from sklearn.metrics import log_loss



df_test = pd.read_csv(INPUT_DATA_PATH + 'test.csv',dtype={'msno' : 'category',

                                                'source_system_tab' : 'category',

                                                'source_screen_name' : 'category',

                                                'source_type' : 'category',

                                                'song_id' : 'category'})



df_train = pd.read_csv(INPUT_DATA_PATH + 'train.csv',dtype={'msno' : 'category',

                                                 'source_system_tab' : 'category',

                                                  'source_screen_name' : 'category',

                                                  'source_type' : 'category',

                                                  'target' : np.uint8,

                                                  'song_id' : 'category'})



df_members = pd.read_csv(INPUT_DATA_PATH + 'members.csv',dtype={'city' : 'category',

                                                      'bd' : np.uint8,

                                                      'gender' : 'category',

                                                      'registered_via' : 'category'},

                                                      parse_dates=['registration_init_time','expiration_date'])

# Convert date to number of days

df_members['membership_days'] = (df_members['expiration_date'] - df_members['registration_init_time']).dt.days.astype(int)



# Remove both date fieldsa since we already have the number of days between them

df_members = df_members.drop(['registration_init_time','expiration_date'], axis=1)
# Merge the members dataframe into the test dataframe

df_test = pd.merge(left = df_test,right = df_members,how='left',on='msno')

df_test.msno = df_test.msno.astype('category')



# Merge the member dataframe into the train dataframe

df_train = pd.merge(left = df_train,right = df_members,how='left',on='msno')

df_train.msno = df_train.msno.astype('category')



# Release memory

del df_members
# Load the songs dataframe

df_songs = pd.read_csv(INPUT_DATA_PATH + 'songs.csv',dtype={'genre_ids': 'category',

                                                  'language' : 'category',

                                                  'artist_name' : 'category',

                                                  'composer' : 'category',

                                                  'lyricist' : 'category',

                                                  'song_id' : 'category'})



# Merge the Test Dataframe with the SONGS dataframe

df_test = pd.merge(left = df_test,right = df_songs,how = 'left',on='song_id')

df_test.song_length.fillna(200000,inplace=True)

df_test.song_length = df_test.song_length.astype(np.uint32)

df_test.song_id = df_test.song_id.astype('category')



# Merge the Train dataframe with the SONGS dataframe

df_train = pd.merge(left = df_train,right = df_songs,how = 'left',on='song_id')

df_train.song_length.fillna(200000,inplace=True)

df_train.song_length = df_train.song_length.astype(np.uint32)

df_train.song_id = df_train.song_id.astype('category')



# Release memory

del df_songs
import lightgbm as lgb



# Create a Cross Validation with 3 splits

kf = KFold(n_splits=3)



# This array will store the predictions made.

predictions = np.zeros(shape=[len(df_test)])



# For each KFold

for train_indices ,validate_indices in kf.split(df_train) : 

    train_data = lgb.Dataset(df_train.drop(['target'],axis=1).loc[train_indices,:],label=df_train.loc[train_indices,'target'])

    val_data = lgb.Dataset(df_train.drop(['target'],axis=1).loc[validate_indices,:],label=df_train.loc[validate_indices,'target'])

    

    # Create the parameters for LGBM

    params = {

        'objective': 'binary',

        'metric': 'binary_logloss',

        'boosting': 'gbdt',

        'learning_rate': 0.1 ,

        'verbose': 0,

        'num_leaves': 108,

        'bagging_fraction': 0.95,

        'bagging_freq': 1,

        'bagging_seed': 1,

        'feature_fraction': 0.9,

        'feature_fraction_seed': 1,

        'max_bin': 128,

        'max_depth': 10,

        'num_rounds': 200,

        'metric' : 'auc',

        } 

    

    # Train the model

    bst = lgb.train(params, train_data, 100, valid_sets=[val_data])

    

    # Make the predictions storing them on the predictions array

    predictions += bst.predict(df_test.drop(['id'],axis=1))

    

    # Release the model from memory for the next iteration

    del bst



print('Training process finished. Generating Output...')



# We get the ammount of predictions from the prediction list, by dividing the predictions by the number of Kfolds.

predictions = predictions/3



# Read the sample_submission CSV

submission = pd.read_csv(INPUT_DATA_PATH + '/sample_submission.csv')

# Set the target to our predictions

submission.target=predictions

# Save the submission file

submission.to_csv('submission.csv',index=False)



print('Output created.')