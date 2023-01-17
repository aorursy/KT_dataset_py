# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.impute import SimpleImputer as SI





import tensorflow as tf

from tensorflow.keras import layers as L

from tensorflow.keras.models import Model



ss = StandardScaler()

si = SI(strategy="median")
dtypes_dict = {'row_id': 'int64',

               'timestamp': 'int64',

               'user_id': 'int32', 

               'content_id': 'int16',

               'content_type_id': 'int8',

               'task_container_id': 'int16', 

               'user_answer': 'int8', 

               'answered_correctly': 'int8',

               'prior_question_elapsed_time': 'float32', 

               'prior_question_had_explanation': 'boolean'

              }



train_df = pd.read_csv('../input/riiid-test-answer-prediction/train.csv',

                      nrows=10**7,

                      dtype=dtypes_dict,

                      index_col=0)  # as row_id is same as index, I am making it default index col

train_df
target_col = 'answered_correctly'



# let's see with how many classes we are dealing with

train_df[target_col].value_counts()
working_data = train_df[train_df[target_col]!=-1]

working_data
working_data.isna().sum()
working_data
print("Number of unique users: ", working_data.user_id.nunique())

print("Number of unique content(or unique user interaction): ", working_data.content_id.nunique())

print("Number of unique tasks(or batch of lectures): ", working_data.task_container_id.nunique())
userGroup = working_data.groupby("user_id")[target_col].mean().reset_index()

userGroup
contentGroup = working_data.groupby("content_id")[target_col].mean().reset_index()

contentGroup
taskGroup = working_data.groupby("task_container_id")[target_col].mean().reset_index()

taskGroup
userGroup.columns = ['user_id', 'user_performance']

contentGroup.columns = ['content_id', 'content_performance']

taskGroup.columns = ['task_container_id', 'task_performance']
working_data = working_data.reset_index()

working_data
features = ['timestamp', 'content_type_id', 'prior_question_elapsed_time', 'prior_question_had_explanation']

cat_cols = ['user_id', 'content_id', 'task_container_id']

selected_data = working_data[features + cat_cols + [target_col]].copy()

selected_data
def preprocess(df):

    """

    Merge user, task and content performance and return df with seleted features.

    """

    df.loc[:, 'timestamp'] = df['timestamp'].rolling(window=5, min_periods=1, center=True).sum()

    df.loc[:, 'prior_question_elapsed_time'] = df['prior_question_elapsed_time'].rolling(window=5, min_periods=1, center=True).sum()

    df = df.merge(userGroup, how='left', on='user_id')

    # deal with possible nan values

    df.loc[:, 'user_performance'] = df['user_performance'].fillna(0.5)

    df = df.merge(contentGroup, how='left', on='content_id')    

    df.loc[:, 'content_performance'] = df['content_performance'].fillna(0.5)

    df = df.merge(taskGroup, how='left', on='task_container_id') 

    df.loc[:, 'task_performance'] = df['task_performance'].fillna(0.5)

    

    # rescale the time values

    df['timestamp'] = ss.fit_transform(df['timestamp'].values.reshape(-1, 1))

    df['prior_question_elapsed_time'] = ss.fit_transform(df['prior_question_elapsed_time'].values.reshape(-1, 1))



    df['prior_question_had_explanation'] = df['prior_question_had_explanation'].map({True:1, False: 0})

    df['prior_question_had_explanation'] = si.fit_transform(df['prior_question_had_explanation'].values.reshape(-1, 1))



    return df
preprocess(selected_data)
final_features = ['timestamp',

                  'content_type_id',

                  'prior_question_elapsed_time',

                  'prior_question_had_explanation',

                  'user_performance',

                  'content_performance',

                  'task_performance']



final_train = preprocess(selected_data)[final_features + [target_col]]

final_train
def create_model(inp_dim):

    

    inp = L.Input(shape=(inp_dim, ))

    

    x = L.Dense(128, activation='relu')(inp)

    x = L.Dense(64, activation='relu')(x)

    x = L.Dense(1, activation='sigmoid')(x)

    

    model = Model(inp, x)

    model.compile(loss='binary_crossentropy',

                 optimizer='adam',

                 metrics=['accuracy'])



    return model



model_v1 = create_model(7)

model_v1.summary()
X= final_train.drop([target_col], axis=1).values

y = final_train[target_col].values

print(X.shape)

print(y.shape)
cv = StratifiedKFold(n_splits=5)

models = []

for i, (tr, val) in enumerate(cv.split(X, y)):

    print("===================")

    print(f"Fold: {i}")

    model_v1.fit(X[tr], y[tr],

                          epochs=5, batch_size=256,

                          validation_data=(X[val], y[val]),

                          callbacks=[tf.keras.callbacks.ModelCheckpoint(f"model_cv{i}.h5", save_best_only=True)])

    models.append(model_v1.load_weights(f"./model_cv{i}.h5"))
import riiideducation



env = riiideducation.make_env()

iter_test = env.iter_test()
for test_df, sample_prediction_df in iter_test:

    y_preds = []

    test_df = preprocess(test_df)

    x_test = test_df[final_features].values

    

    for model in models:

        y_pred = model_v1.predict(x_test, verbose=1)

        y_preds.append(y_pred)

    

    y_preds = sum(y_preds) / len(y_preds)

    test_df[target_col] = y_preds

    env.predict(test_df.loc[test_df['content_type_id'] == 0,

               ['row_id', target_col]])