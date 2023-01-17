import numpy as np

import pandas as pd



from sklearn.metrics import roc_auc_score

        

import tensorflow as tf

import tensorflow.keras.backend as K

from tensorflow.keras.layers import Input, Dense

from sklearn.model_selection import KFold
train = pd.read_csv(

    '/kaggle/input/riiid-test-answer-prediction/train.csv',

    usecols=[

        'user_id', 

        'content_id', 

        'user_answer', 

        'answered_correctly', 

        'prior_question_elapsed_time',

        'prior_question_had_explanation'

    ],

       dtype={

           'user_id': 'int32',

           'content_id': 'int16',

           'user_answer': 'int8',

           'answered_correctly': 'int8',

           'prior_question_elapsed_time': 'float32', 

           'prior_question_had_explanation': 'boolean'

       }

)

train = train.sort_values(['timestamp'], ascending=True)

questions = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/questions.csv')

lectures = pd.read_csv('/kaggle/input/riiid-test-answer-prediction/lectures.csv')
train = train.loc[train['answered_correctly'] != -1].reset_index(drop=True)

train
train['prior_question_had_explanation'] = train['prior_question_had_explanation'].fillna(value = False).astype(bool)

train
features_df = train.iloc[:int(9 /10 * len(train))]

train_df = train.iloc[int(9 /10 * len(train)):]
grouped_by_user_df = features_df.groupby('user_id')

user_answers_df = grouped_by_user_df.agg({'answered_correctly': ['mean', 'count', 'std', 'median', 'skew', 'var']}).copy()

user_answers_df.columns = ['mean_user_accuracy', 'questions_answered', 'std_user_accuracy', 'median_user_accuracy', 'skew_user_accuracy', 'var_user_accuracy']
grouped_by_content_df = features_df.groupby('content_id')

content_answers_df = grouped_by_content_df.agg({'answered_correctly': ['mean', 'count', 'std', 'median', 'skew', 'var']}).copy()

content_answers_df.columns = ['mean_accuracy', 'question_asked', 'std_accuracy', 'median_accuracy', 'skew_accuracy', 'var_accuracy']
import gc

del features_df

del grouped_by_user_df

del grouped_by_content_df



gc.collect()
train_df = train_df.merge(user_answers_df, how='left', on='user_id')

train_df = train_df.merge(content_answers_df, how='left', on='content_id')

train_df
features = [

    'mean_user_accuracy', 

    'questions_answered',

    'std_user_accuracy', 

    'median_user_accuracy',

    'skew_user_accuracy',

    'var_user_accuracy',

    'mean_accuracy', 

    'question_asked',

    'std_accuracy', 

    'median_accuracy',

    'prior_question_elapsed_time', 

    'prior_question_had_explanation',

    'skew_accuracy',

    'var_accuracy'

]

target = 'answered_correctly'
train_df = train_df[features + [target]]

train_df
train_df = train_df.replace([np.inf, -np.inf], np.nan)

train_df = train_df.fillna(0)

train_df
train_df['prior_question_had_explanation'] = train_df['prior_question_had_explanation'].astype(np.int8)

train_df
def create_model():

    model = tf.keras.Sequential([

        tf.keras.layers.Input(14),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dense(100, activation="relu"),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.5),

        tf.keras.layers.Dense(20, activation="relu"),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(1, activation="sigmoid")

    ])

    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])

    return model
res = pd.DataFrame()

res['row_id'] = [i for i in range(9927130)]

res.loc[:, ['answered_correctly']] = 0

models = []



for n, (tr, te) in enumerate(KFold(n_splits=3, random_state=666, shuffle=True).split(train_df[target])):

    print(f'Fold {n}')

    

    model = create_model()

    

    model.fit(

        train_df[features].values[tr],

        train_df[target].values[tr],

        validation_split=0.2,

        epochs=35, 

        batch_size=1024

    )



    res.loc[te, ['answered_correctly']] = model.predict(train_df[features].values[te])

    models.append(model)
print('NN score: ', roc_auc_score(train_df[target].values, res[target].values))
import riiideducation



env = riiideducation.make_env()
iter_test = env.iter_test()
for (test_df, sample_prediction_df) in iter_test:

    y_preds = []

    test_df = test_df.merge(user_answers_df, how = 'left', on = 'user_id')

    test_df = test_df.merge(content_answers_df, how = 'left', on = 'content_id')

    test_df['prior_question_had_explanation'] = test_df['prior_question_had_explanation'].fillna(value = False).astype(bool)

    test_df['prior_question_had_explanation'] = test_df['prior_question_had_explanation'].astype(np.int8)

    test_df = test_df.replace([np.inf, -np.inf], np.nan)

    test_df.fillna(value=0, inplace = True)



    for model in models:

        y_pred = model.predict(test_df[features].values)

        y_preds.append(y_pred)



    y_preds = sum(y_preds) / len(y_preds)

    test_df['answered_correctly'] = y_preds

    env.predict(test_df.loc[test_df['content_type_id'] == 0, ['row_id', 'answered_correctly']])