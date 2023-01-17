import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
dtype = {

    'row_id': 'int64',

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
from collections import Counter





per_value_counts = {}

global_counts = Counter()





columns_target_encode = ['user_id', 'content_id', 'content_type_id', 'task_container_id']
def add_or_update(column, value, count, answered_correctly):

    # column

    column_data = per_value_counts.get(column, {})

    per_value_counts[column] = column_data

    # value

    value_data = column_data.get(value, Counter())

    column_data[value] = value_data

    # counters

    value_data += Counter({'count': count, 'answered_correctly': answered_correctly})





def update_counts(data, column):

    agg = data.groupby(column)['answered_correctly'].agg(['count', 'mean'])

    agg['answered_correctly'] = agg['count'] * agg['mean']

    for idx,row in agg.iterrows():

        add_or_update(column, idx, row['count'], row['answered_correctly'])





def update_global_counts(data):

    global global_counts

    count = data.shape[0]

    clicks = data[data['answered_correctly'] == 1].shape[0]

    global_counts += Counter({'count': count, 'answered_correctly': clicks})





def update_all_counts(data, columns):

    for column in columns:

        update_counts(data, column)

    update_global_counts(data)





def target_encode_value(column, value):

    counts = per_value_counts.get(column, {}).get(value, Counter())

    if 'answered_correctly' in counts:

        return counts['answered_correctly'] / counts['count']

    else:

        return global_counts['answered_correctly'] / global_counts['count']





def target_encode(data, columns):

    out = pd.DataFrame(index=data.index)

    for column in columns:

        out[column] = data[column].apply(lambda value: target_encode_value(column, value))

    return out
from sklearn.preprocessing import StandardScaler





columns_std = ['prior_question_elapsed_time']



scaler = StandardScaler()
columns_copy = ['prior_question_had_explanation']
def make_x(data):

    

    # copy without changes

    x = data[columns_copy].fillna(0)

    

    # target encode

    x_target_encode = target_encode(data, columns_target_encode)

    

    # std

    x_std = pd.DataFrame(scaler.transform(data[columns_std]), index=data.index, columns=columns_std).fillna(0)

    

    return pd.concat([x, x_target_encode, x_std], axis=1)
from sklearn.linear_model import SGDClassifier

from sklearn.metrics import roc_auc_score





estimator = SGDClassifier(loss='log')





chunksize = 10**6



train_count = 0

valid_count = 0



train_from = 1 * chunksize

validate_from = 90 * chunksize



auc = 0.



for train in pd.read_csv('/kaggle/input/riiid-test-answer-prediction/train.csv', dtype=dtype, chunksize=chunksize, iterator=True):

    

    train = train[train['answered_correctly'] != -1]

    

    if train_count >= train_from:

        

        x_train = make_x(train)

        y_train = train['answered_correctly']

        

        if train_count >= validate_from:

            y_pred = estimator.predict_proba(x_train)[:, 1]

            auc += roc_auc_score(y_train, y_pred)

            valid_count += chunksize

        

        estimator.partial_fit(x_train, y_train, classes=[0,1])

        train_count += chunksize

        

        print('Train count:', train_count, 'Validation count:', valid_count)

        

    else:

        

        train_count += chunksize

        print('Warmup count:', train_count)

    

    update_all_counts(train, columns_target_encode)

    scaler.partial_fit(train[columns_std])
print('Validation score:', auc * chunksize / valid_count)
import riiideducation



env = riiideducation.make_env()



iter_test = env.iter_test()
for (test, sample_prediction) in iter_test:

    

    x_test = make_x(test)

    

    test['answered_correctly'] = estimator.predict_proba(x_test)[:, 1]

    

    env.predict(test.loc[test['content_type_id'] == 0, ['row_id', 'answered_correctly']])