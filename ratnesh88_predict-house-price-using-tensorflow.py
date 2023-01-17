import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/boston_train.csv')
train.head()
train.info()
test = pd.read_csv('../input/boston_test.csv')
test.head()
cols = train.columns
cols
train.isna().sum()
default_types =[[0.0]]*len(cols)
default_types    
y_name = 'medv'
batch_size = 128
num_epochs = 400
buffer  = 1000
split = 0.7
def parse_line(line):
    columns = tf.decode_csv(line,default_types)
    features = dict(zip(cols,columns))

    label = features.pop(y_name)
    return features, label
data = tf.data.TextLineDataset('../input/boston_train.csv').skip(1)
def in_training_set(line):
    num_buckets = 1000000
    bucket_id = tf.string_to_hash_bucket_fast(line, num_buckets)
    return bucket_id < int(split * num_buckets)

def in_test_set(line):
    return ~in_training_set(line)
train = (data.filter(in_training_set).map(parse_line))
validation = (data.filter(in_test_set).map(parse_line))
def X():
    return train.repeat().shuffle(buffer).batch(batch_size).make_one_shot_iterator().get_next()
def Y():
    return validation.shuffle(buffer).batch(batch_size).make_one_shot_iterator().get_next()
sess = tf.Session()
#sess.run(validation)
feature_columns = []
for col in cols[1:-1]:
    feature_columns.append(tf.feature_column.numeric_column(col))
model = tf.estimator.DNNRegressor(feature_columns=feature_columns, hidden_units=[10,10])
model.train(input_fn= X,steps=500)
eval_result = model.evaluate(input_fn=Y)
for key in sorted(eval_result):
    print('%s: %s' % (key, eval_result[key]))
test.head()
test_in = tf.estimator.inputs.pandas_input_fn(test, shuffle=False)
test_in
pred_iter = model.predict(input_fn=test_in)
predC = []
for i,pred in enumerate(pred_iter):
    print(test['ID'][i],pred['predictions'][0])
    predC.append(pred['predictions'][0])
    
out_df = pd.DataFrame({"ID":test['ID'], "medv":predC})
file = out_df.to_csv("submission.csv", index=False)
print(os.listdir('../working'))