import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

import os

print(os.listdir("../input"))

data = pd.read_csv('../input/data.csv')

data.head()
data['diagnosis'].value_counts().plot('bar');
data.isna().sum().plot('bar');
data.info()
cols = data.columns

cols
buffer = 128

batch_size = 128

epochs = 1000

split = 0.7

y_label = 'diagnosis'
default_type = []

for col in cols:

    if data[col].dtype == object:

        print(col)

        default_type.append([''])

    else:    

        default_type.append([0.0])

len(default_type) 
default_type[-1] = [0]

def parsing(line):

    parsed = tf.decode_csv(line,default_type[:32])

    print('parsing_line')

    features = dict(zip(cols,parsed))

    #features.pop('Unnamed: 32')

    features.pop('id')

    labels = features.pop(y_label)

    print(labels)

    return features, tf.equal(labels, 'M')
basedata = tf.data.TextLineDataset('../input/data.csv')

basedata = basedata.skip(1)
def in_train_set(line):

    print('in_train_set')

    num_buckets = 100000

    bucket_id = tf.string_to_hash_bucket_fast(line, num_buckets)

    return bucket_id < int(split * num_buckets)



def in_validate_set(line):

    return ~in_train_set(line)
train = basedata.filter(in_train_set).map(parsing)

validation = basedata.filter(in_validate_set).map(parsing)

def X():

    print('X()')

    return train.repeat().shuffle(buffer).batch(batch_size).make_one_shot_iterator().get_next()

def Y():

    return validation.shuffle(buffer).batch(batch_size).make_one_shot_iterator().get_next()
cols
fc = ['radius_mean', 'texture_mean', 'perimeter_mean',

       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',

       'symmetry_mean', 'fractal_dimension_mean',

       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',

       'compactness_se', 'concavity_se', 'symmetry_se',

       'fractal_dimension_se', 'radius_worst', 'texture_worst',

       'perimeter_worst', 'area_worst', 'smoothness_worst',

       'compactness_worst', 'concavity_worst',

       'symmetry_worst', 'fractal_dimension_worst']

feature_columns = list(map(lambda c : tf.feature_column.numeric_column(c), fc))
model = tf.estimator.DNNClassifier(hidden_units=[512,256],feature_columns=feature_columns)
model.train(input_fn= X, steps= epochs)
eval_result = model.evaluate(input_fn=Y)
for key in sorted(eval_result):

    print('%s: %s' % (key, eval_result[key]))
vald = data[150:250]

print(vald[y_label][150])
pred_iter = model.predict(input_fn= tf.estimator.inputs.pandas_input_fn(vald[fc],shuffle=False))

classes = ['B','M']

preds = []

for i,pred in enumerate(pred_iter):

    #print(classes[int(pred['classes'][0])],':- probabilities',pred['probabilities'][0])

    preds.append(int(pred['classes'][0]))
x = vald[y_label].apply(lambda x: 0 if x == 'B' else 1)
import seaborn as sns

import matplotlib.pyplot as plt

plt.figure(figsize=(8,8))

with tf.Graph().as_default():

    cm = tf.confusion_matrix(x,preds)

    with tf.Session() as sess:

        cm_out = sess.run(cm)



        sns.heatmap(cm_out, annot=True, xticklabels=classes, yticklabels=classes);

    plt.xlabel("Predicted");

    plt.ylabel("True");

    plt.title('Confusion Matrix M/B Cancer Type')
sess.close()