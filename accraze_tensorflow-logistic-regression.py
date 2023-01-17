# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

#train.head(3)
train.shape[0]
train.head()
def fix_age(data):
    """Creates Age Groups for every 10 years (i.e. 0-10, 10-20, ..., 70-80)"""
    data['Age_group'] = 0
    for i in range(data.shape[0]):
        for j in range(70, 0, -10):
            if data.at[i, 'Age'] > j:
                data.at[i, 'Age_group'] = int(j/10)
                break
    del data['Age']
    return data
def fix_cabins(data):
    """Removes the cabin number and just leaves section"""
#     print(list(set(data['Cabin'].values))[:10]) # sample of 'Cabin' values
    data['Cabin_section'] = '0'
    for i in range(data.shape[0]):
        if data.at[i, 'Cabin'] != 0:
            data.at[i, 'Cabin_section'] = data.at[i, 'Cabin'][0]
    CABIN_SECTION = list(set(data['Cabin_section'].values))
    for i in range(data.shape[0]):
        data.at[i, 'Cabin_section'] = CABIN_SECTION.index(data.at[i, 'Cabin_section'])
    del data['Cabin']
    return data
def clean_data(data):
    """Remove unused columns, fill na and """
    extra_cols = ['Fare', 'Ticket', 'Name', 'Embarked']
    for col in extra_cols:
        del data[col]
    data = data.fillna(value=0.0)
    data['Sex'].replace(['female','male'],[0,1],inplace=True)
    data = fix_age(data)
    data = fix_cabins(data)
    return data
train = clean_data(train)
train.head()
def convert_to_one_hot(data, mode='train'):
    pclass = np.eye(data['Pclass'].values.max()+1)[data['Pclass'].values]
    age_group = np.eye(data['Age_group'].values.max()+1)[data['Age_group'].values]
    cabin_section = np.eye(data['Cabin_section'].values.max()+1) \
                        [data['Cabin_section'].values.astype(int)] # prevent IndexError
    X = data[['Sex', 'SibSp', 'Parch']].values
    X = np.concatenate([X, age_group], axis=1)
    X = np.concatenate([X, pclass], axis=1)
    X = np.concatenate([X, cabin_section], axis=1)
    X = X.astype(float)
    if mode == 'train':
        y_col = 'Survived'
    else:
        y_col = 'PassengerId'
    y = data[y_col].values
    y = y.astype(float).reshape(-1, 1)
    return X,y
X,y = convert_to_one_hot(train)
from sklearn.model_selection import train_test_split
X_train, X_dev, y_train, y_dev = train_test_split(X, y, test_size=0.1, random_state=0)
test = clean_data(test)
test.head()
X_test,id_test = convert_to_one_hot(train, mode='test')
# Hyperparameters
in_size = X_train.shape[1]
learning_rate = 0.001
epochs = 10000
graph = tf.Graph()
with graph.as_default():
    X_input = tf.placeholder(dtype=tf.float32, shape=[None, in_size], name='X_input')
    y_input = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='y_input')
    
    W1 = tf.Variable(tf.random_normal(shape=[in_size, 1]), name='W1')
    b1 = tf.Variable(tf.random_normal(shape=[1]), name='b1')
    sigm = tf.nn.sigmoid(tf.add(tf.matmul(X_input, W1), b1), name='pred')
    
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_input,
                                                                  logits=sigm, name='loss'))
    train_steps = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    decision_boundary = 0.5
    pred = tf.cast(tf.greater_equal(sigm, decision_boundary), tf.float32, name='pred') # 1 if >= decision_boundary
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, y_input), tf.float32), name='acc')
    
    init_var = tf.global_variables_initializer()
train_feed_dict = {X_input: X_train, y_input: y_train}
test_feed_dict = {X_input: X_test} # no y_input since we need to predict it
dev_feed_dict = {X_input: X_dev, y_input: y_dev} # for accuracy printout
sess = tf.Session(graph=graph)
sess.run(init_var)
current_loss = sess.run(loss, feed_dict=train_feed_dict)
train_acc = sess.run(acc, feed_dict=train_feed_dict)
test_acc = sess.run(acc, feed_dict=dev_feed_dict)
print('step 0: loss {0:.5f}, train_acc {1:.2f}%, test_acc {2:.2f}%'.format(
                       current_loss, 100*train_acc, 100*test_acc))
for step in range(1, epochs+1):
    sess.run(train_steps, feed_dict=train_feed_dict)
    current_loss = sess.run(loss, feed_dict=train_feed_dict)
    train_acc = sess.run(acc, feed_dict=train_feed_dict)
    test_acc = sess.run(acc, feed_dict=dev_feed_dict)
    if step%100 != 0:
        continue
    print('step {0}: loss {1:.5f}, train_acc {2:.2f}%, test_acc {3:.2f}%'.format(
                       step,current_loss, 100*train_acc, 100*test_acc))
# y_pred = sess.run(pred, feed_dict=test_feed_dict).astype(int)
# prediction = pd.DataFrame(np.concatenate([id_test, y_pred], axis=1),
#                           columns=['PassengerId', 'Survived'])