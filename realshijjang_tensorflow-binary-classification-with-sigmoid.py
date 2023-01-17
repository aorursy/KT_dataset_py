import matplotlib.pyplot as plt
%matplotlib inline
import tensorflow as tf
import numpy as np
import pandas as pd
import io
import requests
import math
from scipy import stats
# this is z-score that value minus mean divided by standard deviation
# http://duramecho.com/Misc/WhyMinusOneInSd.html
def feature_normalize(dataset):
    mu = np.mean(dataset,axis=0)
    sigma = np.std(dataset,axis=0)
    return (dataset - mu)/sigma

def str_to_int(df):
    str_columns = df.select_dtypes(['object']).columns
    print(str_columns)
    for col in str_columns:
        df[col] = df[col].astype('category')

    cat_columns = df.select_dtypes(['category']).columns
    df[cat_columns] = df[cat_columns].apply(lambda x: x.cat.codes)
    return df

def count_space_except_nan(x):
    if isinstance(x,str):
        return x.count(" ") + 1
    else :
        return 0

# https://stackoverflow.com/a/42523230
def one_hot(df, cols):
    """
    @param df pandas DataFrame
    @param cols a list of columns to encode 
    @return a DataFrame with one-hot encoding
    """
    for each in cols:
        dummies = pd.get_dummies(df[each], prefix=each, drop_first=False)
        del df[each]
        df = pd.concat([df, dummies], axis=1)
    return df
    
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
print (df_train.isnull().sum())
delete_columns = ["Ticket", "Name", "PassengerId", "Cabin", "Embarked"]

def pre_processing(df):
    df.drop(delete_columns, axis=1, inplace=True)
    # Count room nubmer
    # df_train["Cabin"] = df_train["Cabin"].apply(count_space_except_nan)
    # Replace NaN with mean value
    df["Age"].fillna(df["Age"].mean(), inplace=True)
    # Pclass, Embarked one-hot
    df = one_hot(df, df.loc[:, ["Pclass"]].columns)
    # String to int
    df = str_to_int(df)
    # Age Normalization
    df["Age"] = feature_normalize(df["Age"])
    stats.describe(df).variance
    return df
df_train = pre_processing(df_train)
#save PassengerId for evaluation
test_passenger_id = df_test["PassengerId"]
df_test = pre_processing(df_test)
features = df_train.iloc[:, 1:].values
# features = feature_normalize(features)
labels = df_train.iloc[:, :1].values
print(features.shape, labels.shape)
stats.describe(features).variance

real_test_x = df_test.values
print(real_test_x.shape)
rnd_indices = np.random.rand(len(features)) < 0.80

train_x = features[rnd_indices]
train_y = labels[rnd_indices]
test_x = features[~rnd_indices]
test_y = labels[~rnd_indices]

feature_count = train_x.shape[1]
label_count = train_y.shape[1]
print(feature_count, label_count)
# inputs
training_epochs = 3000
learning_rate = 0.01
hidden_layers = feature_count - 1
cost_history = np.empty(shape=[1],dtype=float)

X = tf.placeholder(tf.float32,[None,feature_count])
Y = tf.placeholder(tf.float32,[None,label_count])
is_training=tf.Variable(True,dtype=tf.bool)
# models

initializer = tf.contrib.layers.xavier_initializer()
h0 = tf.layers.dense(X, hidden_layers, activation=tf.nn.relu, kernel_initializer=initializer)
# h0 = tf.nn.dropout(h0, 0.95)
h1 = tf.layers.dense(h0, label_count, activation=None)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=h1)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# prediction = tf.argmax(h0, 1)
# correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

predicted = tf.nn.sigmoid(h1)
correct_pred = tf.equal(tf.round(predicted), Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# session

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(training_epochs + 1):
        sess.run(optimizer, feed_dict={X: train_x, Y: train_y})
        loss, _, acc = sess.run([cost, optimizer, accuracy], feed_dict={
                                 X: train_x, Y: train_y})
        cost_history = np.append(cost_history, acc)
        if step % 500 == 0:
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                step, loss, acc))
            
    # Test model and check accuracy
    print('Test Accuracy:', sess.run([accuracy, tf.round(predicted)], feed_dict={X: test_x, Y: test_y}))
    
    # Save test result
    test_predict_result = sess.run(tf.cast(tf.round(predicted), tf.int32), feed_dict={X: real_test_x})
    evaluation = test_passenger_id.to_frame()
    evaluation["Survived"] = test_predict_result
    evaluation.to_csv('result.csv', index=False)
print(cost_history.shape)
plt.plot(range(len(cost_history)),cost_history)
plt.axis([0,training_epochs,0,1])
plt.show()
evaluation
