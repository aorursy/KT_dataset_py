# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf



input_data = pd.read_csv("../input/train.csv")

input_data.head()
# Split data

msk = np.random.randn(len(input_data)) < 0.8

train_data = input_data.loc[msk]

test_data = input_data.loc[~msk]
# Manipulating Source Data

def process_data(data):

    P_class = data["Pclass"]

    sexes = []



    f_m = ["female", "male"]

    for sex in data["Sex"]:

        sexes.append(f_m.index(sex))



    age = data["Age"]

    age_average = age.mean()



    for i in range(0, len(age)):

        if pd.isnull(age.iat[i]):

            age.iat[i] = age_average



    sibsp = data["SibSp"]

    parch = data["Parch"]

    

    fare = data["Fare"]

    fare_average = fare.mean()

    

    for i in range(0, len(fare)):

        if pd.isnull(fare.iat[i]):

            fare.iat[i] = fare_average



    cabin = []



    cabins = ["T", "A", "B", "C", "D", "E", "F"]

    for c in data["Cabin"]:

        c = str(c)

        if c[0] in cabins:

            cabin.append(cabins.index(c[0]))

        else:

            cabin.append(4)

    

    embark = []

    embarks = ["C", "Q", "S"]

    for e in data["Embarked"]:

        if e in embarks:

            embark.append(embarks.index(e))

        else:

            embark.append(10)



    X = np.stack((sexes, embark), axis=1)

    return X



train_x = process_data(train_data)

raw_train_labels = train_data["Survived"]



X_df = pd.DataFrame(train_x,

                 columns=["Sex", "Embark"])

X_df.head()
# Process Labels

def process_labels(raw_labels):

    labels = [[0, 0] for x in range(0, len(raw_labels))]



    for i in range(0, len(raw_labels)):

        if raw_labels.iat[i] == 1:

            labels[i][1] = 1

        else:

            labels[i][0] = 1

            

    return labels



train_labels = process_labels(raw_train_labels)
tf.reset_default_graph()



# Creating tensorflow graph

# Inputs

x = tf.placeholder(tf.float32, [None, 2], name="inputs")



# Weights

W = tf.Variable(tf.zeros([2, 2]), name="W")

b = tf.Variable(tf.zeros([2]), name="b")



# Output

# y = Wx + b

y = tf.matmul(x, W) + b



# Labels

y_ = tf.placeholder(tf.float32, [None, 2], name="labels")



# Loss

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
# Train

train = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)



sess = tf.InteractiveSession()

tf.global_variables_initializer().run()



for _ in range(0, 1000):

    sess.run(train, feed_dict={x: train_x, y_: train_labels})



predictions = tf.argmax(y, 1)
# Set up test data

test_x = process_data(test_data)

raw_test_labels = test_data["Survived"]

test_labels = process_labels(raw_test_labels)
# Test with test data

correct = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))



print(sess.run(accuracy, feed_dict={x: test_x, y_: test_labels}))
val_data = pd.read_csv("../input/test.csv")

val_data.head()
val_X = process_data(val_data)

val_X_df = pd.DataFrame(val_X)

          #       columns=["P_class", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin", "Embarked"])

val_X_df.head()
# Test for missing values in test_X

np.any(np.isnan(val_X))
output = predictions.eval(feed_dict={x: val_X})

ids = val_data["PassengerId"]



output = pd.Series(output)



predictions_df = pd.DataFrame(np.stack([output, ids], axis=1),

                              columns=["Survived", "PassengerId"])

predictions_df.head()
# Save output to csv

predictions_df.to_csv("output.csv", index=False)