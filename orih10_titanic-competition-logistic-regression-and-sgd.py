import tensorflow as tf

import numpy as np

import pandas as pd

import csv

from sklearn.preprocessing import Imputer, LabelEncoder, MinMaxScaler, LabelBinarizer

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

%matplotlib inline  
train_data = pd.read_csv('train.csv')

test_data = pd.read_csv('test.csv')

original_test_data = test_data
train_data.head()
test_data.head()
train_data.describe()
survivors = train_data[train_data['Survived']==1]

males = train_data[train_data['Sex']=='male'].count()['PassengerId']

females = train_data[train_data['Sex']=='female'].count()['PassengerId']

total = train_data.count()['PassengerId']



male_surv = survivors[survivors['Sex']=='male'].sum()['Survived']

female_surv = survivors[survivors['Sex']=='female'].sum()['Survived']



df = pd.Series([male_surv, female_surv], index=['Male', 'Female'], name='Survivors')

df.plot.pie(autopct='%.2f', figsize=(8, 8), title='Survivors by gender', table=True)



surv_bar = pd.DataFrame([[males, females], [male_surv, female_surv], [males-male_surv, females-female_surv]],

                         columns=['Males', 'Females'], index=['Total', 'Survived', 'Not Survived'])

surv_bar.plot.bar()



male_vs_female = pd.DataFrame([[males, male_surv], [females, female_surv]],

                              columns=['Males', 'Females'], index=['Not Survived', 'Survived'])

male_vs_female.plot.pie(autopct='%.2f', figsize=(8, 4), subplots=True)

def nan_padding(data, columns):

    for column in columns:

        imputer=Imputer()

        data[column]=imputer.fit_transform(data[column].values.reshape(-1,1))

    return data
nan_columns = ["Age", "SibSp", "Parch"]

train_data = nan_padding(train_data, nan_columns)

test_data = nan_padding(test_data, nan_columns)
train_data
def dummy_data(data, columns):

    for column in columns:

        data = pd.concat([data, pd.get_dummies(data[column], prefix=column)], axis=1)

        data = data.drop(column, axis=1)

    return data

dummy_columns = ["Pclass"]

train_data=dummy_data(train_data, dummy_columns)

test_data=dummy_data(test_data, dummy_columns)
train_data
def sex_to_int(data):

    le = LabelEncoder()

    le.fit(["male","female"])

    data["Sex"]=le.transform(data["Sex"]) 

    return data



train_data = sex_to_int(train_data)

test_data = sex_to_int(test_data)
train_data
def normalize_age(data):

    scaler = MinMaxScaler()

    data["Age"] = scaler.fit_transform(data["Age"].values.reshape(-1,1))

    return data

train_data = normalize_age(train_data)

test_data = normalize_age(test_data)
train_data
def drop_not_concerned(data, columns):

    return data.drop(columns, axis=1)



not_concerned_columns = ["PassengerId","Name", "Ticket", "Fare", "Cabin", "Embarked"]

train_data = drop_not_concerned(train_data, not_concerned_columns)

test_data = drop_not_concerned(test_data, not_concerned_columns)
train_data
def split_valid_test_data(data, fraction=(1 - 0.8)):

    data_y = data["Survived"]

    lb = LabelBinarizer()

    data_y = lb.fit_transform(data_y)



    data_x = data.drop(["Survived"], axis=1)



    train_x, valid_x, train_y, valid_y = train_test_split(data_x, data_y, test_size=fraction)



    return train_x.values, train_y, valid_x, valid_y
train_x, train_y, valid_x, valid_y = split_valid_test_data(train_data)

print("train_x:{}".format(train_x.shape))

print("train_y:{}".format(train_y.shape))



print("train_y content:{}".format(train_y[:3]))



print("valid_x:{}".format(valid_x.shape))

print("valid_y:{}".format(valid_y.shape))
x = tf.placeholder(tf.float32, shape=(None, train_x.shape[1]))

y = tf.placeholder(tf.float32, shape=(None, 1))

weights = tf.Variable(tf.random_normal(shape=[train_x.shape[1], 1]), name='weights')

bias = tf.Variable(tf.random_normal(shape=[1]), name='bias')

sigmoid = tf.nn.sigmoid(tf.add(tf.matmul(x, weights), bias), name='sigmoid')

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=sigmoid, name='loss'))



learning_rate = 0.1

batch_size = train_x.shape[1] #128

training_epochs = 2000
optimaizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

pred = tf.cast(tf.greater_equal(sigmoid, 0.5), tf.float32, name='prediction')

acc = tf.reduce_mean(tf.cast(tf.equal(pred, y), tf.float32), name='accurecy')

init = tf.global_variables_initializer()


with tf.Session() as sess:

    sess.run(init)

    total_batch = int(len(train_x)/batch_size)

    acc_list = []

    avg_cost = []

    for epoch in range(training_epochs+1):

        temp_cost = []

        for batch in range(total_batch):

            _, c = sess.run([optimaizer, loss], feed_dict={x: train_x[batch*batch_size:(batch+1)*batch_size],

                                                           y:train_y[batch*batch_size:(batch+1)*batch_size]})

            temp_cost.append(c)

            avg_cost.append(c)

            acc_list.append(sess.run(acc, feed_dict={x: valid_x, y:valid_y}))

        if epoch % 10 ==0:

            acc_train = sess.run(acc, feed_dict={x: valid_x, y:valid_y})

            print('Epoch: {}/{}, cost: {}, acc:{}'.format(epoch, training_epochs, np.mean(temp_cost), acc_train))

    print('Optimization Finished!')

    evaluation_pred = sess.run(acc, feed_dict={x: valid_x, y:valid_y})

    print('Accuracy: {}'.format(evaluation_pred))

    test_pred = sess.run(pred, feed_dict={x: test_data})
with open('prediction.csv', 'w', newline='') as csvfile:

    pred_writer = csv.writer(csvfile)

    pred_writer.writerow(('PassengerId', 'Survived'))

    for i in range(len(test_pred)):

        pred_writer.writerow((original_test_data['PassengerId'][i], int(test_pred[i])))