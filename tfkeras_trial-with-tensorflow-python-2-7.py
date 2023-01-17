import tensorflow as tf

import pandas as pd

import numpy as np
data = pd.read_csv('../input/No-show-Issue-Comma-300k.csv')
data.head(3)
max_age, min_age = data.Age.max(), data.Age.min()
data.Age = data.Age.apply(lambda x: float(x-min_age)/(max_age-min_age))
data.Age.describe()
data.head(3)
data.Gender = data.Gender.apply(lambda x: 0 if x =='M' else 1)
data.head(3)
min_atime, max_atime = data.AwaitingTime.min(), data.AwaitingTime.max()
data.AwaitingTime = data.AwaitingTime.apply(lambda x: float(x-min_atime)/(max_atime-min_atime))
data.Status = data.Status.apply(lambda x: 1 if x =='Show-Up' else 0)
list_wod = data.DayOfTheWeek.unique()
dic_wod = {}

for i, e in enumerate(list_wod):

    dic_wod[e] = i
data.DayOfTheWeek = data.DayOfTheWeek.apply(lambda x: float(dic_wod[x])/len(list_wod))
data.head(3)
data.AppointmentRegistration.apply(lambda x: len(x)).unique()
data.ApointmentData.apply(lambda x: len(x)).unique()
data.AppointmentRegistration.apply(lambda x: x[2:4]).unique()
data.ApointmentData.apply(lambda x: x[2:4]).unique()
import re
pat = re.compile(r'201(\d*)-(\d*)-(\d*).(\d*).*$')
arr_reg = np.ndarray([len(data), 4])



for i, e in enumerate(data.AppointmentRegistration):

    arr_reg[i] = list(re.search(pat, e).groups())
arr_reg[:3], arr_reg.dtype
df_reg = pd.DataFrame(arr_reg, columns=['reg_y', 'reg_m', 'reg_d', 'reg_h'])
df_reg = (df_reg-df_reg.min())/(df_reg.max()-df_reg.min())
df_reg.head(3)
arr_apo = np.ndarray([len(data), 4])



for i, e in enumerate(data.ApointmentData):

    arr_apo[i] = list(re.search(pat, e).groups())
df_apo = pd.DataFrame(arr_apo, columns=['apo_y', 'apo_m', 'apo_d', 'apo_h'])
df_apo = (df_apo-df_apo.min())/(df_apo.max()-df_apo.min())
data = data.join(df_reg)
data.head(3)
data = data.join(df_apo)
data = data.drop(['AppointmentRegistration', 'ApointmentData'], axis=1)
data.head(3)
labels = pd.get_dummies(data.Status, prefix='label')
data = data.join(labels)
data.head(3)
np.unique(arr_apo[:,-1])
data = data.drop('apo_h', axis=1)
data.info()
arr_data = data.as_matrix()
arr_data[:3]
np.random.shuffle(arr_data)
folds = np.split(arr_data, 5)
len(folds), folds[0].shape
g = tf.Graph()



with g.as_default():

    features = tf.placeholder(dtype=tf.float32, shape=[None, 20])

    labels = tf.placeholder(dtype=tf.float32, shape=[None, 2])

    

    w1 = tf.Variable(tf.truncated_normal(shape=[20, 10]))

    b1 = tf.Variable(tf.zeros(shape=[10]))

    n1 = tf.matmul(features, w1)+b1

    o1 = tf.nn.relu(n1)

    

    w2 = tf.Variable(tf.truncated_normal(shape=[10, 2]))

    b2 = tf.Variable(tf.zeros(shape=[2]))

    n2 = tf.matmul(o1, w2)+b2

    o2 = tf.nn.softmax(n2)

    

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=n2)

    op_train = tf.train.GradientDescentOptimizer(0.000001).minimize(cross_entropy)

    op_init = tf.global_variables_initializer()

    

    correct_prediction = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(labels, axis=1), tf.argmax(o2, axis=1)), tf.float32))
accrs = []



for i in range(5):

    print('start fold #', i)

    

    train_features = np.concatenate([fold[:,:-2] for idx_fold, fold in enumerate(folds) if idx_fold!=i])

    train_labels = np.concatenate([fold[:, -2:] for idx_fold, fold in enumerate(folds) if idx_fold!=i])

    test_features, test_labels = folds[i][:,:-2], folds[i][:,-2:]

    

    with tf.Session(graph=g) as sess:

        sess.run(op_init)

        

        for j in range(500):

            _, cur_accr = sess.run([op_train, correct_prediction], feed_dict={features:train_features, labels:train_labels})

            

            if j%100 == 0 and j > 0:

                print('accuracy of #%d epoch: %f' % (j, np.mean(cur_accr)))

        

        accr = sess.run(correct_prediction, feed_dict={features:test_features, labels:test_labels})

        accrs.append(accr)

        print('accuracy of #%d fold: %f' % (i, accr))



print('avg accuracy:', sum(accrs)/len(accrs))
np.mean(folds[0][:,-1])