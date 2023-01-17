class DataSet(object):

  def __init__(self, images, labels):

    self.images = images

    self.labels = labels



    self._images = images

    self._labels = labels

    self._epochs_completed = 0

    self._index_in_epoch = 0

    self._num_examples = len(images)

  

  def to_shuffle(self):

    perm = np.arange(self._num_examples)

    np.random.shuffle(perm)

    self._images = self.images[perm]

    self._labels = self.labels[perm]



  def next_batch(self, batch_size, fake_data=False, shuffle=True):

    start = self._index_in_epoch



    if start == 0 and self._epochs_completed == 0 and shuffle:

      self.to_shuffle()

    

    if start + batch_size > self._num_examples:   #this end with next start

      self._epochs_completed += 1



      rest_num_examples = self._num_examples - start

      images_rest_part = self._images[start : self._num_examples]

      labels_rest_part = self._labels[start : self._num_examples]



      if shuffle:

        self.to_shuffle()

      

      start = 0

      self._index_in_epoch = batch_size - rest_num_examples

      end = self._index_in_epoch

      images_new_part = self._images[start : end]

      labels_new_part = self._labels[start : end]



      return np.concatenate((images_rest_part, images_new_part), axis=0), np.concatenate((labels_rest_part, labels_new_part), axis=0)



    else:

      self._index_in_epoch += batch_size

      end = self._index_in_epoch

      return self._images[start : end], self._labels[start : end]
def PreprocessData(raw_df):

    df = raw_df.drop(['name'], axis=1)

#     raw_df.isnull().sum()

    age_mean = df['age'].mean()

    df['age'] = df['age'].fillna(age_mean)

    fare_mean = df['fare'].mean()

    df['fare'] = df['fare'].fillna(fare_mean)

    df['sex'] = df['sex'].map({'female':0, 'male':1}).astype(int)

    x_onehot_df = pd.get_dummies(data=df, columns=['embarked'])



    dfarray = x_onehot_df.values

    label = dfarray[:,0].reshape([-1, 1])

    features = dfarray[:,1:]

    

    scaledFeatures = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(features)

    return scaledFeatures, label
import os



mainPath = '/kaggle/input'

filePath = os.path.join(mainPath, 'titanic3.xls')
from sklearn import preprocessing

import numpy as np

import pandas as pd

import tensorflow as tf

from time import time

import matplotlib.pyplot as plt



np.random.seed(10)
all_df = pd.read_excel(filePath)

all_df.head()
cols = ['survived', 'name', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']

all_df = all_df[cols]

all_df.head()
msk = np.random.rand(len(all_df)) < 0.8

print(msk)

print(~msk)

train_df = all_df[msk]

test_df = all_df[~msk]
x_train, y_train = PreprocessData(train_df)

x_test, y_test = PreprocessData(test_df)

print(x_train.shape)

print(y_train.shape)

print(x_train[:2])

print(y_train[:2])

print()

print(x_test.shape)

print(y_test.shape)

print(x_test[:2])

print(y_test[:2])



train_data = DataSet(x_train, y_train)
def layer(input_dim, output_dim, bottom, activation=None):

    W = tf.Variable(tf.random.uniform([input_dim, output_dim]))

    b = tf.Variable(tf.random.uniform([1, output_dim]))

    xWb = tf.matmul(bottom, W) + b

    

    if activation:

        outputs = activation(xWb)

    else:

        outputs = xWb

    return outputs
# from tensorflow.keras.models import Sequential

# from tensorflow.keras.layers import Dense, Dropout



# model = Sequential()

# model.add(Dense(units=40, input_dim=9, kernel_initializer='uniform', activation='relu'))

# model.add(Dense(units=30, kernel_initializer='uniform', activation='relu'))

# model.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])



# train_his = model.fit(x=x_train, y=y_train, validation_split=0.1, epochs=30, batch_size=30, verbose=2)
# import matplotlib.pyplot as plt



# def show_train_history(train_history, train, validation):



#     plt.plot(train_history.history[train])



#     plt.plot(train_history.history[validation])



#     plt.title('Train History')



#     plt.ylabel('train')



#     plt.xlabel('Epoch')



#     plt.legend(['train', 'validation'], loc='center right')



#     plt.show()

# show_train_history(train_his, 'accuracy', 'val_accuracy')
# scores = model.evaluate(x=x_test, y=y_test)

# print(scores)
tf.compat.v1.disable_eager_execution()



X = tf.compat.v1.placeholder('float', [None, 9])

y_label = tf.compat.v1.placeholder('float', [None, 1])



h1 = layer(9, 40, X, tf.nn.relu)

h2 = layer(40, 30, h1, tf.nn.relu)

y_predict = layer(30, 1, h2, tf.nn.sigmoid)



loss_func = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_label, y_predict))



op = tf.compat.v1.train.AdamOptimizer(learning_rate=0.001).minimize(loss_func)



correct_pre = tf.equal(y_label, tf.where(tf.math.greater(y_predict, 0.5), 1.0, 0.0))



accuracy = tf.reduce_mean(tf.cast(correct_pre, 'float'))
epochs = 200

batch_size = 10

totalBatchs = int(x_train.shape[0] / batch_size)



epoch_list = list()

loss_list = list()

acc_list = list()



print('Start')

startTime = time()



sess = tf.compat.v1.Session()

sess.run(tf.compat.v1.global_variables_initializer())



for epoch in range(epochs):

    for i in range(totalBatchs):

        batch_x, batch_y = train_data.next_batch(batch_size)

        sess.run(op, feed_dict={X: batch_x, y_label: batch_y})

    

    loss, acc = sess.run([loss_func, accuracy], feed_dict={X: x_test, y_label: y_test})



    epoch_list.append(epoch)

    loss_list.append(loss)

    acc_list.append(acc)

    print('Train Epoch: {:02d}'.format(epoch + 1), 'Loss= {:.9f}'.format(loss), 'Accuracy= {}'.format(acc))

duration = time() - startTime

print('Train Finished takes:', duration)
print('Accuracy:', sess.run(accuracy, feed_dict={X: x_test, y_label: y_test}))
fig = plt.gcf()

fig.set_size_inches(10, 5)

plt.plot(epoch_list, loss_list, label='Loss')

plt.xlabel('Epoch')

plt.ylabel('Loss')

plt.legend()

plt.show()
fig2 = plt.gcf()

fig2.set_size_inches(10, 5)

plt.plot(epoch_list, acc_list, label='Accuracy')

plt.xlabel('Epoch')

plt.ylabel('Accuracy')

plt.legend()

plt.show()
Jack = pd.Series([0, 'Jack', 3, 'male', 23, 1, 0, 5.0000, 'S'])

Rose = pd.Series([1, 'Rose', 1, 'female', 20, 1, 0, 100.0000, 'S'])

JR_df = pd.DataFrame([list(Jack), list(Rose)], columns=['survived', 'name', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked'])

all_df = pd.concat([all_df, JR_df])

all_df[-2:]
all_features, all_label = PreprocessData(all_df)

print(all_features)

print(all_label)
all_probability = sess.run(y_predict, feed_dict={X: all_features})

print(all_probability)

end_df = all_df

end_df.insert(len(all_df.columns), 'probability', all_probability)
end_df.tail(2)