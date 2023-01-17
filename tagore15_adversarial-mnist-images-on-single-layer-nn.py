import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import tensorflow as tf



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print("Avaiable Datasets: ")

print(check_output(["ls", "../input"]).decode("utf8"))

print("Files in Digit Recognizer Dataset: ")

print(check_output(["ls", "../input/digit-recognizer"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
# read the data

df_train = pd.read_csv("../input/digit-recognizer/train.csv")

df_test = pd.read_csv("../input/digit-recognizer/test.csv")

df_train.head()
# Preprocessing..not doing train-test split

Y = df_train['label']

X = df_train.drop('label', axis=1)

# normalize input image between -1 and 1

X = X.apply(lambda x: x*2.0/255.0 - 1.0)
# explore the digits

fig1, ax1 = plt.subplots(1,15, figsize=(15,10))

for i in range(15):

    ax1[i].imshow(X.iloc[i].values.reshape((28,28)), cmap='gray')

    ax1[i].axis('off')

    ax1[i].set_title(Y[i])
# create the model

x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))

b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x, W) + b

# Define actual outputs

y_ = tf.placeholder(tf.float32, [None, 10])
# define loss and optimizer

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# Batch Generator

def get_batch(BATCH_SIZE=64):

    while (1):

        for i in range(0, len(X), BATCH_SIZE):

            x_train, y_train = X[i:i+BATCH_SIZE], Y[i:i+BATCH_SIZE]

            yield (x_train, y_train)
# Training

gen = get_batch(64)

sess = tf.InteractiveSession()

tf.global_variables_initializer().run()

for i in range(500):

    batch_xs, batch_ys = next(gen)

    batch_ys = np.eye(10)[batch_ys]

    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})



    # Test trained model

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    if i%50 == 0:

        print('ACCURACY:', sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))
# Apply FGSM

grad = tf.gradients(cross_entropy, x)

signed_grad = tf.sign(grad)

scaled_signed_grad = 0.8 * signed_grad

# generate adverserial example and rescale to get image between 0 and 255

adv_x = (((tf.clip_by_value((x + scaled_signed_grad), -1, 1)+ 1.0) * 0.5) * 255.0)
# Let us generate adversarial of 7th image in data i.e. 

fig, ax = plt.subplots(1,1, figsize=(1,10))

ax.imshow(X.iloc[6].values.reshape((28,28)), cmap='gray')

ax.axis('off')

ax.set_title("True")
# generate adversarial

one_hot_output = np.eye(10)[Y[6]].reshape(1,10)

advsarial_out = sess.run(adv_x, feed_dict={x: X.iloc[6].values.reshape(1,784), y_:one_hot_output})
fig, ax = plt.subplots(1,1, figsize=(1,10))

ax.imshow(advsarial_out.reshape((28,28)).reshape((28,28)), cmap='gray')

ax.axis('off')

ax.set_title("Fake")
# Original Prediction

original_prediction = sess.run(y, feed_dict={x: X.iloc[6].values.reshape(1,784), y_:one_hot_output})

print('Original Prediction',np.argmax(original_prediction, 1))
# Adversarial Prediction

adversarial_prediction = sess.run(y, feed_dict={x: advsarial_out.reshape(1,784), y_:one_hot_output})

print('Fake Prediction', np.argmax(adversarial_prediction, 1))