import numpy as np

import pandas as pd

import tensorflow as tf

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import metrics

from sklearn.decomposition import PCA
dataset = pd.read_csv('../input/data.csv')

dataset = dataset.iloc[:, 1:-1]

dataset.head()
from sklearn.preprocessing import LabelEncoder



diagnosis_unique, diagnosis_count = np.unique(dataset['diagnosis'].values, return_counts = True)



for i in range(diagnosis_unique.shape[0]):

    print (diagnosis_unique[i], ': ', diagnosis_count[0])
dataset['diagnosis'] = LabelEncoder().fit_transform(dataset['diagnosis'])

correlation = dataset.corr()

plt.figure(figsize = (20, 20))

sns.heatmap(correlation, vmax = 1, square = True, annot = False)

plt.show()
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import Normalizer



# change into numpy form for neural network use later    

dataset_matrix = dataset.values

dataset_matrix[:, 0] = LabelEncoder().fit_transform(dataset_matrix[:, 0])

label_matrix = dataset_matrix[:, 0]

dataset_matrix = dataset_matrix[:, 1:]



normalize_dataset_matrix = Normalizer().fit_transform(dataset_matrix)

std_normalize_dataset = StandardScaler().fit_transform(normalize_dataset_matrix)



mean_vec = np.mean(std_normalize_dataset, axis = 0)

cov_mat = (std_normalize_dataset - mean_vec).T.dot((std_normalize_dataset - mean_vec)) / (std_normalize_dataset.shape[0] - 1)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

eig_pairs.sort(key = lambda x: x[0], reverse=True)

tot = sum(eig_vals)

var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse = True)]

cum_var_exp = np.cumsum(var_exp)

plt.figure(figsize = (10, 5))

plt.bar(range(len(eig_pairs)), var_exp, alpha = 0.5, align = 'center', label = 'individual explained variance')

plt.step(range(len(eig_pairs)), cum_var_exp, where = 'mid', label = 'cumulative explained variance')

plt.ylabel('Explained variance ratio')

plt.xlabel('Principal components')

plt.legend(loc = 'best')

plt.tight_layout()

plt.show()
x_5d = PCA(n_components = 5).fit_transform(std_normalize_dataset)

colors = ['b', 'r']

for n, i in enumerate(np.unique(label_matrix)):

    plt.scatter(x_5d[:,0][label_matrix == i], x_5d[:,1][label_matrix == i], c = colors[n], label = diagnosis_unique[n], alpha = 0.7)

plt.legend()

plt.show()
class first_network:

    def __init__(self, learning_rate, x_shape, y_shape):

        self.X = tf.placeholder("float", [None, x_shape])

        self.Y = tf.placeholder("float", [None, y_shape])

        

        hidden1 = tf.Variable(tf.random_normal([x_shape, 512]))

        hidden2 = tf.Variable(tf.random_normal([512, 256]))

        hidden3 = tf.Variable(tf.random_normal([256, 128]))

        output = tf.Variable(tf.random_normal([128, y_shape]))



        hidden_bias1 = tf.Variable(tf.random_normal([512], stddev = 0.1))

        hidden_bias2 = tf.Variable(tf.random_normal([256], stddev = 0.1))

        hidden_bias3 = tf.Variable(tf.random_normal([128], stddev = 0.1))

        output_bias = tf.Variable(tf.random_normal([y_shape], stddev = 0.1))

        

        feedforward1 = tf.nn.relu(tf.matmul(self.X, hidden1) + hidden_bias1)

        feedforward2 = tf.nn.relu(tf.matmul(feedforward1, hidden2) + hidden_bias2)

        feedforward3 = tf.nn.relu(tf.matmul(feedforward2, hidden3) + hidden_bias3)

        

        self.logits = tf.matmul(feedforward3, output) + output_bias

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.Y, logits = self.logits))

        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)

        

        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))

        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))



class second_network:

    def __init__(self, learning_rate, x_shape, y_shape, beta = 0.00005):

        self.X = tf.placeholder("float", [None, x_shape])

        self.Y = tf.placeholder("float", [None, y_shape])

        

        hidden1 = tf.Variable(tf.random_normal([x_shape, 512]))

        hidden2 = tf.Variable(tf.random_normal([512, 256]))

        hidden3 = tf.Variable(tf.random_normal([256, 128]))

        output = tf.Variable(tf.random_normal([128, y_shape]))



        hidden_bias1 = tf.Variable(tf.random_normal([512], stddev = 0.1))

        hidden_bias2 = tf.Variable(tf.random_normal([256], stddev = 0.1))

        hidden_bias3 = tf.Variable(tf.random_normal([128], stddev = 0.1))

        output_bias = tf.Variable(tf.random_normal([y_shape], stddev = 0.1))

        

        feedforward1 = tf.nn.dropout(tf.nn.relu(tf.matmul(self.X, hidden1) + hidden_bias1), 0.5)

        feedforward2 = tf.nn.dropout(tf.nn.relu(tf.matmul(feedforward1, hidden2) + hidden_bias2), 0.5)

        feedforward3 = tf.nn.dropout(tf.nn.relu(tf.matmul(feedforward2, hidden3) + hidden_bias3), 0.5)

        

        self.logits = tf.matmul(feedforward3, output) + output_bias

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.Y, logits = self.logits))

        self.cost += tf.nn.l2_loss(hidden1) * beta + tf.nn.l2_loss(hidden2) * beta + tf.nn.l2_loss(hidden3) * beta + tf.nn.l2_loss(output) * beta

        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)

        

        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))

        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

        

class third_network:

    def __init__(self, learning_rate, x_shape, y_shape, beta = 0.00005):

        self.X = tf.placeholder("float", [None, x_shape])

        self.Y = tf.placeholder("float", [None, y_shape])

        

        hidden1 = tf.Variable(tf.random_normal([x_shape, 512]))

        hidden2 = tf.Variable(tf.random_normal([512, 256]))

        hidden3 = tf.Variable(tf.random_normal([256, 128]))

        output = tf.Variable(tf.random_normal([128, y_shape]))



        hidden_bias1 = tf.Variable(tf.random_normal([512], stddev = 0.1))

        hidden_bias2 = tf.Variable(tf.random_normal([256], stddev = 0.1))

        hidden_bias3 = tf.Variable(tf.random_normal([128], stddev = 0.1))

        output_bias = tf.Variable(tf.random_normal([y_shape], stddev = 0.1))

        

        feedforward1 = tf.nn.relu(tf.matmul(self.X, hidden1) + hidden_bias1)

        feedforward1 = tf.nn.dropout(tf.layers.batch_normalization(feedforward1), 0.5)

        feedforward2 = tf.nn.relu(tf.matmul(feedforward1, hidden2) + hidden_bias2)

        feedforward2 = tf.nn.dropout(tf.layers.batch_normalization(feedforward2), 0.5)

        feedforward3 = tf.nn.relu(tf.matmul(feedforward2, hidden3) + hidden_bias3)

        feedforward3 = tf.nn.dropout(tf.layers.batch_normalization(feedforward3), 0.5)

        

        self.logits = tf.matmul(feedforward3, output) + output_bias

        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = self.Y, logits = self.logits))

        self.cost += tf.nn.l2_loss(hidden1) * beta + tf.nn.l2_loss(hidden2) * beta + tf.nn.l2_loss(hidden3) * beta + tf.nn.l2_loss(output) * beta

        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.cost)

        

        correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.Y, 1))

        self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
def train(model, x_train, y_train, x_test, y_test, epoch, batch):

    LOST, ACC_TRAIN, ACC_TEST = [], [], []

    for i in range(epoch):

        loss, acc_train = 0, 0

        for n in range(0, (x_train.shape[0] // batch) * batch, batch):

            onehot = np.zeros((batch, np.unique(y_train).shape[0]))

            

            # change to one-hot for cross entropy

            for k in range(batch):

                onehot[k, int(y_train[n + k])] = 1.0

            

            cost, _ = sess.run([model.cost, model.optimizer], feed_dict = {model.X : x_train[n: n + batch, :], model.Y : onehot})

            acc_train += sess.run(model.accuracy, feed_dict = {model.X : x_train[n: n + batch, :], model.Y : onehot})

            loss += cost

            

        loss /= (x_train.shape[0] // batch)

        acc_train /= (x_train.shape[0] // batch)

        LOST.append(loss); ACC_TRAIN.append(acc_train)

        

        print ('epoch: ', i + 1, ', loss: ', loss, ', accuracy: ', acc_train)

        

        onehot = np.zeros((y_test.shape[0], np.unique(y_test).shape[0]))

        

        # change to one-hot for cross entropy

        for k in range(y_test.shape[0]):

            onehot[k, int(y_test[k])] = 1.0

            

        testing_acc, logits = sess.run([model.accuracy, tf.cast(tf.argmax(model.logits, 1), tf.int32)], feed_dict = {model.X : x_test, model.Y : onehot})

        

        print ('testing accuracy: ', testing_acc)

        print (metrics.classification_report(y_test, logits, target_names = diagnosis_unique))

        

        ACC_TEST.append(testing_acc)

        

    plt.subplot(1, 2, 1)

    x_component = [i for i in range(len(LOST))]

    plt.plot(x_component, LOST)

    plt.xlabel('epoch')

    plt.ylabel('loss')

    plt.subplot(1, 2, 2)

    plt.plot(x_component, ACC_TRAIN, label = 'train accuracy')

    plt.plot(x_component, ACC_TEST, label = 'test accuracy')

    plt.legend()

    plt.xlabel('epoch')

    plt.ylabel('loss')

    plt.show()
EPOCH = 10

BATCH = 32

LEARNING_RATE = 0.001



from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(dataset_matrix, label_matrix, test_size = 0.2)
tf.reset_default_graph()

sess = tf.InteractiveSession()

model = first_network(LEARNING_RATE, X_train.shape[1], diagnosis_unique.shape[0])

sess.run(tf.global_variables_initializer())

train(model, X_train, Y_train, X_test, Y_test, EPOCH, BATCH)
tf.reset_default_graph()

sess = tf.InteractiveSession()

model = second_network(LEARNING_RATE, X_train.shape[1], diagnosis_unique.shape[0])

sess.run(tf.global_variables_initializer())

train(model, X_train, Y_train, X_test, Y_test, EPOCH, BATCH)
tf.reset_default_graph()

sess = tf.InteractiveSession()

model = third_network(LEARNING_RATE, X_train.shape[1], diagnosis_unique.shape[0])

sess.run(tf.global_variables_initializer())

train(model, X_train, Y_train, X_test, Y_test, EPOCH, BATCH)
X_train, X_test, Y_train, Y_test = train_test_split(normalize_dataset_matrix, label_matrix, test_size = 0.2)
tf.reset_default_graph()

sess = tf.InteractiveSession()

model = first_network(LEARNING_RATE, X_train.shape[1], diagnosis_unique.shape[0])

sess.run(tf.global_variables_initializer())

train(model, X_train, Y_train, X_test, Y_test, EPOCH, BATCH)
tf.reset_default_graph()

sess = tf.InteractiveSession()

model = second_network(LEARNING_RATE, X_train.shape[1], diagnosis_unique.shape[0])

sess.run(tf.global_variables_initializer())

train(model, X_train, Y_train, X_test, Y_test, EPOCH, BATCH)
tf.reset_default_graph()

sess = tf.InteractiveSession()

model = third_network(LEARNING_RATE, X_train.shape[1], diagnosis_unique.shape[0])

sess.run(tf.global_variables_initializer())

train(model, X_train, Y_train, X_test, Y_test, EPOCH, BATCH)
x_2d = x_5d[:, :2]

X_train, X_test, Y_train, Y_test = train_test_split(x_2d, label_matrix, test_size = 0.2)



tf.reset_default_graph()

sess = tf.InteractiveSession()

model = first_network(LEARNING_RATE, X_train.shape[1], diagnosis_unique.shape[0])

sess.run(tf.global_variables_initializer())

train(model, X_train, Y_train, X_test, Y_test, 20, BATCH)
plt.figure(figsize = (30, 10))

x_min, x_max = x_2d[:, 0].min() - 0.5, x_2d[:, 0].max() + 0.5

y_min, y_max = x_2d[:, 1].min() - 0.5, x_2d[:, 1].max() + 0.5

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))



ax = plt.subplot(1, 2, 1)

ax.set_title('Input data')

ax.scatter(X_train[:, 0], X_train[:, 1], c = Y_train, cmap = plt.cm.Set1, label = diagnosis_unique)

ax.scatter(X_test[:, 0], X_test[:, 1], c = Y_test, cmap = plt.cm.Set1, alpha = 0.6)

ax.set_xlim(xx.min(), xx.max())

ax.set_ylim(yy.min(), yy.max())

ax.set_xticks(())

ax.set_yticks(())





ax = plt.subplot(1, 2, 2)

contour = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)

Z = sess.run(tf.nn.softmax(model.logits), feed_dict = {model.X: contour})

temp_answer = []

for q in range(Z.shape[0]):

    temp_answer.append(np.argmax(Z[q]))

Z = np.array(temp_answer)

Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, cmap = plt.cm.Set1, alpha = 0.4)

ax.scatter(X_train[:, 0], X_train[:, 1], c = Y_train, cmap = plt.cm.Set1, label = diagnosis_unique)

ax.scatter(X_test[:, 0], X_test[:, 1], c = Y_test, cmap = plt.cm.Set1, alpha = 0.6)

ax.set_xlim(xx.min(), xx.max())

ax.set_ylim(yy.min(), yy.max())

ax.set_xticks(())

ax.set_yticks(())

ax.set_title('hypothesis space')

plt.tight_layout()

plt.show()