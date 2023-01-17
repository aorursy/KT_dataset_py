import numpy as np

import pandas as pd

import tensorflow as tf



import matplotlib.pyplot as plt

%matplotlib inline
try:

    data = pd.read_csv("../input/creditcard.csv")

except Exception as e:

    data = pd.read_csv("creditcard.csv")

data.head()
count_classes = pd.value_counts(data['Class'], sort = True).sort_index()

count_classes.plot(kind = 'bar')

plt.title("Fraud class histogram")

plt.xlabel("Class")

plt.ylabel("Frequency")
from sklearn.preprocessing import StandardScaler



data['normAmount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1))



# hour = data['Time'].apply(lambda x: np.ceil(float(x)/3600) % 24)

# data['hour'] = StandardScaler().fit_transform(hour.reshape(-1, 1))



data = data.drop(['Time','Amount'],axis=1)

data.head()
class Autoencoder(object):



    def __init__(self, n_hidden_1, n_hidden_2, n_input, learning_rate):

        self.n_hidden_1 = n_hidden_1

        self.n_hidden_2 = n_hidden_2

        self.n_input = n_input



        self.learning_rate = learning_rate



        self.weights, self.biases = self._initialize_weights()



        self.x = tf.placeholder("float", [None, self.n_input])



        self.encoder_op = self.encoder(self.x)

        self.decoder_op = self.decoder(self.encoder_op)



        self.cost = tf.reduce_mean(tf.pow(self.x - self.decoder_op, 2))

        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.cost)



        init = tf.initialize_all_variables()

        self.sess = tf.Session()

        self.sess.run(init)



    def _initialize_weights(self):

        weights = {

            'encoder_h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),

            'encoder_h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),

            'decoder_h1': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_hidden_1])),

            'decoder_h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_input])),

        }

        biases = {

            'encoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),

            'encoder_b2': tf.Variable(tf.random_normal([self.n_hidden_2])),

            'decoder_b1': tf.Variable(tf.random_normal([self.n_hidden_1])),

            'decoder_b2': tf.Variable(tf.random_normal([self.n_input])),

        }



        return weights, biases



    def encoder(self, X):

        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, self.weights['encoder_h1']),

                                       self.biases['encoder_b1']))

        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['encoder_h2']),

                                       self.biases['encoder_b2']))

        return layer_2



    def decoder(self, X):

        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, self.weights['decoder_h1']),

                                       self.biases['decoder_b1']))

        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self.weights['decoder_h2']),

                                       self.biases['decoder_b2']))

        return layer_2



    def calc_total_cost(self, X):

        return self.sess.run(self.cost, feed_dict={self.x: X})



    def partial_fit(self, X):

        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})

        return cost



    def transform(self, X):

        return self.sess.run(self.encoder_op, feed_dict={self.x: X})



    def reconstruct(self, X):

        return self.sess.run(self.decoder_op, feed_dict={self.x: X})
from sklearn.cross_validation import train_test_split
good_data = data[data['Class'] == 0]

bad_data = data[data['Class'] == 1]

#print 'bad: {}, good: {}'.format(len(bad_data), len(good_data))
X_train, X_test = train_test_split(data, test_size=0.2, random_state=42)



X_train = X_train[X_train['Class']==0]

X_train = X_train.drop(['Class'], axis=1)



y_test = X_test['Class']

X_test = X_test.drop(['Class'], axis=1)



X_train = X_train.values

X_test = X_test.values
X_good = good_data.ix[:, good_data.columns != 'Class']

y_good = good_data.ix[:, good_data.columns == 'Class']



X_bad = bad_data.ix[:, bad_data.columns != 'Class']

y_bad = bad_data.ix[:, bad_data.columns == 'Class']
model = Autoencoder(n_hidden_1=15, n_hidden_2=3, n_input=X_train.shape[1], learning_rate = 0.01)
training_epochs = 100

batch_size = 256

display_step = 100

record_step = 10
total_batch = int(X_train.shape[0]/batch_size)



cost_summary = []



for epoch in range(training_epochs):

    cost = None

    for i in range(total_batch):

        batch_start = i * batch_size

        batch_end = (i + 1) * batch_size

        batch = X_train[batch_start:batch_end, :]

        

        cost = model.partial_fit(batch)

    

    if epoch % display_step == 0 or epoch % record_step == 0:

        total_cost = model.calc_total_cost(X_train)

        

        if epoch % record_step == 0:

            cost_summary.append({'epoch': epoch+1, 'cost': total_cost})

        

        if epoch % display_step == 0:

            print("Epoch:{}, cost={:.9f}".format(epoch+1, total_cost))
f, ax1 = plt.subplots(1, 1, figsize=(10,4))



ax1.plot(list(map(lambda x: x['epoch'], cost_summary)), list(map(lambda x: x['cost'], cost_summary)))

ax1.set_title('Cost')



plt.xlabel('Epochs')

plt.show()
encode_decode = None

total_batch = int(X_test.shape[0]/batch_size) + 1

for i in range(total_batch):

    batch_start = i * batch_size

    batch_end = (i + 1) * batch_size

    batch = X_test[batch_start:batch_end, :]

    batch_res = model.reconstruct(batch)

    if encode_decode is None:

        encode_decode = batch_res

    else:

        encode_decode = np.vstack((encode_decode, batch_res))
def get_df(orig, ed, _y):

    rmse = np.mean(np.power(orig - ed, 2), axis=1)

    return pd.DataFrame({'rmse': rmse, 'target': _y})
df = get_df(X_test, encode_decode, y_test)
df.describe()
fig = plt.figure(figsize=(10,4))

ax = fig.add_subplot(111)

_ = ax.hist(df[df['target']== 0].rmse.values, bins=20)
fig = plt.figure(figsize=(10,4))

ax = fig.add_subplot(111)

_ = ax.hist(df[(df['target']== 0) & (df['rmse'] < 10)].rmse.values, bins=20)
fig = plt.figure(figsize=(10,4))

ax = fig.add_subplot(111)

_ = ax.hist(df[df['target'] > 0].rmse.values, bins=20)
fig = plt.figure(figsize=(10,4))

ax = fig.add_subplot(111)

_ = ax.hist(df[(df['target'] > 0) & (df['rmse'] < 10)].rmse.values, bins=20)
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc, roc_auc_score, 

                             roc_curve, recall_score, classification_report, f1_score,

                             precision_recall_fscore_support) 
import itertools



def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix.

    Normalization can be applied by setting `normalize=True`.

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=0)

    plt.yticks(tick_marks, classes)



    if normalize:

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    else:

        1



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")



    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')
fpr, tpr, thresholds = roc_curve(df.target, df.rmse)

roc_auc = auc(fpr,tpr)



# Plot ROC

plt.title('Receiver Operating Characteristic')

plt.plot(fpr, tpr, 'b',label='AUC = %0.4f'% roc_auc)

plt.legend(loc='lower right')

plt.plot([0,1],[0,1],'r--')

plt.xlim([-0.001, 1])

plt.ylim([0, 1.001])

plt.ylabel('True Positive Rate')

plt.xlabel('False Positive Rate')

plt.show()
precision, recall, th = precision_recall_curve(df.target, df.rmse)

plt.plot(recall, precision, 'b', label='Precision-Recall curve')

plt.xlabel('Recall')

plt.ylabel('Precision')

plt.show()
plt.plot(th, precision[1:], 'b', label='Threshold-Precision curve')

plt.xlabel('Threshold')

plt.ylabel('Precision')

plt.show()
# Compute confusion marix

y_pred = [1 if p > 2 else 0 for p in df.rmse.values]

cnf_matrix = confusion_matrix(df.target, y_pred)

np.set_printoptions(precision=2)



print("Recall metric in the testing dataset: ", float(cnf_matrix[1,1])/(cnf_matrix[1,0]+cnf_matrix[1,1]))



class_names = [0,1]

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names, title='Confusion matrix')

plt.show()
f1_score(y_pred=y_pred, y_true=df.target)
precision_recall_fscore_support(y_pred=y_pred, y_true=df.target)