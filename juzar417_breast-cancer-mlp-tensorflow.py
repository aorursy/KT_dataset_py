# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import tensorflow as tf

from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

import matplotlib.pyplot as plt

import os

seed = 42

rng = np.random.RandomState(seed)

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
path = '../input/data.csv'



df = pd.read_csv(path)

df.head()
df = df.drop(['id'], axis=1)
corr = df.corr().round(2)

#Mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(25,25))



cmap = sns.diverging_palette(240, 10, as_cmap = True)



sns.heatmap(corr, mask = mask, cmap = cmap, vmax = 1, center=0, square = True, linewidth=.5, annot=True)

plt.tight_layout()
cols = ['radius_worst',

       'texture_worst',

       'perimeter_worst',

       'area_worst',

       'smoothness_worst',

       'compactness_worst',

       'concavity_worst',

       'concave points_worst',

       'symmetry_worst',

       'fractal_dimension_worst',

       'perimeter_mean',

       'perimeter_se',

       'area_mean',

       'area_se',

       'concavity_mean',

       'concavity_se',

       'concave points_mean',

       'concave points_se']



df = df.drop(cols, axis=1)
corr = df.corr().round(2)

#Mask for the upper triangle

mask = np.zeros_like(corr, dtype=np.bool)

mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(25,25))



cmap = sns.diverging_palette(240, 10, as_cmap = True)



sns.heatmap(corr, mask = mask, cmap = cmap, vmax = 1, center=0, square = True, linewidth=.5, annot=True)

plt.tight_layout()
X_df = df

X_df = X_df.drop(['Unnamed: 32','diagnosis'], axis=1)

y_df = df['diagnosis']

y_df = [0 if x == 'B' else 1 for x in y_df]
def dense_to_one_hot(labels_dense, num_classes=2):

    """Convert class labels from scalars to one-hot vectors"""

    num_labels = labels_dense.shape[0]

    index_offset = np.arange(num_labels) * num_classes

    labels_one_hot = np.zeros((num_labels, num_classes))

    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot



def preproc(unclean_batch_x):

    #convert to 0-1 normalized INputs

    temp_batch = unclean_batch_x/unclean_batch_x.max()

    return temp_batch



def batch_create(batch_size, dataset_len, dataset_name):

    mask = rng.choice(dataset_len, batch_size)

    batch_x = eval('x_tr').iloc[mask].values.reshape(-1, n_input)

    batch_x = preproc(batch_x)

    if dataset_name == 'train':

        batch_y = eval('y_tr')[mask]

        batch_y = dense_to_one_hot(batch_y)

        return batch_x, batch_y
x_tr, x_ts, y_tr, y_ts = train_test_split(X_df, y_df, test_size = 0.3, random_state = 42)

y_tr = np.asarray(y_tr, dtype = np.int)

y_ts = np.asarray(y_ts, dtype = np.int)

y_ts = dense_to_one_hot(y_ts)

print(x_ts.shape)

print(x_tr.shape)

print(y_ts.shape)

print(y_tr.shape)
lr = 0.005

n_epoch = 100

batch_size = 64

disp_step = 10



#Network parameters

n_h1 = 32

n_h2 = 12

n_input = x_tr.shape[1]

n_classes = 2



#Graph Input Initialization

X = tf.placeholder("float", [None, n_input])

Y = tf.placeholder("float",[None, n_classes])



# Initialize weights and Biases

weights = {

    'h1':tf.Variable(tf.random_normal([n_input, n_h1])),

    'h2':tf.Variable(tf.random_normal([n_h1, n_h2])),

    'out':tf.Variable(tf.random_normal([n_h2,n_classes]))

}



biases = {

    'b1':tf.Variable(tf.random_normal([n_h1])),

    'b2':tf.Variable(tf.random_normal([n_h2])),

    'out':tf.Variable(tf.random_normal([n_classes]))

}



def mlp_model(x):

    layer_1 = tf.add(tf.matmul(x, weights['h1']),biases['b1'])

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])

    o_layer = tf.matmul(layer_2, weights['out'] + biases['out'])

    return o_layer



# construct the model

nn_model = mlp_model(X)



#Define loss and optimizer

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(

    logits = nn_model, labels=Y))

optimizer = tf.train.AdamOptimizer(learning_rate = lr)

train_op = optimizer.minimize(loss_op)



init = tf.global_variables_initializer()



with tf.Session() as sess:

    sess.run(init)

    

    for epoch in range(n_epoch):

        avg_cost = 0.0

        total_batch = int(x_tr.shape[0]/batch_size)

        

        for i in range(total_batch):

            batch_x, batch_y = batch_create(batch_size, x_tr.shape[0], 'train')

            _, c = sess.run([train_op, loss_op], feed_dict = {X:batch_x, Y:batch_y})

            #compute average loss

            avg_cost += c / total_batch

        

        if epoch % disp_step == 0:

            print('Epoch: ' '%04d' %(epoch+1), "cost = {:.9f}".format(avg_cost))

    print("Optimization Finished!")

    

    pred = tf.nn.softmax(nn_model)

    correct_predictions = tf.equal(tf.argmax(pred, 1), tf.argmax(Y,1))

    

    acc = tf.reduce_mean(tf.cast(correct_predictions, "float"))

    print(acc)

    print("Accuracy :" , acc.eval({X:x_ts, Y:y_ts})*100)
X_train, X_test, y_train, y_test = train_test_split(X_df,y_df,test_size=0.3, random_state=40)
y_train = np.asarray(y_train)

y_train = np.reshape(y_train, (-1,1))

print(y_train.shape)

y_test = np.asarray(y_test)

y_test = np.reshape(y_test, (-1,1))

print(y_test.shape)

print(X_train.shape)
import statsmodels.api as sm

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import confusion_matrix, classification_report, precision_score

clf = MLPClassifier(solver='lbfgs', alpha=5e-3,learning_rate='constant',hidden_layer_sizes=(32,6), random_state=40)

clf.fit(X_train,y_train)
results = clf.predict(X_test)

cfm_MLP = confusion_matrix(y_test, results)

true_negativeMLP = cfm_MLP[0][0]

false_positiveMLP = cfm_MLP[0][1]

false_negativeMLP = cfm_MLP[1][0]

true_positiveMLP = cfm_MLP[1][1]
print('Confusion Matrix : \n', cfm_MLP, '\n')



print('True Negative: ', true_negativeMLP)

print('False Positive: ', false_positiveMLP)

print('False Negative: ', false_negativeMLP)

print('True Positive: ', true_positiveMLP)



print('Correct Predictions ',

     round((true_negativeMLP + true_positiveMLP) / len(results) * 100,1), '%')