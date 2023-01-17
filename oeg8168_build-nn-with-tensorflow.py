import warnings

warnings.filterwarnings("ignore")
import numpy as np

import pandas as pd



import tensorflow as tf

from tqdm import tqdm_notebook
from sklearn.preprocessing import StandardScaler



def data_preprocessing(df_input):

    # numeric feature standardization

    sc = StandardScaler()    

    column_names = df_input.columns[1:11]

    df = pd.DataFrame(sc.fit_transform(df_input.iloc[:, 1:11]))

    df.columns = column_names

    

    # reverse one-hot encoding

    Wilderness_Area = df_input.iloc[:, 11:15].idxmax(1).str.replace('Wilderness_Area', '')

    df['Wilderness_Area'] = pd.to_numeric(Wilderness_Area)

    

    # reverse one-hot encoding

    Soil_Type = df_input.iloc[:, 15:55].idxmax(1).str.replace('Soil_Type', '')

    df['Soil_Type'] = pd.to_numeric(Soil_Type)

    

    return df.join(df_input.iloc[:, 11:55])
train_df = pd.read_csv('../input/train.csv')

test_df = pd.read_csv('../input/test.csv')
train_df.head()
X = data_preprocessing(train_df)

y = train_df.iloc[:, -1]
X.describe()
# oversampling by Soil_Type



for i in range(5):

    threshold = X.Soil_Type.value_counts().median()

    need_over_sample_types = X.Soil_Type.value_counts()[(X.Soil_Type.value_counts() < threshold)].index

    oversampling_rows = X[X['Soil_Type'].isin(need_over_sample_types)].copy()

    

    y = y.append(y[oversampling_rows.index])

    X = X.append(oversampling_rows)



    y.reset_index(drop=True, inplace=True)

    X.reset_index(drop=True, inplace=True)
# oversampling by Wilderness_Area



for i in range(5):

    oversampling_rows = X[X['Wilderness_Area']==X.Wilderness_Area.value_counts().idxmin()].copy()

    

    y = y.append(y[oversampling_rows.index])

    X = X.append(oversampling_rows)



    y.reset_index(drop=True, inplace=True)

    X.reset_index(drop=True, inplace=True)
# prepare data for model

labels = pd.get_dummies(y)

num_labels = len(set(y))

feature_size = X.shape[1]
# setting hyperparameter

epochs = 3000

batch_size = 4096



reg_rate = 0.001

drop_rate = 0.1
tf.reset_default_graph() 



with tf.name_scope('input'):

    X_input = tf.placeholder(shape=(None, feature_size), 

                             name='X_input',

                             dtype=tf.float32)

    y_out = tf.placeholder(shape=(None, num_labels), 

                           name='y',

                           dtype=tf.float32)

    training_mode = tf.placeholder(tf.bool, name='training')



with tf.variable_scope('hidden'):

    X_h1 = tf.layers.dense(inputs=X_input, units=256, activation=tf.nn.relu,

                           kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_rate))

    X_h1 = tf.layers.dropout(inputs=X_h1, rate=drop_rate, training=training_mode)



    X_h2 = tf.layers.dense(inputs=X_h1, units=128, activation=tf.nn.relu,

                           kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_rate))

    X_h2 = tf.layers.dropout(inputs=X_h2, rate=drop_rate, training=training_mode)

    

    X_h3 = tf.layers.dense(inputs=X_h2, units=64, activation=tf.nn.relu,

                           kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_rate))

    X_h3 = tf.layers.dropout(inputs=X_h3, rate=drop_rate, training=training_mode)

        

    X_h4 = tf.layers.dense(inputs=X_h3, units=32, activation=tf.nn.relu,

                           kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_rate))

    X_h4 = tf.layers.dropout(inputs=X_h4, rate=drop_rate, training=training_mode)

    

    X_h5 = tf.layers.dense(inputs=X_h4, units=16, activation=tf.nn.relu,

                           kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_rate))

    X_h5 = tf.layers.dropout(inputs=X_h5, rate=drop_rate, training=training_mode)

    

    X_hf = tf.layers.dense(inputs=X_h5, units=8, activation=tf.nn.relu,

                           kernel_regularizer=tf.contrib.layers.l2_regularizer(reg_rate))

    

with tf.variable_scope('output'):

    output = tf.layers.dense(X_hf, num_labels, name='output')



with tf.name_scope('loss'):

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=y_out), name='cross_entropy')

    reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    loss = cross_entropy + reg_loss

    

with tf.name_scope('train'):

    train_step = tf.train.AdamOptimizer().minimize(loss)
from sklearn.utils import shuffle



# create a session and train the model

print('--- training start ---')

sess = tf.Session()

sess.run(tf.global_variables_initializer())



es_count = 0

best_loss = float('inf')

total_batch = len(X)//batch_size

for i in tqdm_notebook(range(epochs)):

    iter_loss = best_loss

    

    for j in range(total_batch):

        batch_idx_start = j * batch_size

        batch_idx_stop = (j+1) * batch_size



        X_batch = X[batch_idx_start : batch_idx_stop] 

        y_batch = labels[batch_idx_start : batch_idx_stop]



        batch_losses, _ = sess.run([loss, train_step], feed_dict={X_input: X_batch, y_out: y_batch, training_mode: True})



        batch_loss = batch_losses.mean()

        

        if batch_loss < iter_loss:

            iter_loss = batch_loss

            

    X, labels = shuffle(X, labels)



    # early stoping

    if iter_loss < best_loss:

        best_loss = iter_loss

        es_count = 0

    else:

        es_count += 1

        

    if es_count > 100:

        break

    

    if i%100==0:

        print('iter: {:2d}, loss: {:.3f}'.format(i, best_loss))



print('iter: {:2d}, best_loss: {:.3f}'.format(i, best_loss))

print('--- training done ---')
# perform prediction

X_test = data_preprocessing(test_df)

y_out = sess.run(output, feed_dict={X_input: X_test, training_mode: False})

predict_class = y_out.argmax(axis=1)+1
sess.close()
output = pd.DataFrame({'Id': test_df.loc[:, 'Id'], 'class': predict_class})

output.to_csv('./output-tf.csv', index=False)