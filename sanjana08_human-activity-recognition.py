import sys

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from sklearn.metrics import classification_report, confusion_matrix



from keras.models import Sequential

from keras.layers import Dense, Dropout

from keras.optimizers import Adam



import tensorflow.compat.v1 as tf

tf.disable_v2_behavior() 



pd.set_option('display.max_columns', None)

pd.set_option('display.max_rows',None)
train = pd.read_csv("../input/human-activity-recognition-with-smartphones/train.csv")

test = pd.read_csv("../input/human-activity-recognition-with-smartphones/test.csv")
train.shape, test.shape
train.head()
train.describe().transpose()
print('Total number of missing values in train : ', train.isna().values.sum())

print('Total number of missing values in test : ', test.isna().values.sum())
train['subject'].unique()
train['Activity'].unique()
pd.crosstab(train.subject, train.Activity, margins=True)
chart = sns.countplot(x=train['Activity'])

chart.set_xticklabels(chart.get_xticklabels(), rotation=45, horizontalalignment='right')
train.Activity.value_counts()
facetgrid = sns.FacetGrid(train, hue='Activity', height=5, aspect=3)

facetgrid.map(sns.distplot,'tBodyAccMag-mean()', hist=False).add_legend()
facetgrid = sns.FacetGrid(train, hue='Activity', height=5,aspect=3)

facetgrid.map(sns.distplot,'tBodyGyroMag-mean()', hist=False).add_legend()
fig = plt.figure(figsize=(32,24))

ax1 = fig.add_subplot(221)

ax1 = sns.stripplot(x='Activity', y='tBodyAcc-max()-X', data=train.loc[train['subject']==1], jitter=True)

ax2 = fig.add_subplot(222)

ax2 = sns.stripplot(x='Activity', y='tBodyAcc-max()-Y', data=train.loc[train['subject']==1], jitter=True)

plt.show()
plt.figure(figsize=(10,7))

sns.boxplot(x='Activity', y='angle(X,gravityMean)', data=train)

plt.ylabel("Angle between X-axis and gravityMean")

plt.title('Box plot of angle(X,gravityMean) column across various activities')

plt.xticks(rotation = 90)
plt.figure(figsize=(10,7))

sns.boxplot(x='Activity', y='angle(Y,gravityMean)', data = train)

plt.ylabel("Angle between Y-axis and gravityMean")

plt.title('Box plot of angle(Y,gravityMean) column across various activities')

plt.xticks(rotation = 90)
train.drop(['subject'], axis=1, inplace=True)

test.drop(['subject'], axis=1, inplace=True)
X_train = train.iloc[:,:-1]

y_train = train.iloc[:,-1]



X_test = test.iloc[:,:-1]

y_test = test.iloc[:,-1]
encoder = LabelEncoder()

y_train = encoder.fit_transform(y_train)

y_train = pd.get_dummies(y_train).values



y_test = encoder.fit_transform(y_test)

y_test = pd.get_dummies(y_test).values
X_train.shape, y_train.shape, X_test.shape, y_test.shape
scaler=MinMaxScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.transform(X_test)
model = Sequential()

model.add(Dense(units=64, input_dim=X_train.shape[1], activation='relu'))

model.add(Dense(units=128, activation='relu'))

model.add(Dense(units=32, activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(units=6, activation='softmax'))



model.compile(optimizer=Adam(lr=0.001), metrics=['accuracy'], loss='categorical_crossentropy')

print(model.summary())
history=model.fit(X_train, y_train, batch_size=256, epochs=20, validation_data=(X_test, y_test), shuffle=True)
plt.plot(history.history['acc'])

plt.plot(history.history['val_acc'])

plt.title('Model Accuracy')

plt.xlabel('Epochs')

plt.ylabel('Accuracy')

plt.legend(['train', 'test'], loc='upper left')

plt.show
plt.plot(history.history['loss'])

plt.plot(history.history['val_loss'])

plt.title('Model Loss')

plt.xlabel('Epochs')

plt.ylabel('Loss')

plt.legend(['train', 'test'], loc='upper left')

plt.show
y_pred = model.predict(X_test)
pred = np.argmax(y_pred,axis = 1) 

y_actual = np.argmax(y_test,axis = 1)
confusion_matrix(y_actual, pred)
print(classification_report(y_actual, pred))
results = pd.DataFrame({'Actual':y_actual, 'Predicted':pred})

results.iloc[:75,:]
tf.reset_default_graph()
n_input = 561

n_hidden1 = 64

n_hidden2 = 128

n_hidden3 = 32

n_output = 6
x = tf.placeholder(tf.float32, shape=[None, 561])

y = tf.placeholder(tf.float32, shape=[None, 6])

hold_prob = tf.placeholder(tf.float32)
def init_weights(shape):

    w = tf.truncated_normal(shape=shape, stddev=0.1)

    return tf.Variable(w)
def init_bias(shape):

    b = tf.constant(0.1, shape=shape)

    return tf.Variable(b)
def next_batch(j, batch_size):

    x = X_train[j:j+batch_size]

    y = y_train[j:j+batch_size]

    j = (j+batch_size)%len(X_train)

    return x,y,j
hidden1 = {'weights':init_weights([n_input,n_hidden1]), 'bias':init_bias([n_hidden1])}

hidden2 = {'weights':init_weights([n_hidden1, n_hidden2]), 'bias':init_bias([n_hidden2])}

hidden3 = {'weights':init_weights([n_hidden2, n_hidden3]), 'bias':init_bias([n_hidden3])}

output = {'weights':init_weights([n_hidden3, n_output]), 'bias':init_bias([n_output])}
def feed_forward(x):

    h1 = tf.add(tf.matmul(x,hidden1['weights']), hidden1['bias'])

    h1 = tf.nn.relu(h1)

    

    h2 = tf.add(tf.matmul(h1,hidden2['weights']), hidden2['bias'])

    h2 = tf.nn.relu(h2)

    

    h3 = tf.add(tf.matmul(h2,hidden3['weights']), hidden3['bias'])

    h3 = tf.nn.relu(h3)

    

    dropout = tf.nn.dropout(h3, hold_prob)

    

    out = tf.matmul(dropout,output['weights']) +output['bias']

    out = tf.nn.softmax(out)

    

    return out
y_pred = feed_forward(x)
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_pred))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

train = optimizer.minimize(cross_entropy)
init = tf.global_variables_initializer()
epochs = 5000

j = 0

batch_size = 256

train_acc = []

test_acc = []

with tf.Session() as sess:

    sess.run(init)

    for i in range(epochs):

        x_batch, y_batch, j = next_batch(j, batch_size)

        sess.run(train, feed_dict={x:x_batch, y:y_batch, hold_prob:0.5})

        

        if (i % 100 == 0 and i != 0):

            print('Epoch', i, 'completed out of', epochs)

            correct = tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1))

            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

            trainacc = sess.run(accuracy, feed_dict = {x: x_batch, y: y_batch, hold_prob:0.5})

            print('Train set Accuracy:', trainacc)

            train_acc.append(trainacc)

            testacc = sess.run(accuracy, feed_dict = {x: X_test, y: y_test, hold_prob:1.0})

            print('Test set Accuracy:', testacc)

            test_acc.append(testacc)

            print()
plt.plot(train_acc)

plt.plot(test_acc)

plt.legend(['train','test'])