# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib as mpl

import tensorflow as tf
from sklearn import preprocessing

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
train_data.head(10)
train_data.info()
train_data.describe()
df_s = train_data[train_data.Survived==1].groupby('Pclass').size()
df_d = train_data[train_data.Survived==0].groupby('Pclass').size()

x_idx = [1,2,3]

fig = plt.figure(figsize=(12,6))

b1 = plt.bar(x_idx, df_d, width=0.5, color='red')
b2 = plt.bar(x_idx, df_s, width=0.5, bottom=df_d, color='green')

plt.title('Survivors vs. Pclass')
plt.xticks(x_idx, ('1st Class', '2nd Class', '3rd Class'))
plt.xlabel('Pclass')
plt.legend((b1[0], b2[1]), ('Not survived', 'Survived'))


plt.show()
df = train_data[['Survived','Sex']].groupby(['Sex', 'Survived']).Sex.count().unstack()

fig, ax = plt.subplots(figsize=(10,6))
cmap = mpl.colors.ListedColormap(['red', 'green'])

df.plot(kind='Bar', stacked=True, title='Survived vs. Gender', ax=ax, colormap=cmap, width=0.5)
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16,8))

emb_surv = train_data[['Survived','Embarked']]
emb_surv = emb_surv.groupby(['Embarked','Survived']).Embarked.count().unstack()
emb_surv.plot(kind='Bar', stacked=True, ax=ax[0], colormap=cmap, width=0.5, title='Survived vs Embarked')

emb_pclass = train_data[['Pclass','Embarked']]
emb_pclass = emb_pclass.groupby(['Embarked','Pclass']).Pclass.count().unstack()
emb_pclass.plot(kind='Bar', stacked=True, ax=ax[1], title='Embarked vs Pclass')
fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=(12,6))
fig.subplots_adjust(wspace=.05)

df_survived = train_data[(train_data.Survived==1) & (train_data.Age > 0)].Age;
df_deads = train_data[(train_data.Survived==0) & (train_data.Age > 0)].Age;

ax[0].hist(df_survived, bins=10, range=(0,100))
ax[0].set_title('Histogram of Age (survived)')
ax[1].hist(df_deads, bins=10, range=(0,100))
ax[1].set_title('Histogram of Age (not survived)')

plt.show()
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=3, sharey=True, figsize=(16,8))
fig.subplots_adjust(wspace=.05, hspace=.3)

df_1c = train_data[train_data.Pclass == 1]
df_2c = train_data[train_data.Pclass == 2]
df_3c = train_data[train_data.Pclass == 3]

ax1[0].hist(df_1c[df_1c.Survived == 1].Age, bins=10, range=(0,100))
ax1[0].set_title('1st Class (survived)')
ax1[0].set_ylabel('Amount')
ax1[1].hist(df_2c[df_2c.Survived == 1].Age, bins=10, range=(0,100))
ax1[1].set_title('2nd Class (survived)')
ax1[2].hist(df_3c[df_3c.Survived == 1].Age, bins=10, range=(0,100))
ax1[2].set_title('3rd Class (survived)')

ax2[0].hist(df_1c[df_1c.Survived == 0].Age, bins=10, range=(0,100))
ax2[0].set_title('1st Class (not survived)')
ax2[0].set_xlabel('Age')
ax2[0].set_ylabel('Amount')
ax2[1].hist(df_2c[df_2c.Survived == 0].Age, bins=10, range=(0,100))
ax2[1].set_title('2nd Class (not survived)')
ax2[1].set_xlabel('Age')
ax2[2].hist(df_3c[df_3c.Survived == 0].Age, bins=10, range=(0,100))
ax2[2].set_title('3rd Class (not survived)')
ax2[2].set_xlabel('Age')

plt.show()
#train_data['Age'].hist(by=train_data['Pclass'], bins=20)
fig, ax = plt.subplots(nrows=1, ncols=2, sharex=False, sharey=True, figsize=(12,6))
fig.subplots_adjust(wspace=.05)

df_survived = train_data[train_data.Survived==1].Fare;
df_deads = train_data[train_data.Survived==0].Fare;

ax[0].hist(df_survived, bins=10)
ax[0].set_title('Histogram of Fare (survived)')
ax[1].hist(df_deads, bins=10)
ax[1].set_title('Histogram of Fare (not survived)')

plt.show()
fig, ax = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(16,5))
fig.subplots_adjust(wspace=0.05)

#cherbourg
ax[0].hist(train_data[train_data.Embarked == 'C'].Fare, bins=10)
ax[0].set_title('Cherbourg')
ax[0].set_xlabel('Fare')

# queenstown
ax[1].hist(train_data[train_data.Embarked == 'Q'].Fare, bins=5)
ax[1].set_title('Queenstown')
ax[1].set_xlabel('Fare')

#southampton
ax[2].hist(train_data[train_data.Embarked == 'S'].Fare, bins=10)
ax[2].set_title('Southampton')
ax[2].set_xlabel('Fare')

plt.show()

df_emb = train_data.groupby(['Embarked']).agg(['size', 'sum'])
df_emb['Fare'].head()
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=2, figsize=(16,10), sharey=True)
fig.subplots_adjust(wspace=0.05, hspace=0.2)

df_surv = train_data[train_data.Survived == 1]
df_dead = train_data[train_data.Survived == 0]

# SibSp Survived
ax1[0].hist(df_surv.SibSp, bins=10, range=(0,8))
ax1[0].set_title('SibSb Survived')

# SibSp not Survived
ax1[1].hist(df_dead.SibSp, bins=10, range=(0,8))
ax1[1].set_title('SibSb Not Survived')

# Parch Survived
ax2[0].hist(df_surv.Parch, bins=10, range=(0,6))
ax2[0].set_title('Parch Survived')

# Parch not Survived
ax2[1].hist(df_dead.Parch, bins=10, range=(0,6))
ax2[1].set_title('Parch Not Survived')
train_set = train_data[['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'SibSp', 'Parch']]

train_set.info()
train_set.head(10)
sex_map = {'male':1, 'female':2}
emb_map = {'S':1, 'C':2, 'Q':3}
min_max = preprocessing.MinMaxScaler()

def normMinMax(df_input):
    aux = df_input.values.reshape(-1,1)
    aux_norm = min_max.fit_transform(aux)
    return pd.DataFrame(aux_norm)

def dataPreparation(input):

    features = input.copy()
    
    # mappings
    features['Sex'] = features['Sex'].map(sex_map)
    features['Embarked'] = features['Embarked'].map(emb_map)
    
    features.loc[features.Embarked.isnull(), 'Embarked'] = emb_map['S']

    # normalization
    print("Mean age: ", round(features.Age.mean()))
    features.loc[features.Age.isnull(), 'Age'] = features.Age.mean()
    features['Age'] = normMinMax(features['Age'])

    print("Mean fare: ", features.Fare.mean())
    features.loc[features.Fare.isnull(), 'Fare'] = features.Fare.mean()    
    features['Fare'] = normMinMax(features['Fare'])
    
    # normalize SibSp and Parch data
    features['SibSp'] = normMinMax(features['SibSp'])
    features['Parch'] = normMinMax(features['Parch'])
    
    features['has_SibSp'] = (features['SibSp'] > 0).astype(int)
    features['has_Parch'] = (features['Parch'] > 0).astype(int)
    
    return features
train_features = dataPreparation(train_set)

train_labels = pd.DataFrame(train_features.pop('Survived'))

train_features.head(10)
train_features.info()
# global parameters
epochs = 10000
learning_rate=0.00001
# building the neural network
tf.reset_default_graph()

y = tf.placeholder(tf.float32, shape=[None, 1], name='y')
x = tf.placeholder(tf.float32, shape=[None, train_features.shape[1]], name='x')

#x_tensor = tf.contrib.layers.fully_connected(x, 2**5)

x_tensor = tf.contrib.layers.fully_connected(x, 2**6)

x_tensor = tf.contrib.layers.fully_connected(x_tensor, 2**6)
x_tensor = tf.contrib.layers.fully_connected(x_tensor, 2**6)
x_tensor = tf.layers.dropout(x_tensor)
x_tensor = tf.contrib.layers.fully_connected(x_tensor, 2**6)
x_tensor = tf.contrib.layers.fully_connected(x_tensor, 2**6)
x_tensor = tf.layers.dropout(x_tensor)
x_tensor = tf.contrib.layers.fully_connected(x_tensor, 2**6)

yhat= tf.contrib.layers.fully_connected(x_tensor, 1, activation_fn=tf.nn.sigmoid)

yhat= tf.identity(yhat, name='logits')

# usual logistic regression cost
cost = tf.reduce_mean( -y*tf.log(yhat)  - (1-y)*tf.log(1-yhat))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(cost)

# we presume survived if prediction says that a passenger has more than 50% chance of survival
prediction = (yhat> 0.5)

# accuracy
correct_pred = tf.equal(prediction, (y > 0.5))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')
df_loss = []
df_acc = []

# training the neural network
sess = tf.Session()
# Initializing the variables
sess.run(tf.global_variables_initializer())

for epoch in range(epochs):
    sess.run(train, feed_dict={x: train_features, y: train_labels})

    loss = sess.run(cost, feed_dict={
        x: train_features,
        y: train_labels})

    valid_acc = sess.run(accuracy, feed_dict={
        x: train_features,
        y: train_labels})

    df_loss.append(loss)
    df_acc.append(valid_acc)

    if ((epoch+1) % 1000 == 0):
        print('Epoch {:>5}, Loss: {:>10.4f} Accuracy: {:.6f}'.format(epoch+1, loss, valid_acc))
        
df_loss = pd.DataFrame(df_loss)
df_acc = pd.DataFrame(df_acc)
fig, ax = plt.subplots(1, 2, figsize=(16,6))

ax[0].plot(df_loss)
ax[0].set_title('Loss')

ax[1].plot(df_acc)
ax[1].set_title('Accuracy')
test_data.head()
test_set = test_data[['Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'SibSp', 'Parch']]
test_set.head(10)
test_features = dataPreparation(test_set)

test_features.head()
test_features.info()
# predict the test set
predicted = pd.DataFrame(sess.run(prediction, feed_dict={x:test_features}))
predicted.columns = ['Survived']

submission = pd.DataFrame({'PassengerId' : test_data['PassengerId'], 'Survived' : predicted['Survived'].astype(int)})

submission.info()

submission.head(10)
submission.to_csv('results.csv', index=False)