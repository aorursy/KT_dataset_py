import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix 
import tensorflow as tf

#matplotlib inline
train_org = pd.read_csv('../input/train.csv')
test_org = pd.read_csv('../input/test.csv')
PassengerId = test_org['PassengerId']

train_copy = train_org.copy()
test_copy = test_org.copy()

train_copy['Cabin_Category'] = train_org['Cabin'].apply(lambda x: 0 \
                                                        if type(x) == float else 1).astype(int)

train_copy['Age'] = train_copy['Age'].fillna(value = 999)
train_copy['Age_Category'] = 8
train_copy.loc[train_copy['Age'] <= 10, 'Age_Category'] = 0
train_copy.loc[(train_copy['Age'] > 10) & (train_copy['Age'] <= 20), 'Age_Category'] = 1
train_copy.loc[(train_copy['Age'] > 20) & (train_copy['Age'] <= 30), 'Age_Category'] = 2
train_copy.loc[(train_copy['Age'] > 30) & (train_copy['Age'] <= 40), 'Age_Category'] = 3
train_copy.loc[(train_copy['Age'] > 40) & (train_copy['Age'] <= 50), 'Age_Category'] = 4
train_copy.loc[(train_copy['Age'] > 50) & (train_copy['Age'] <= 60), 'Age_Category'] = 5
train_copy.loc[(train_copy['Age'] > 60) & (train_copy['Age'] <= 70), 'Age_Category'] = 6
train_copy.loc[(train_copy['Age'] > 70) & (train_copy['Age'] <= 80), 'Age_Category'] = 7
train_copy['Age_Category'] = train_copy['Age_Category'].astype(int)

train_copy['SibSp_Category'] = 4
train_copy.loc[train_copy['SibSp'] == 0, 'SibSp_Category'] = 0
train_copy.loc[train_copy['SibSp'] == 1, 'SibSp_Category'] = 1
train_copy.loc[train_copy['SibSp'] == 2, 'SibSp_Category'] = 2
train_copy.loc[train_copy['SibSp'] == 3, 'SibSp_Category'] = 3
train_copy['SibSp_Category'] = train_copy['SibSp_Category'].astype(int)

train_copy['Parch_Category'] = 4
train_copy.loc[train_copy['Parch'] == 0, 'Parch_Category'] = 0
train_copy.loc[train_copy['Parch'] == 1, 'Parch_Category'] = 1
train_copy.loc[train_copy['Parch'] == 2, 'Parch_Category'] = 2
train_copy.loc[train_copy['Parch'] == 3, 'Parch_Category'] = 3
train_copy['Parch_Category'] = train_copy['Parch_Category'].astype(int)

train_copy['Pclass_Category'] = 2
train_copy.loc[train_copy['Pclass'] == 1, 'Pclass_Category'] = 0
train_copy.loc[train_copy['Pclass'] == 2, 'Pclass_Category'] = 1
train_copy['Pclass_Category'] = train_copy['Pclass_Category'].astype(int)

train_copy['Sex_Category'] = 1
train_copy.loc[train_copy['Sex'] == 'female', 'Sex_Category'] = 0
train_copy['Sex_Category'] = train_copy['Sex_Category'].astype(int)

train_copy['Embarked_Category'] = 0
train_copy.loc[train_copy['Embarked'] == 'S', 'Embarked_Category'] = 0
train_copy.loc[train_copy['Embarked'] == 'C', 'Embarked_Category'] = 1
train_copy.loc[train_copy['Embarked'] == 'Q', 'Embarked_Category'] = 2
train_copy['Embarked_Category'] = train_copy['Embarked_Category'].astype(int)

train_copy['Fare_Category'] = 1
train_copy.loc[train_copy['Fare'] <= 7.9104, 'Fare_Category'] = 0
train_copy.loc[(train_copy['Fare'] > 7.9104) & (train_copy['Fare'] <= 14.4542), 'Fare_Category'] = 1
train_copy.loc[(train_copy['Fare'] > 14.4542) & (train_copy['Fare'] <= 31.0), 'Fare_Category'] = 2
train_copy.loc[train_copy['Fare'] > 31.0, 'Fare_Category'] = 3
train_copy['Fare_Category'] = train_copy['Fare_Category'].astype(int)

drop_elements = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', \
                'Fare', 'Cabin', 'Embarked']
train_copy = train_copy.drop(drop_elements, axis = 1)

test_copy['Cabin_Category'] = test_org['Cabin'].apply(lambda x: 0 \
                                                      if type(x) == float else 1).astype(int)

test_copy['Age'] = test_copy['Age'].fillna(value = 999)
test_copy['Age_Category'] = 8
test_copy.loc[test_copy['Age'] <= 10, 'Age_Category'] = 0
test_copy.loc[(test_copy['Age'] > 10) & (test_copy['Age'] <= 20), 'Age_Category'] = 1
test_copy.loc[(test_copy['Age'] > 20) & (test_copy['Age'] <= 30), 'Age_Category'] = 2
test_copy.loc[(test_copy['Age'] > 30) & (test_copy['Age'] <= 40), 'Age_Category'] = 3
test_copy.loc[(test_copy['Age'] > 40) & (test_copy['Age'] <= 50), 'Age_Category'] = 4
test_copy.loc[(test_copy['Age'] > 50) & (test_copy['Age'] <= 60), 'Age_Category'] = 5
test_copy.loc[(test_copy['Age'] > 60) & (test_copy['Age'] <= 70), 'Age_Category'] = 6
test_copy.loc[(test_copy['Age'] > 70) & (test_copy['Age'] <= 80), 'Age_Category'] = 7
test_copy['Age_Category'] = test_copy['Age_Category'].astype(int)

test_copy['SibSp_Category'] = 4
test_copy.loc[test_copy['SibSp'] == 0, 'SibSp_Category'] = 0
test_copy.loc[test_copy['SibSp'] == 1, 'SibSp_Category'] = 1
test_copy.loc[test_copy['SibSp'] == 2, 'SibSp_Category'] = 2
test_copy.loc[test_copy['SibSp'] == 3, 'SibSp_Category'] = 3
test_copy['SibSp_Category'] = test_copy['SibSp_Category'].astype(int)

test_copy['Parch_Category'] = 4
test_copy.loc[test_copy['Parch'] == 0, 'Parch_Category'] = 0
test_copy.loc[test_copy['Parch'] == 1, 'Parch_Category'] = 1
test_copy.loc[test_copy['Parch'] == 2, 'Parch_Category'] = 2
test_copy.loc[test_copy['Parch'] == 3, 'Parch_Category'] = 3
test_copy['Parch_Category'] = test_copy['Parch_Category'].astype(int)

test_copy['Pclass_Category'] = 2
test_copy.loc[test_copy['Pclass'] == 1, 'Pclass_Category'] = 0
test_copy.loc[test_copy['Pclass'] == 2, 'Pclass_Category'] = 1
test_copy['Pclass_Category'] = test_copy['Pclass_Category'].astype(int)

test_copy['Sex_Category'] = 1
test_copy.loc[test_copy['Sex'] == 'female', 'Sex_Category'] = 0
test_copy['Sex_Category'] = test_copy['Sex_Category'].astype(int)

test_copy['Embarked_Category'] = 0
test_copy.loc[test_copy['Embarked'] == 'S', 'Embarked_Category'] = 0
test_copy.loc[test_copy['Embarked'] == 'C', 'Embarked_Category'] = 1
test_copy.loc[test_copy['Embarked'] == 'Q', 'Embarked_Category'] = 2
test_copy['Embarked_Category'] = test_copy['Embarked_Category'].astype(int)

test_copy['Fare_Category'] = 1
test_copy.loc[test_copy['Fare'] <= 7.9104, 'Fare_Category'] = 0
test_copy.loc[(test_copy['Fare'] > 7.9104) & (test_copy['Fare'] <= 14.4542), 'Fare_Category'] = 1
test_copy.loc[(test_copy['Fare'] > 14.4542) & (test_copy['Fare'] <= 31.0), 'Fare_Category'] = 2
test_copy.loc[test_copy['Fare'] > 31.0, 'Fare_Category'] = 3
test_copy['Fare_Category'] = test_copy['Fare_Category'].astype(int)

drop_elements = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', \
                'Fare', 'Cabin', 'Embarked']
test_copy = test_copy.drop(drop_elements, axis = 1)
plt.figure(figsize = (5, 3))
plt.title('Pearson Correlation of Features')
sns.heatmap(train_copy.astype(float).corr(), linewidths = 0.1, linecolor = 'white')
plt.show()
y_data = np.identity(2)[train_copy['Survived']]

x_data1 = np.identity(2)[train_copy['Cabin_Category']]
x_data2 = np.identity(9)[train_copy['Age_Category']]
x_data3 = np.identity(5)[train_copy['SibSp_Category']]
x_data4 = np.identity(5)[train_copy['Parch_Category']]
x_data5 = np.identity(3)[train_copy['Pclass_Category']]
x_data6 = np.identity(2)[train_copy['Sex_Category']]
x_data7 = np.identity(3)[train_copy['Embarked_Category']]
x_data8 = np.identity(4)[train_copy['Fare_Category']]
x_data = np.concatenate([x_data1, x_data2, x_data3, x_data4, x_data5, x_data6,\
                        x_data7, x_data8], axis = 1)

test_data1 = np.identity(2)[test_copy['Cabin_Category']]
test_data2 = np.identity(9)[test_copy['Age_Category']]
test_data3 = np.identity(5)[test_copy['SibSp_Category']]
test_data4 = np.identity(5)[test_copy['Parch_Category']]
test_data5 = np.identity(3)[test_copy['Pclass_Category']]
test_data6 = np.identity(2)[test_copy['Sex_Category']]
test_data7 = np.identity(3)[test_copy['Embarked_Category']]
test_data8 = np.identity(4)[test_copy['Fare_Category']]
test_data = np.concatenate([test_data1, test_data2, test_data3, test_data4, \
                            test_data5, test_data6, test_data7, test_data8], axis = 1)
def weight_variable(name, shape):
    initializer = tf.truncated_normal_initializer(mean = 0.0, stddev = 0.01, dtype = tf.float32)
    return tf.get_variable(name, shape, initializer = initializer)

def bias_variable(name, shape):
    initializer = tf.constant_initializer(value = 0.0, dtype = tf.float32)
    return tf.get_variable(name, shape, initializer = initializer)

def mlp(x, n_in, n_units, n_classes, keep_prob, reuse = False):
    with tf.variable_scope('mlp', reuse = reuse):
        w_1 = weight_variable('w_1', [n_in, n_units])
        b_1 = bias_variable('b_1', [n_units])
        f = tf.matmul(x, w_1) + b_1

        # dropout
        f = tf.nn.dropout(f, keep_prob)

        # relu
        f = tf.nn.relu(f)

        w_2 = weight_variable('w_2', [n_units, n_classes])
        b_2 = bias_variable('b_2', [n_classes])
        f = tf.matmul(f, w_2) + b_2

    return f

def loss_cross_entropy(y, t):
    cross_entropy = - tf.reduce_mean(tf.reduce_sum(t * tf.log(tf.clip_by_value(y, 1e-10, 1.0)), axis = 1))
    return cross_entropy

def accuracy(y, t):
    correct_preds = tf.equal(tf.argmax(y, axis = 1), tf.argmax(t, axis = 1))
    accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
    return accuracy

def training(loss, learning_rate, var_list):
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
    train_step = optimizer.minimize(loss, var_list = var_list)
    return train_step
n_in = 33
n_units = 64
n_classes = 2
learning_rate = 0.01
n_epoch = 3
n_iter = 100
batch_size = 32
show_step = 100
tf.reset_default_graph()

x = tf.placeholder(shape = [None, n_in], dtype = tf.float32)
y = tf.placeholder(shape = [None, n_classes], dtype = tf.float32)
keep_prob = tf.placeholder(shape = [], dtype = tf.float32)

logits = mlp(x, n_in, n_units, n_classes, keep_prob, reuse = False)
probs = tf.nn.softmax(logits)
loss = loss_cross_entropy(probs, y)

var_list = tf.trainable_variables('mlp')
train_step = training(loss, learning_rate, var_list)
acc =  accuracy(probs, y)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for e in range(n_epoch):
        train_indices = np.random.choice(len(x_data), round((len(x_data)*0.8)), \
                                         replace = False)
        test_indices = np.array(list(set(range(len(x_data))) - set(train_indices)))
        x_train = x_data[train_indices]
        x_test = x_data[test_indices]
        y_train = y_data[train_indices]
        y_test = y_data[test_indices]

        history_loss_train = []
        history_loss_test = []
        history_acc_train = []
        history_acc_test = []

        for i in range(n_iter):
            # Training
            rand_index = np.random.choice(len(x_train), size = batch_size)
            x_batch = x_train[rand_index]
            y_batch = y_train[rand_index]

            feed_dict = {x: x_batch, y: y_batch, keep_prob: 0.7}
            sess.run(train_step, feed_dict = feed_dict)

            temp_loss = sess.run(loss, feed_dict = feed_dict)
            temp_acc = sess.run(acc, feed_dict = feed_dict)

            history_loss_train.append(temp_loss)
            history_acc_train.append(temp_acc)

            if (i + 1) % show_step == 0:
                print ('-' * 100)
                print ('Epoch: ' + str(e + 1) + ' Iteration: ' + str(i + 1) + \
                       '  Loss: ' + str(temp_loss) + '  Accuracy: ' + str(temp_acc))

            # Test
            rand_index = np.random.choice(len(x_test), size = batch_size)
            x_batch = x_test[rand_index]
            y_batch = y_test[rand_index]

            feed_dict = {x: x_batch, y: y_batch, keep_prob: 1.0}
            temp_loss = sess.run(loss, feed_dict = feed_dict)
            temp_acc = sess.run(acc, feed_dict = feed_dict)

            history_loss_test.append(temp_loss)
            history_acc_test.append(temp_acc)

        print ('-' * 100)    
        fig = plt.figure(figsize = (10, 3))
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(range(n_iter), history_loss_train, 'b-', label = 'Training')
        ax1.plot(range(n_iter), history_loss_test, 'r-', label = 'Test')
        ax1.set_title('Loss')
        ax1.legend(loc = 'upper right')
        ax1.grid(True)

        ax2 = fig.add_subplot(1, 2, 2)
        ax2.plot(range(n_iter), history_acc_train, 'b-', label = 'Training')
        ax2.plot(range(n_iter), history_acc_test, 'r-', label = 'Test')
        ax2.set_ylim(0.0, 1.0)
        ax2.set_title('Accuracy')
        ax2.legend(loc = 'lower right')
        ax2.grid(True)

        plt.show()
        
    feed_dict = {x: x_data, keep_prob: 1.0}
    y_train_pred = sess.run(probs, feed_dict = feed_dict)
    feed_dict = {x: test_data, keep_prob: 1.0}
    y_test_pred = sess.run(probs, feed_dict = feed_dict)
acc_train = accuracy_score(np.argmax(y_data, axis = 1), np.argmax(y_train_pred, axis = 1))
f1_train = f1_score(np.argmax(y_data, axis = 1), np.argmax(y_train_pred, axis = 1))
confmat_train = confusion_matrix(np.argmax(y_data, axis = 1), np.argmax(y_train_pred, axis = 1))

print ('Accuracy:', acc_train)
print ('f1:', f1_train)
plt.figure(figsize = (5, 3))
sns.heatmap(confmat_train, linewidths = 0.1, linecolor = 'white', \
            annot = True, fmt = 'd')
plt.show()
submission = pd.DataFrame({'PassengerId': PassengerId,
                           'Survived': np.argmax(y_test_pred, axis = 1)})
submission = pd.DataFrame({'PassengerId': PassengerId,
                           'Survived': np.argmax(y_test_pred, axis = 1)})
submission.to_csv('submission.csv', index=False)