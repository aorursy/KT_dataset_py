import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))
%matplotlib inline
# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df.columns
f1 = 'Age'
f2 = 'Embarked'
colors = ['red' if l == 0 else 'green' for l in df.Survived]
plt.scatter(df[f1], df[f2], color=colors)
#remove columns I am not going to use
dfy = df.Survived
dfy = pd.get_dummies(dfy, columns=['Survived'])
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'])
df = pd.get_dummies(df, columns=['Pclass', 'Sex', 'Embarked'])
df_test = df_test.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'])
df_test = pd.get_dummies(df_test, columns=['Pclass', 'Sex', 'Embarked'])
df.Age = df.Age.fillna(df.Age.mean())
df_test.Age = df_test.Age.fillna(df_test.Age.mean())
df_test.Fare = df_test.Fare.fillna(df_test.Fare.mean())
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
columns = df.columns

df = pd.DataFrame(sc.fit_transform(df), columns=columns)
df_test = pd.DataFrame(sc.transform(df_test), columns=columns)
#Logistic Regressiontf
W1 = tf.Variable(tf.random_uniform([12, 8], dtype=tf.float32, minval=-0.2, maxval=0.2))
W2 = tf.Variable(tf.random_uniform([8, 2], dtype=tf.float32, minval=-0.2, maxval=0.2))
b1 = tf.Variable(tf.random_uniform([8], dtype=tf.float32, minval=-0.2, maxval=0.2))
b2 = tf.Variable(tf.random_uniform([2], dtype=tf.float32, minval=-0.2, maxval=0.2))

x = tf.placeholder(tf.float32, [None, 12])
y = tf.placeholder(tf.float32, [None, 2])

h1 = tf.maximum(tf.matmul(x, W1) + b1, 0)
#ops
pred = tf.sigmoid(tf.matmul(h1, W2) + b2)
cost = tf.reduce_mean(-y*tf.log(pred) - (1-y)*tf.log(1-pred))
correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
learning_rate = 0.001
training_epochs = 12000
batch_size = 64
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df, dfy, test_size=0.2)
# Start training
with tf.Session() as sess:
    sess.run(init)
    print("Accuracy:", accuracy.eval({x: X_test, y: y_test}))
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(X_train.shape[0]/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs = X_train[i:i+batch_size]
            batch_ys = y_train[i:i+batch_size]
            # Fit training using batch data
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch) % 100 == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost), "Accuracy:", accuracy.eval({x: X_train, y: y_train}))

    print("Optimization Finished!")

    # Test model
    print("Accuracy:", accuracy.eval({x: X_test, y: y_test}))
    final_preds = pred.eval({x: df_test})
custom_pred = np.argmax(final_preds, axis=1)
df_test = pd.read_csv('../input/test.csv')
df_test = df_test[['PassengerId']]
df_test['Survived'] = custom_pred
df_test.to_csv('results.csv', index=False)