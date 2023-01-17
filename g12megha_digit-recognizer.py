import numpy as np

import pandas as pd

import tensorflow as tf
images = pd.read_csv('train.csv')
import matplotlib.pyplot as plt

%matplotlib inline

plt.imshow(images.iloc[0][1:].values.reshape([28,28]))
X = images[images.columns[1:]].as_matrix()

Y = np.zeros([len(images),10])

for i in range(len(images)):

  Y[i][images.iloc[i]['label']] = 1

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.33, random_state = 42)

y_train.shape
X_test.shape
samples = images.count()

batch_size = 100



epoch = 1

learning_rate = .0005



input_size = len(images.columns)-1

hiddenLayer1_size = 256

hiddenLayer2_size = 256

output_size = 10
x = tf.placeholder(tf.float32, shape=[None, input_size])

y = tf.placeholder(tf.float32, shape=[None, output_size])



weights = {

   'h1': tf.Variable(tf.random_normal([input_size, hiddenLayer1_size])),

   'h2': tf.Variable(tf.random_normal([hiddenLayer1_size, hiddenLayer2_size])),

   'out': tf.Variable(tf.random_normal([hiddenLayer2_size, output_size]))

}

biases = {

   'h1': tf.Variable(tf.random_normal([hiddenLayer1_size])),

   'h2': tf.Variable(tf.random_normal([hiddenLayer2_size])),

   'out': tf.Variable(tf.random_normal([output_size]))

}
def createDeepNetwork(x,weights, biases):

   l1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['h1']))

   l2 = tf.nn.relu(tf.add(tf.matmul(l1, weights['h2']), biases['h2']))

   outlayer = tf.add(tf.matmul(l2, weights['out']), biases['out'])

   return outlayer

predictions = createDeepNetwork(x,weights,biases)
cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions,labels=y))

optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
init = tf.global_variables_initializer()

session = tf.InteractiveSession()

session.run(init)
len(X)
batch_size = 100

for i in range(60):

    total_batch = int(len(X_train)/batch_size)

    avg_cost = 0.0

    for bid in range(total_batch):

        batch_x = X_train[bid*batch_size:(bid+1)*batch_size]

        batch_y = y_train[bid*batch_size:(bid+1)*batch_size]

        _,ct = session.run([optimiser,cost],feed_dict={x:batch_x,y:batch_y})

        #print("Cost . " , ct)

       

        avg_cost += ct / total_batch

        #print("avg cost ", avg_cost)

    print("average cost for the round " , avg_cost)

correct_prediction = tf.equal(tf.arg_max(predictions,1),tf.arg_max(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
print(session.run(accuracy, feed_dict={x:X_test, y:y_test}))
test_data = pd.read_csv("test.csv")

Xtest = test_data.as_matrix()
test_pred = session.run(predictions,feed_dict={x:Xtest})
outValue = session.run(tf.arg_max(test_pred,1))
outValue
output = pd.DataFrame(outValue, columns=["Label"],index=range(1,28001))

output.index.names = ['ImageId']

output.to_csv('output.csv')