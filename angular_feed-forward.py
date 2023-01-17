import csv
import tensorflow as tf
import pandas as reader
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

tf = tf.compat.v1
tf.disable_v2_behavior()

train = reader.read_csv("../input/digit-recognizer/train.csv")

# Reading Test Data Set

inputData = train.drop(columns=['label'])
test = reader.read_csv("../input/digit-recognizer/test.csv")
x_test=test.values
totalrows=test.shape[0]
# Setting output
result = train['label'].values
encoder = LabelEncoder()
encoder.fit(result)
result = encoder.transform(result)



def hot_encoder(label):
    total=len(label)
    unique=len(np.unique(label))
    one_hot = np.zeros((total,unique))
    one_hot[np.arange(total),label]=1
    return one_hot

Y=hot_encoder(result)
print("Y",Y.shape)

learning_rate = 0.045
noOfClasses = 10
training_cycles = 400
dim = inputData.shape[1]
hiddenlayer1 = 100
hiddenlayer2 = 100

x = tf.placeholder(tf.float32,[None, dim])
weight = tf.Variable(tf.ones([dim,noOfClasses]))
theeta = tf.Variable(tf.ones([noOfClasses]))
y = tf.compat.v1.placeholder(tf.float32, [None,noOfClasses])
def feedforward(x,weights,thetas):
    layer1 = tf.add(tf.matmul(x, weights['h1']), thetas['b1'])
    layer1 = tf.nn.sigmoid(layer1)
    layer2 = tf.add(tf.matmul(layer1, weights['h2']),thetas['b1'])
    layer2 = tf.nn.relu(layer2)
    output = tf.matmul(layer2, weights['output'])+thetas['output']

    return output


weights = {
    'h1': tf.Variable(tf.truncated_normal([dim, hiddenlayer1])),
    'h2': tf.Variable(tf.truncated_normal([hiddenlayer1, hiddenlayer2])),
    'output': tf.Variable(tf.truncated_normal([hiddenlayer2, noOfClasses]))
}
thetas = {
    'b1': tf.Variable(tf.truncated_normal([hiddenlayer1])),
    'b2': tf.Variable(tf.truncated_normal([hiddenlayer2])),
    'output': tf.Variable(tf.truncated_normal([noOfClasses]))
}

saver=tf.train.Saver()
neural = feedforward(x,weights,thetas)
error = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=neural,labels=y))
optimize = tf.train.AdamOptimizer(learning_rate).minimize(error)
inputTrain, inputTest, resultTrain, resultTest = train_test_split(inputData, Y, test_size=0.2, random_state=1,
                                                                  stratify=Y)
init = tf.global_variables_initializer()
session = tf.Session()
session.run(init)
predict = 0
if predict == 1:
 saver.restore(session,'model/feedforward3')
accuracy_coll=[]
mse_hist=[]
accuracy = 0
cost_history=0
steps=0
accracy_hist = []
cycles = []
if predict !=1:
    for i in range(training_cycles):
        steps=steps+1

        session.run(fetches=[optimize], feed_dict={
                x: inputTrain,
                y: resultTrain
            })
          
        correct_prediction=tf.equal(tf.argmax(neural,1),tf.argmax(y,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        cycles.append(i)
          
        accuracy=(session.run(accuracy,feed_dict={x:inputTest,y:resultTest}))
         
        print("iteration :",steps ,"Accuracy",accuracy)
        accracy_hist.append(accuracy)


print("Training Complete")
print(cycles,accracy_hist)
plt.plot(cycles,accracy_hist)
plt.xlabel("Training Cycles")
plt.ylabel("Accuracy")
plt.title("Feed Forward Neural Network")
plt.show()
res = [[]]

imageid = 1
if predict == 1:
    finalprediction = tf.argmax(neural,1)
    for i in range(totalrows):

        prediction = session.run(finalprediction, feed_dict={x: x_test[i].reshape(1,784)})
        no = prediction[0]
        print(no)
        res.append([imageid, no])
        imageid = imageid+1

    with open('feedforwardoutputtest3.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(res)
if False:
    save_path=saver.save(session,'model/feedforward3')
    print(save_path)



















