import pandas as pd
import tensorflow as tf
import numpy as np
data = pd.read_csv('../input/data.csv')
cleaned = data[data.columns[2:]]
cleaned.head(3)
# making labels
# B is benign,so B = 0 
# M is malignant,so M = 1

total_number = len(data.diagnosis)
label = np.zeros([total_number, 1], dtype=np.float64)
for i in range(total_number):
    if data.diagnosis[i]=='M':
        label[i]=1.0

# 0-350 will be used for training         
train_input_data = cleaned.iloc[0:350].values
train_label_data = label[0:350]

# 350-450 will be used for prediction
predict_input_data = cleaned.iloc[450:550].values
predict_label_data = label[450:550]
X = tf.placeholder(tf.float64, shape=[None, 30])
Y = tf.placeholder(tf.float64, shape=[None, 1])
W = tf.Variable(tf.random_uniform([30, 2], -1., 1., dtype=tf.float64), name='W')
b = tf.Variable(tf.zeros([2], dtype = tf.float64), name = 'b')
L = tf.add(tf.matmul(X,W), b)
L = tf.nn.relu(L)
model = tf.nn.softmax(L)
cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(model), axis = 1))
optimizer = tf.train.AdamOptimizer(0.01)
train_op = optimizer.minimize(cost)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
malignant_label = predict_label_data[predict_label_data == 1.]
benign_label = predict_label_data[predict_label_data == 0.]
print("Number of malignant data in predict_label_data : ", len(malignant_label))
print("Number of benign data in predict_label_data : ", len(benign_label))


benign = []
malignant = []
for i in range(100):
    if predict_label_data[i] == 1:
        malignant.append(predict_input_data[i])
    else:
        benign.append(predict_input_data[i])
malignant = np.asarray(malignant)
benign = np.asarray(benign)
malignant_label = np.reshape(malignant_label, (len(malignant_label),1))
benign_label = np.reshape(benign_label, (len(benign_label),1))
for step in range(100):
    sess.run(train_op, feed_dict = {X:train_input_data, Y:train_label_data})
    
    if (step+1)%10 == 0:
        print(step+1, sess.run(cost, feed_dict = {X:train_input_data, Y:train_label_data}))
        
prediction = tf.argmax(model, axis = 1)
target = tf.argmax(Y, axis = 1)

is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("Accuracy : ", sess.run(accuracy*100, feed_dict = {X:predict_input_data, Y:predict_label_data}))
print("Sensitivity : ", sess.run(accuracy*100, feed_dict = {X:malignant, Y:malignant_label}))
print("Specificity : ", sess.run(accuracy*100, feed_dict = {X:benign, Y:benign_label}))
