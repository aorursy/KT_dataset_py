# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Regexp
import re
mapTitle = {}

def getData(inputData):
    inputSex = inputData['Sex'].values.tolist()
    inputSex = [0 if i == 'female' else 1 for i in inputSex]
    
    inputAge = inputData['Age'].values.tolist()
    inputName = inputData['Name'].values.tolist()
    
#     for i in range(len(inputName)):
#         m = re.search('.+, ([^.]+).*', inputName[i])
#         vTitle = m.group(1)
        
#         nAge = inputAge[i]
#         if math.isnan(nAge) == False:
#             if (vTitle in mapTitle) == False:
#                 mapTitle[vTitle] = []
#             mapTitle[vTitle].append(nAge)
# #     print(mapTitle)
    
#     hostAge = {}
#     for k in mapTitle.keys():
#         hostAge[k] = int(np.mean(mapTitle[k]))
# #     print(hostAge)
    hostAge = {'Mr': 32, 'Mrs': 36, 'Miss': 21, 'Master': 5, 'Don': 40, 'Rev': 41, 'Dr': 43, 'Mme': 24, 'Ms': 28, 'Major': 48, 'Lady': 48, 'Sir': 49, 'Mlle': 24, 'Col': 54, 'Capt': 70, 'the Countess': 33, 'Jonkheer': 38, 'Dona': 39}

    for i in range(len(inputAge)):
        vAge = inputAge[i]
        vName = inputName[i]
        if math.isnan(vAge):
            m = re.search('.+, ([^.]+).*', vName)
            vTitle = m.group(1)
            vAge = hostAge[vTitle]
            inputAge[i] = vAge
    print(inputAge)
    
    # Pclass
    inputPclass = inputData['Pclass'].values.tolist()

    # Fare
    inputFare = inputData['Fare'].values.tolist()
#     for i in range(len(inputFare)):
#         vFare = inputFare[i]
#         if math.isnan(vFare) :
#             vFare = 0
        
#         nFare = int(vFare / 10)
#         inputFare[i] = nFare

    inputEmbarked = inputData['Embarked'].values.tolist()    
    for i in range(len(inputEmbarked)):
        vEmbarked = inputEmbarked[i]
        if vEmbarked == 'S':
            nEmbarked = 3
        elif vEmbarked == 'C':
            nEmbarked = 2
        else:
            nEmbarked = 1
        inputEmbarked[i] = nEmbarked

#     inputTicket = inputData['Ticket'].values.tolist()
#     for i in range(len(inputTicket)):
#         vTicket = inputTicket[i]
#         vTicket = vTicket[:1]
#         if vTicket.isalpha():
#             nTicket = ord(vTicket) - ord('A') + 10
#         else:
#             nTicket = int(vTicket)
#         inputTicket[i] = nTicket

#     inputSibSp = inputData['SibSp'].values.tolist()
#     inputParch = inputData['Parch'].values.tolist()
#     inputFamily = []
#     for i in range(len(inputSibSp)):
#         nSibSp = int(inputSibSp[i])
#         nParch = int(inputParch[i])
#         inputFamily.append(nSibSp + nParch)
    
    result = np.vstack((inputSex, inputPclass, inputFare, inputEmbarked)).T # 78947
    result = np.vstack((inputSex, inputPclass, inputFare, inputEmbarked, inputAge)).T 
    
#     result = np.vstack((inputPclass, inputFare, inputEmbarked)).T
#     result = np.vstack((inputSex, inputPclass, inputFare, inputEmbarked, inputTicket)).T
    
#     result = np.vstack((inputSex, inputAge, inputPclass, inputFare, inputEmbarked)).T
#     result = np.vstack((inputSex, inputAge, inputPclass, inputFare, inputEmbarked, inputSibsp, inputParch)).T
    print(result)
    return result

inputSize = 5
train = pd.read_csv('../input/train.csv')

x_data = getData(train)
# print(x_data)

survived_data = train["Survived"].values.tolist()
for i in range(len(survived_data)):
    if survived_data[i] == 0:
        survived_data[i] = [1, 0]
    else:
        survived_data[i] = [0, 1]

X = tf.placeholder(tf.float32, [None, inputSize])
Y = tf.placeholder(tf.float32, [None, 2])

W1 = tf.Variable(tf.random_normal([inputSize, 10], stddev=0.01))
b1 = tf.Variable(tf.zeros([10]))
L1 = tf.nn.sigmoid(tf.add(tf.matmul(X, W1), b1))

W2 = tf.Variable(tf.random_normal([10, 10], stddev=0.01))
b2 = tf.Variable(tf.zeros([10]))
L2 = tf.nn.sigmoid(tf.add(tf.matmul(L1, W2), b2))

W3 = tf.Variable(tf.random_normal([10, 2], stddev=0.01))
b3 = tf.Variable(tf.zeros([2]))
model = tf.add(tf.matmul(L2, W3), b3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
optimizer = tf.train.AdamOptimizer(0.04).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# batch_size = 100
# total_batch = int(len(sex_data) / batch_size)
# x_data = age_data

# print(x_data)
fig, ax = plt.subplots()
# Create violinplot
ax.plot(range(len(x_data)), x_data)

# Show the plot
plt.show()
# Any results you write to the current directory are saved as output.
# print(x_data)
for epoch in range(5000):
    total_cost = 0

    _, cost_val = sess.run([optimizer, cost], feed_dict={X: x_data, Y: survived_data})

    if epoch % 10 == 0:
        print(cost_val)


#     
#     for i in range(total_batch):
#         batch_xs, batch_ys = 

# prediction = tf.argmax(model, 1)
# is_correct = tf.equal(tf.argmax(model, 1), target)
# print(sess.run(X))
# print(survived_data)
# print(sess.run(prediction,feed_dict={X: age_data}))
# print(sess.run(model, feed_dict={X: age_data}))
# accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

# print(sess.run(accuracy, feed_dict={X: x_data, Y: survived_data}))
# print(sess.run(tf.argmax(model, 1, None, None, tf.float32), feed_dict={X: age_data}))
# Y = tf.placeholder(tf.float32, [None, 10])
# target = tf.argmax(Y, 1)
# print(sess.run(target, feed_dict={Y: survived_data}))
# input [1,2,3,4,5]
# ouput 0 

# Y = tf.placeholder(tf.float32, [None, 1])
# target = tf.argmax(Y, 1)
# print(sess.run(target, feed_dict={Y: survived_data}))
# [[1],[2],[111]]
test = pd.read_csv('../input/test.csv')
test_data = getData(test)

result = sess.run(tf.argmax(model, 1), feed_dict = {X: test_data})

df_submission = pd.DataFrame({
   "PassengerId": test["PassengerId"],
   "Survived": result
})
df_submission.to_csv('./result1.csv', index=False)
# print(result)
print('done')