# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here'sseveral helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import csv

# Load training and test set and look at some of the data

trainData = pd.read_csv('../input/train.csv')
testData = pd.read_csv('../input/test.csv')

trainData.head(5)
#First plots: find out how each parameter varies in relation to the survival of the passenger
#Reason: understand which parameters are the most important for the problem

survived = trainData[(trainData.Survived==1)]

fig, axs = plt.subplots(1,2)

print("Economic class")
trainData['Pclass'].value_counts().sort_index().plot.bar(ax=axs[0],figsize=(12, 6), fontsize=16)
survived['Pclass'].value_counts().sort_index().plot.bar(ax=axs[1], fontsize=16)

axs[0].set_title("People on the Titanic", fontsize=20)
axs[1].set_title("People who survived", fontsize=20)
print("Sex")

fig2, axs2 = plt.subplots(1,2)

trainData['Sex'].value_counts().sort_index().plot.bar(ax=axs2[0], figsize=(12, 6), fontsize=16)
survived['Sex'].value_counts().sort_index().plot.bar(ax=axs2[1], fontsize=16)

axs2[0].set_title("People on the Titanic", fontsize=20)
axs2[1].set_title("People who survived", fontsize=20)
print("Age")

fig3, axs3 = plt.subplots(1,2)

trainData['Age'].plot.hist(ax=axs3[0], figsize=(12, 6), fontsize=16)
survived['Age'].plot.hist(ax=axs3[1], fontsize=16)

axs3[0].set_title("People on the Titanic", fontsize=20)
axs3[1].set_title("People who survived", fontsize=20)
print("Number of siblings/spouses aboard")

fig4, axs4 = plt.subplots(1,2)

trainData['SibSp'].value_counts().sort_index().plot.bar(ax=axs4[0], figsize=(12, 6), fontsize=16)
survived['SibSp'].value_counts().sort_index().plot.bar(ax=axs4[1], fontsize=16)

axs4[0].set_title("People on the Titanic", fontsize=20)
axs4[1].set_title("People who survived", fontsize=20)
print("Number of parents/children aboard")

fig5, axs5 = plt.subplots(1,2)

trainData['Parch'].value_counts().sort_index().plot.bar(ax=axs5[0], figsize=(12, 6), fontsize=16)
survived['Parch'].value_counts().sort_index().plot.bar(ax=axs5[1], fontsize=16)

axs5[0].set_title("People on the Titanic", fontsize=20)
axs5[1].set_title("People who survived", fontsize=20)
print("Port of embarkation")

fig6, axs6 = plt.subplots(1,2)

trainData['Embarked'].value_counts().sort_index().plot.bar(ax=axs6[0], figsize=(12, 6),fontsize=16)
survived['Embarked'].value_counts().sort_index().plot.bar(ax=axs6[1], fontsize=16)

axs6[0].set_title("People on the Titanic", fontsize=20)
axs6[1].set_title("People who survived", fontsize=20)
print("Fare")

fig7, axs7 = plt.subplots(1,2)

trainData['Fare'].plot.hist(ax=axs7[0], figsize=(12, 6), fontsize=16)
survived['Fare'].plot.hist(ax=axs7[1], fontsize=16)

axs7[0].set_title("People on the Titanic", fontsize=20)
axs7[1].set_title("People who survived", fontsize=20)
#counting how many entries are in the dataset for each of the variables
print("Number of non-null values in each column in dataset")
print("Survived = ", trainData['Survived'].count())
print("Pclass = ", trainData['Pclass'].count())
print("Sex = ", trainData['Sex'].count())
print("Age = ", trainData['Age'].count())
print("SibSp = ", trainData['SibSp'].count())
print("Parch = ", trainData['Parch'].count())
print("Ticket = ", trainData['Ticket'].count())
print("Fare = ", trainData['Fare'].count())
print("Cabin = ", trainData['Cabin'].count())
print("Embarked = ", trainData['Embarked'].count())
def logisticRegression(X, Y, Xtest, alpha, nEpoch, sh, lamb=0):
#Inputs: training data set, labels, data set to predict, learning rate, number of epochs, number of features in the data, regularization parameter
    
    x = tf.placeholder(tf.float32, [None, sh])
    y = tf.placeholder(tf.float32, [None, 1])
    
    #threshold for logistic regression to decide between 1 and 0 options
    p5 = tf.constant(0.5)

    #create variables
    W = tf.Variable(tf.random_normal([sh, 1], mean=0.0, stddev=0.05))
    b = tf.Variable([0.])

    #start logistic regression calculations, using a gradient descent optimizer
    y_pred = tf.matmul(x, W) + b
    y_pred_sigmoid = tf.sigmoid(y_pred)

    x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_pred, labels=y)
    loss = tf.reduce_mean(x_entropy)
    
    train_step = tf.train.GradientDescentOptimizer(alpha).minimize(loss)
    delta = tf.abs((y - y_pred_sigmoid))
    correct_prediction = tf.cast(tf.less(delta, p5), tf.int32)
    prediction = tf.cast(tf.round(y_pred_sigmoid), tf.int32)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Train for the select number of epochs
    init = tf.initialize_all_variables()
    costs = []
    accs = []

    with tf.Session() as sess:
        sess.run(init)

        print('Training...')
        for i in range(nEpoch):
            fd_train = {x: X, y: Y.reshape((-1, 1))}
            train_step.run(fd_train)
            fd_test = {x: Xtest}
            pred = prediction.eval(fd_test)
        
            if i % 5000 == 0:
                loss_step = loss.eval(fd_train)
                train_accuracy = accuracy.eval(fd_train)
                costs.append(loss_step)
                accs.append(train_accuracy)
                print("cost at epoch ", i, " is: ", loss_step)
    
    #return list of costs, the prediction results and the list of accuracies measured in the training set
    return costs, pred, accs
def neuralNetwork(X, Y, Xtest, alpha, nEpoch, sh):
#Inputs: training data set, labels, data set to predict, learning rate, number of epochs, number of features in the data
    
    tf.reset_default_graph()
    
    x = tf.placeholder(tf.float32, [sh, None])
    y = tf.placeholder(tf.float32, [1, None]) 
    
    #threshold to decide between 1 and 0 options
    p5 = tf.constant(0.5)

    #create variables
    W1 = tf.get_variable("W1", [5, sh], initializer = tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [5, 1], initializer = tf.zeros_initializer())
    
    W2 = tf.get_variable("W2", [2, 5], initializer = tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [2, 1], initializer = tf.zeros_initializer())
    
    W3 = tf.get_variable("W3", [4, 2], initializer = tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [4, 1], initializer = tf.zeros_initializer())

    W4 = tf.get_variable("W4", [1, 4], initializer = tf.contrib.layers.xavier_initializer())
    b4 = tf.get_variable("b4", [1, 1], initializer = tf.zeros_initializer())
    
    #start calculations for the neural network
    Z1 = tf.add(tf.matmul(W1, x), b1)
    A1 = tf.nn.relu(Z1)
    
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    A2 = tf.nn.relu(Z2)
    
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    A3 = tf.nn.relu(Z3)
    
    Z4 = tf.add(tf.matmul(W4, A3), b4)
    
    logits = tf.transpose(Z4)
    labels = tf.transpose(y)
    y_pred_sigmoid = tf.sigmoid(logits)

    x_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(x_entropy)
    
    train_step = tf.train.AdamOptimizer(alpha).minimize(loss)
    delta = tf.abs((y - y_pred_sigmoid))
    correct_prediction = tf.cast(tf.less(delta, p5), tf.int32)
    prediction = tf.cast(tf.round(y_pred_sigmoid), tf.int32)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    #Train for the selected number of epochs
    init = tf.global_variables_initializer()
    costs = []
    accs = []

    with tf.Session() as sess:
        sess.run(init)

        print('Training...')
        for i in range(nEpoch):
            fd_train = {x: X, y: Y.reshape((1, -1))}
            train_step.run(fd_train)
            fd_test = {x: Xtest}
            pred = prediction.eval(fd_test)
            
            if i % 5000 == 0:
                loss_step = loss.eval(fd_train)
                train_accuracy = accuracy.eval(fd_train)
                costs.append(loss_step)
                accs.append(train_accuracy)
                print("Cost at epoch ", i, " is ", loss_step)
                
    #return list of costs, the prediction results and the list of accuracies measured in the training set       
    return costs, pred, accs
#number of training examples
m = trainData['Name'].count()

#create matrices X and Y with the desired features and substitute categorical features by numerical ones
X = trainData[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
X['Sex'] = X['Sex'].replace(to_replace=['male', 'female'], value=[0, 1])

Y = trainData['Survived']

#create matrices X and Y with the desired features and substitute categorical features by numerical ones
Xtest = testData[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
Xidx = testData[['PassengerId']]
Xtest['Sex'] = Xtest['Sex'].replace(to_replace=['male', 'female'], value=[0, 1])
#select only examples that contain information for the age
X_age = X[np.isnan(X['Age'])!=True]
Y_age = Y[np.isnan(X['Age'])!=True]

m = len(Y_age)

Y_age = np.reshape(Y_age, (m, 1))

Xtest_age = Xtest[np.isnan(Xtest['Age'])!=True]
Xidx_age = Xidx[np.isnan(Xtest['Age'])!=True]

#train model with 5 features
costs, predictions, accuracy = logisticRegression(X_age, Y_age, Xtest_age, 0.01, 100000, 5)

print("Model trained!")
#select only the used features from the data
X_noage = X[['Pclass', 'Sex', 'SibSp', 'Parch']]
Y_noage = Y

Xtest_noage = Xtest[np.isnan(Xtest['Age'])]
Xtest_noage = Xtest_noage[['Pclass', 'Sex', 'SibSp', 'Parch']]
Xidx_noage = Xidx[np.isnan(Xtest['Age'])]

#train model with 4 features
costs_na, predictions_na, accuracy_na = logisticRegression(X_noage, Y_noage, Xtest_noage, 0.01, 20000, 4)

print("Model trained!")
#select only examples that contain information for the age
X_age = X[np.isnan(X['Age'])!=True]
Y_age = Y[np.isnan(X['Age'])!=True]

m = len(Y_age)

Y_age = np.reshape(Y_age, (m, 1))

Xtest_age = Xtest[np.isnan(Xtest['Age'])!=True]
Xidx_age = Xidx[np.isnan(Xtest['Age'])!=True]

#train model with 5 features
costs_nn, predictions_nn, accuracy_nn = neuralNetwork(np.transpose(X_age), np.transpose(Y_age), np.transpose(Xtest_age), 0.003, 50000, 5)

print("Model trained!")
#select only the used features from the data
X_noage = X[['Pclass', 'Sex', 'SibSp', 'Parch']]
Y_noage = Y

Xtest_noage = Xtest[np.isnan(Xtest['Age'])]
Xtest_noage = Xtest_noage[['Pclass', 'Sex', 'SibSp', 'Parch']]
Xidx_noage = Xidx[np.isnan(Xtest['Age'])]

#train model with 4 features
costs_na_nn, predictions_na_nn, accuracy_na_nn = neuralNetwork(np.transpose(X_noage), np.transpose(Y_noage), np.transpose(Xtest_noage), 0.003, 20000, 4)

print("Model trained!")
pd.options.mode.chained_assignment = None

#calculate the average fare and age of passengers
fare_avg = np.sum(trainData['Fare'])/trainData['Fare'].count()
age_avg = np.sum(trainData['Age'])/trainData['Age'].count()

#Select the features from the data and substitute strings by integers
X2 = trainData[['Pclass', "Age", "Sex", "SibSp", "Parch", 'Fare', 'Embarked']]
X2['Sex'] = X2['Sex'].replace(to_replace=['male', 'female'], value=[0, 1])
X2['Embarked'] = X2['Embarked'].replace(to_replace=['S', 'C', 'Q'], value=[0, 1, 2])

#if the value for one of these features is not present, substitute by their average
for i in range(0, 891):
    if np.isnan(X2['Fare'][i]):
        X2['Fare'][i] = fare_avg
    if np.isnan(X2['Age'][i]):
        X2['Age'][i] = age_avg
    if np.isnan(X2['Embarked'][i]):
        X2['Embarked'][i] = 0

#same thing done before, but now for the test data set
X2test = testData[['Pclass', "Age", "Sex", "SibSp", "Parch", 'Fare', 'Embarked']]
X2test['Sex'] = X2test['Sex'].replace(to_replace=['male', 'female'], value=[0, 1])
X2test['Embarked'] = X2test['Embarked'].replace(to_replace=['S', 'C', 'Q'], value=[0, 1, 2])

for i in range(0, 418):
    if np.isnan(X2test['Fare'][i]):
        X2test['Fare'][i] = fare_avg
    if np.isnan(X2test['Age'][i]):
        X2test['Age'][i] = age_avg
    if np.isnan(X2test['Embarked'][i]):
        X2test['Embarked'][i] = 0

#train the logistic regression model
costs_agelg, predictions_agelg, accuracy_agelg = logisticRegression(X2, Y, X2test, 0.003, 40000, 7)

print("Logistic regression trained")

#train the neural network model
costs_agenn, predictions_agenn, accuracy_agenn = neuralNetwork(np.transpose(X2), np.transpose(Y), np.transpose(X2test), 0.003, 100000, 7)

print("Neural network has been trained")

#use imputation to fix the data for NaN values
X3 = trainData[['Pclass', "Age", "Sex", "SibSp", "Parch", 'Fare', 'Embarked']]

X3['Sex'] = X3['Sex'].replace(to_replace=['male', 'female'], value=[0, 1])
X3['Embarked'] = X3['Embarked'].replace(to_replace=['S', 'C', 'Q'], value=[0, 1, 2])

from sklearn.preprocessing import Imputer
imp = Imputer()
X3 = imp.fit_transform(X3)

#the same is done for the test set
X3test = testData[['Pclass', "Age", "Sex", "SibSp", "Parch", 'Fare', 'Embarked']]

X3test['Sex'] = X3test['Sex'].replace(to_replace=['male', 'female'], value=[0, 1])
X3test['Embarked'] = X3test['Embarked'].replace(to_replace=['S', 'C', 'Q'], value=[0, 1, 2])

X3test = imp.fit_transform(X3test)

#train only neural network, because it seems to get better predictions
costs_imp, predictions_imp, accuracy_imp = neuralNetwork(np.transpose(X3), np.transpose(Y), np.transpose(X3test), 0.003, 100000, 7)
print("Neural network with imputation is trained!")
print("Logistic Regression with Age: ", costs[-1], " and with no Age: ", costs_na[-1])
print("Neural Network with Age: ", costs_nn[-1], " and with no Age: ", costs_na_nn[-1])
print("More features for Logistic Regression: ", costs_agelg[-1], " and Neural Network: ", costs_agenn[-1])
print("Neural network with imputation: ", costs_imp[-1])

#plot the loss in terms of number of epochs for each set of data
plt.figure(1, figsize=(12, 6))
plt.subplot(121)
plt.plot(costs)
plt.title('Cross Entropy Loss - LR, with age')
plt.xlabel('epoch')
plt.ylabel('loss')

plt.subplot(122)
plt.plot(costs_na)
plt.title('Cross Entropy Loss - LR, without age')
plt.xlabel('epoch')
plt.show()

plt.figure(2, figsize=(12, 6))
plt.subplot(121)
plt.plot(costs_nn)
plt.title('Cross Entropy Loss - NN, with age')
plt.xlabel('epoch')
plt.ylabel('loss')

plt.subplot(122)
plt.plot(costs_na_nn)
plt.title('Cross Entropy Loss - NN, without age')
plt.xlabel('epoch')
plt.ylabel('loss')

plt.figure(3, figsize=(12, 6))
plt.subplot(131)
plt.plot(costs_agelg)
plt.title('Cross Entropy Loss - LG, with more features')
plt.xlabel('epoch')
plt.ylabel('loss')

plt.subplot(132)
plt.plot(costs_agenn)
plt.title('Cross Entropy Loss - NN, with more features')
plt.xlabel('epoch')
plt.ylabel('loss')

plt.subplot(133)
plt.plot(costs_imp)
plt.title('Cross Entropy Loss - NN, imputation')
plt.xlabel('epoch')
plt.ylabel('loss')
#plot the accuracy in terms of number of epochs for each set of data
print("Logistic Regression with Age: ", accuracy[-1], " and with no Age: ", accuracy_na[-1])
print("Neural Network with Age: ", accuracy_nn[-1], " and with no Age: ", accuracy_na_nn[-1])
print("More Features for Logistic Regression: ", accuracy_agelg[-1], " and Neural Network: ", accuracy_agenn[-1])
print("Neural Network with imputation: ", accuracy_imp[-1])

plt.figure(4, figsize=(12, 6))
plt.subplot(121)
plt.plot(accuracy)
plt.title('Model Accuracy - LR, with age')
plt.xlabel('epoch')
plt.ylabel('accuracy')

plt.subplot(122)
plt.plot(accuracy_na)
plt.title('Model Accuracy - LR, without age')
plt.xlabel('epoch')

plt.figure(5, figsize=(12, 6))
plt.subplot(121)
plt.plot(accuracy_nn)
plt.title('Model Accuracy - NN, with age')
plt.xlabel('epoch')
plt.ylabel('accuracy')

plt.subplot(122)
plt.plot(accuracy_na_nn)
plt.title('Model Accuracy - NN, without age')
plt.xlabel('epoch')
plt.ylabel('accuracy')

plt.figure(6, figsize=(12, 6))
plt.subplot(131)
plt.plot(accuracy_agelg)
plt.title('Model Accuracy - LG, with more features')
plt.xlabel('epoch')
plt.ylabel('accuracy')

plt.subplot(132)
plt.plot(accuracy_agenn)
plt.title('Model Accuracy - NN, with more features')
plt.xlabel('epoch')
plt.ylabel('accuracy')

plt.subplot(133)
plt.plot(accuracy_imp)
plt.title('Model accuracy - NN, imputation')
plt.xlabel('epoch')
plt.ylabel('accuracy')

plt.show()
#Build a matrix combining all the results obtained so far and output it to a csv file for each of the models tested
names = testData[['PassengerId']]

result_age = np.concatenate((np.reshape(Xidx_age, (332, 1)),predictions), axis=1)
result_noage = np.concatenate((np.reshape(Xidx_noage, (86, 1)),predictions_na), axis=1)

result = np.concatenate((result_age, result_noage), axis=0)

myfile = open('output_lg.csv','w')

column= ['PassengerId', 'Survived']

wrtr = csv.writer(myfile, delimiter=',')
wrtr.writerow(column)
wrtr.writerows(result)
myfile.close()

result_age = np.concatenate((np.reshape(Xidx_age, (332, 1)),predictions_nn), axis=1)
result_noage = np.concatenate((np.reshape(Xidx_noage, (86, 1)),predictions_na_nn), axis=1)

result_nn = np.concatenate((result_age, result_noage), axis=0)

myfile = open('output_nn.csv','w')

column= ['PassengerId', 'Survived']

wrtr = csv.writer(myfile, delimiter=',')
wrtr.writerow(column)
wrtr.writerows(result_nn)
myfile.close()

result_agelg = np.concatenate((np.reshape(Xidx, (418, 1)),predictions_agelg), axis=1)

myfile = open('output_agelg.csv','w')

column= ['PassengerId', 'Survived']

wrtr = csv.writer(myfile, delimiter=',')
wrtr.writerow(column)
wrtr.writerows(result_agelg)
myfile.close()

result_agenn = np.concatenate((np.reshape(Xidx, (418, 1)),predictions_agenn), axis=1)

myfile = open('output_agenn.csv','w')

column= ['PassengerId', 'Survived']

wrtr = csv.writer(myfile, delimiter=',')
wrtr.writerow(column)
wrtr.writerows(result_agenn)
myfile.close()

result_imp = np.concatenate((np.reshape(Xidx, (418, 1)),predictions_imp), axis=1)

myfile = open('output_imp.csv','w')

column= ['PassengerId', 'Survived']

wrtr = csv.writer(myfile, delimiter=',')
wrtr.writerow(column)
wrtr.writerows(result_imp)
myfile.close()