# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import tensorflow as tf

from sklearn.model_selection import train_test_split
#Read input data

df_train = pd.read_csv("../input/train.csv")

df_train.head(5)
# shape (#of rows, columns) - Data is very less but still will use Neural Network

df_train.shape

# Drop column that it seems to me not required

df_train = df_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis= 1)
#check unique values in Embarked column

df_train["Embarked"].unique()
df_train_1 = df_train

df_train.head(5)
df_train = df_train_1

# One hot encoding for categorical 'Embarked' column (Embarked_S, Embarked_C, Embarked_Q) and add it to the existing dataframe

#df_train["Embarked"].isna().sum()

df_train = pd.concat([df_train, pd.get_dummies(df_train['Embarked'], prefix = 'Embarked')], axis = 1)

df_train.head(5)

#drop original Embarked column

df_train = df_train.drop(['Embarked'], axis= 1)

df_train.head(5)
df_train.head(5)
#Similarly one hot encode "Sex" column (Sex_male, Sex_female )

# FYI: Not doing label encoding because label encode generally works well if the categorical value is ordinal in nature

df_train = pd.concat([df_train, pd.get_dummies(df_train['Sex'], prefix= 'Sex')], axis= 1)

#drop original "Sex" column

df_train = df_train.drop(['Sex'], axis = 1)

df_train.head(5)
# Whether to Normalize or standardize the Age/Fare value 

# Normalize(MinMax) = (x - min(x)) / (max(x) - min(x))

# standardize = (x - mean)/ sd



ageMin = df_train['Age'].min()

ageMax = df_train['Age'].max()

df_train['Age'] = (df_train['Age'] - ageMin)/(ageMax - ageMin)

df_train.head(5)
#Fare normalize

fareMin = df_train['Fare'].min()

fareMax = df_train['Fare'].max()

df_train['Fare'] = (df_train['Fare'] - fareMin)/(fareMax - fareMin)

df_train.head(5)
#Check the latest data in dataframe

df_train.info()
#Count of Missing value in Age

print(df_train['Age'].isnull().sum())



#Replace it with mean()

df_train['Age'] = df_train['Age'].fillna(df_train['Age'].mean())

print(df_train['Age'].isnull().sum())
#Split train test data with 80:20 ratio

y_train_array = df_train["Survived"].values

X_train_array = df_train.drop(["Survived"], axis= 1).values

print("Training data:",X_train_array.shape)

print("Test data:", y_train_array.shape)

X_train, X_test, y_train, y_test = train_test_split(X_train_array, y_train_array, test_size = 0.2)

print('After splitting train and test data:')

print(X_train.shape)

print(X_test.shape)

print(y_train.shape)

print(y_test.shape)
#Tensor flow model with 2 hidden layer

#Linear(Input) -> Relu(L1) -> Linear -> Relu(L2) -> [Softmax(Output layer)]
tf.reset_default_graph() 
inputSize = tf.placeholder(tf.float32, shape = (None, X_train.shape[1]), name= 'inputSize')

outputLabel = tf.placeholder(tf.float32, shape= (None, 2), name= "outputLabel")

#1st layer - Using Dropout regularization technique

# Linear -> Relu

tf.set_random_seed(1)



hiddenLayerSize_1 = 128

inputLayerSize_1 = X_train.shape[1]



#W1 = tf.Variable(tf.random_normal([hiddenLayerSize_1, inputLayerSize_1], stddev=0.01), name='W1')

#b1 = tf.Variable(tf.constant(0.0, shape=(hiddenLayerSize_1, 1)), name='b1')

W1 = tf.get_variable("W1", [hiddenLayerSize_1,inputLayerSize_1], initializer= tf.contrib.layers.xavier_initializer(seed = 1))

b1 = tf.get_variable("b1", [hiddenLayerSize_1,1], initializer= tf.zeros_initializer())

y1 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(W1, tf.transpose(inputSize)), b1)), keep_prob=0.7

                  )





#2nd layer - Using Dropout regularization technique

# Linear -> Relu

hiddenLayerSize_2 = 256

tf.set_random_seed(1)

#W2 = tf.Variable(tf.random_normal([hiddenLayerSize_2, hiddenLayerSize_1], stddev=0.01), name='W2')

#b2 = tf.Variable(tf.constant(0.0, shape=(hiddenLayerSize_2, 1)), name='b2')

W2 = tf.get_variable("W2", [hiddenLayerSize_2, hiddenLayerSize_1], initializer= tf.contrib.layers.xavier_initializer(seed = 1))

b2 = tf.get_variable("b2", [hiddenLayerSize_2,1], initializer= tf.zeros_initializer())

y2 = tf.nn.dropout(tf.nn.relu(tf.add(tf.matmul(W2, y1), b2)), keep_prob=0.7)
#Output layer (Softmax)

tf.set_random_seed(1)

#Wo = tf.Variable(tf.random_normal([2, hiddenLayerSize_2], stddev=0.01), name='Wo')

#bo = tf.Variable(tf.random_normal([2, 1]), name='bo')

Wo = tf.get_variable("Wo", [2, hiddenLayerSize_2], initializer= tf.contrib.layers.xavier_initializer(seed = 1))

bo = tf.get_variable("bo", [2,1], initializer= tf.zeros_initializer())

yo = tf.transpose(tf.add(tf.matmul(Wo, y2), bo))
 # Using Cross entrpy loss with Gradient Descent Optimizer OR Adam Optimizer

learningRate = tf.placeholder(tf.float32, shape=(), name='learningRate')

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=yo, labels=outputLabel))





optimizer = tf.train.GradientDescentOptimizer(learningRate).minimize(loss)

#optimizer = tf.train.AdamOptimizer(learningRate).minimize(loss)
#Predict

predict = tf.nn.softmax(yo)

predictlabel = tf.argmax(predict, 1)

#predictlabel

correctPrediction = tf.equal(tf.argmax(predict, 1), tf.argmax(outputLabel, 1))

accuracy = tf.reduce_mean(tf.cast(correctPrediction, "float"))

accuracy
#2 label Survived and Not Survived

labelTrain = (np.arange(2) == y_train[:,None]).astype(np.float32)

labelTest = (np.arange(2) == y_test[:,None]).astype(np.float32)
#Start Tensor flow and train model

init = tf.global_variables_initializer()



sess = tf.Session()

sess.run(init)
#tf.reset_default_graph() 

#train model

#For 50 epoch

for learnRate in [0.05, 0.01, 0.005]:

    for epoch in range(100):

        epoch_cost = 0.0



        # For each epoch i.e go through all the samples.

        for i in range(X_train.shape[0]):

            _, cost = sess.run([optimizer, loss], feed_dict={learningRate:learnRate, 

                                                          inputSize: X_train[i, None],

                                                          outputLabel: labelTrain[i, None]})

            epoch_cost += cost

        epoch_cost /= X_train.shape[0]    



        # Print the cost in this epcho

        if epoch % 10 == 0:

            print("Epoch: {:3d}    Train Cost: {:.4f}".format(epoch, epoch_cost))
acc_train = accuracy.eval(session= sess, feed_dict={inputSize: X_train, outputLabel: labelTrain})

print("Train accuracy: {:3.2f}%".format(acc_train*100.0))



acc_test = accuracy.eval(session= sess,feed_dict={inputSize: X_test, outputLabel: labelTest})

print("Test accuracy:  {:3.2f}%".format(acc_test*100.0))
#Predict on test data

df_test = pd.read_csv('../input/test.csv')

df_test.head(5)
#Pre-processing step

# Drop column that it seems to me not required

df_test = df_test.drop(['Name', 'Ticket', 'Cabin'], axis= 1)



# One hot encoding for categorical 'Embarked' column (Embarked_S, Embarked_C, Embarked_Q) and add it to the existing dataframe

#df_test["Embarked"].isna().sum()

df_test = pd.concat([df_test, pd.get_dummies(df_test['Embarked'], prefix = 'Embarked')], axis = 1)



#drop original Embarked column

df_test = df_test.drop(['Embarked'], axis= 1)



df_test = pd.concat([df_test, pd.get_dummies(df_test['Sex'], prefix= 'Sex')], axis= 1)

#drop original "Sex" column

df_test = df_test.drop(['Sex'], axis = 1)



ageMin = df_test['Age'].min()

ageMax = df_test['Age'].max()

df_test['Age'] = (df_test['Age'] - ageMin)/(ageMax - ageMin)



fareMin = df_test['Fare'].min()

fareMax = df_test['Fare'].max()

df_test['Fare'] = (df_test['Fare'] - fareMin)/(fareMax - fareMin)



X_test = df_test.drop('PassengerId', axis=1).values
# Predict test output

for i in range(X_test.shape[0]):

    df_test.loc[i, 'Survived'] = sess.run(predictlabel, feed_dict={inputSize: X_test[i, None]}).squeeze()
output = pd.DataFrame()

output['PassengerId'] = df_test['PassengerId']

output['Survived'] = df_test['Survived'].astype(int)

output.to_csv('./prediction.csv', index=False)

output.head()
