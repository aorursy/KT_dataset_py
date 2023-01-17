import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import tensorflow as tf

%matplotlib inline
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')
train.head()
train.describe(include='all')
test.describe(include='all')
train = train.drop(['Cabin','Ticket'],axis = 1)

test = test.drop(['Cabin','Ticket'],axis = 1)
train['Age'] = train['Age'].fillna(np.round(train['Age'].mean()))

test['Age'] = test['Age'].fillna(np.round(test['Age'].mean()))
train.head()
train['Name'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

test['Name'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
train.head()
train.Name = train.Name.replace('Miss','Mrs')

train.Name = train.Name.replace('Ms','Mrs')

train.Name = train.Name.replace('Mme','Mrs')

train.Name = train.Name.replace('Lady','Mrs')

train.Name = train.Name.replace('Dr','Master')

train.Name = train.Name.replace('Sir','Mr')

train.Name = train.Name.replace(['Rev','Mlle','Major','Col','Capt','Don','Dona','Jonkheer','Countess'],'Rare')

test.Name = test.Name.replace('Miss','Mrs')

test.Name = test.Name.replace('Ms','Mrs')

test.Name = test.Name.replace('Mme','Mrs')

test.Name = test.Name.replace('Lady','Mrs')

test.Name = test.Name.replace('Dr','Master')

test.Name = test.Name.replace('Sir','Mr')

test.Name = test.Name.replace(['Rev','Mlle','Major','Col','Capt','Don','Dona','Jonkheer','Countess'],'Rare')
mapping = {'Master':1, 'Mr':2, 'Mrs':3, 'Rare':4}

train.Name = train.Name.map(mapping)

test.Name = test.Name.map(mapping)
mapping = {'male':1, 'female':2}

train.Sex = train.Sex.map(mapping)

test.Sex = test.Sex.map(mapping)
test.Embarked = test.Embarked.fillna(test.Embarked.mode()[0])

train.Embarked = train.Embarked.fillna(train.Embarked.mode()[0])

mapping = {'S':1, 'C':2, 'Q':3}

train.Embarked = train.Embarked.map(mapping)

test.Embarked = test.Embarked.map(mapping)
train.loc[train.Age < 10,'Age'] = 1

train.loc[(train.Age >= 10) & (train.Age < 25),'Age'] = 2

train.loc[(train.Age >= 25) & (train.Age < 35),'Age'] = 3

train.loc[(train.Age >= 35) & (train.Age < 50),'Age'] = 4

train.loc[train.Age >= 50,'Age'] = 5

test.loc[test.Age < 10,'Age'] = 1

test.loc[(test.Age >= 10) & (test.Age < 25),'Age'] = 2

test.loc[(test.Age >= 25) & (test.Age < 35),'Age'] = 3

test.loc[(test.Age >= 35) & (test.Age < 50),'Age'] = 4

test.loc[test.Age >= 50,'Age'] = 5
train.Fare = train.Fare.fillna(train.Fare.median())

test.Fare = test.Fare.fillna(test.Fare.median())

train.loc[train.Fare > 300,'Fare'] = 270

test.loc[test.Fare > 300,'Fare'] = 270
train.loc[train.Fare <= 67,'Fare'] = 1

train.loc[(train.Fare > 67) & (train.Fare <= 135),'Fare'] = 2

train.loc[(train.Fare > 135) & (train.Fare <= 202),'Fare'] = 3

train.loc[(train.Fare > 202) & (train.Fare <= 270),'Fare'] = 4

test.loc[test.Fare <= 67,'Fare'] = 1

test.loc[(test.Fare > 67) & (test.Fare <= 135),'Fare'] = 2

test.loc[(test.Fare > 135) & (test.Fare <= 202),'Fare'] = 3

test.loc[(test.Fare > 202) & (test.Fare <= 270),'Fare'] = 4
train = train.drop(['PassengerId'],axis = 1)

PId = test['PassengerId']

test = test.drop(['PassengerId'],axis = 1)
x_train = train.iloc[:,1:]

y_train = train.iloc[:,0]

y_train = np.expand_dims(y_train, 1)

x_test = test
# Parameters

learning_rate = 0.1



# Network Parameters

n_input = x_train.shape[1]



n_hidden_1 = 32  # 1st layer number of features

n_hidden_2 = 64  # 2nd layer number of features



# placeholders for a tensor that will be always fed.

X = tf.placeholder(tf.float32, shape=[None, n_input])

Y = tf.placeholder(tf.float32, shape=[None, 1])



W = tf.Variable(tf.random_normal([n_input, 1]), name='weight')

b = tf.Variable(tf.random_normal([1]), name='bias')



# Hypothesis using sigmoid: tf.div(1., 1. + tf.exp(tf.matmul(X, W)))

hypothesis = tf.sigmoid(tf.matmul(X, W) + b)



# cost/loss function

cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1 - Y)*tf.log(1 - hypothesis))



optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)



# Accuracy computation

# True if hypothesis>0.5 else False

predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)

accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))
training_epochs = 15

batch_size = 32

display_step = 1

step_size = 1000



# Launch the graph

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())



    # Training cycle

    for epoch in range(training_epochs):

        avg_cost = 0.

        avg_accuracy = 0.

        # Loop over step_size

        for step in range(step_size):

            # Pick an offset within the training data, which has been randomized.

            # Note: we could use better randomization across epochs.

            offset = (step * batch_size) % (y_train.shape[0] - batch_size)

            # Generate a minibatch.

            batch_data = x_train.iloc[offset:(offset + batch_size), :]

            batch_labels = y_train[offset:(offset + batch_size), :]



            # Run optimization op (backprop) and cost op (to get loss value)

            _, c, a = sess.run([optimizer, cost, accuracy], feed_dict={X: batch_data,

                                                          Y: batch_labels})

            avg_cost += c / step_size

            avg_accuracy += a / step_size



        # Display logs per epoch step

        if epoch % display_step == 0:

            print("Epoch:", '%02d' % (epoch + 1), "cost={:.4f}".format(avg_cost), "train accuracy={:.4f}".format(avg_accuracy))

    print("Optimization Finished!")

    

    ## 4. Results (creating submission file)

    

    outputs = sess.run(predicted, feed_dict={X: x_test})

    submission = ['PassengerId,Survived']



    for id, prediction in zip(PId, outputs):

        submission.append('{0},{1}'.format(id, int(prediction)))



    submission = '\n'.join(submission)



    with open('../working/submission.csv', 'w') as outfile:

        outfile.write(submission)
from sklearn.cross_validation import train_test_split

Xtrain, Xtest, ytrain, ytest = train_test_split(train.drop(['Survived'],axis = 1), train['Survived'], test_size = 0.33, random_state = 42)
Xtrain.describe()
from sklearn.ensemble import GradientBoostingClassifier

gbdt = GradientBoostingClassifier()

gbdt.fit(Xtrain, ytrain)
gbdt.score(Xtrain,ytrain)
gbdt.score(Xtest,ytest)
from sklearn.ensemble import GradientBoostingClassifier

gbdt = GradientBoostingClassifier()

gbdt.fit(train.drop(['Survived'],axis = 1), train['Survived'])
sub = pd.DataFrame({'PassengerId': PId, 'Survived':gbdt.predict(test)})
sub.to_csv('submission.csv', index = False)