# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior() 
df = pd.read_csv("../input/titanic/train.csv")

df.head()

df_test = pd.read_csv("../input/titanic/test.csv")
#Prepare data and drop unused columns 

df = df.drop(columns=['PassengerId','Name', 'Ticket', 'Cabin', 'Fare'])

df = df.fillna({'Age': df['Age'].mean()})    #fill NA age values with average age 

df['Sex'] = df["Sex"].astype('category')

df['Sex'] = df['Sex'].cat.codes

df['Embarked'] = df['Embarked'].astype('category')

df['Embarked'] = df["Embarked"].cat.codes



#One hot encoding of survival

df = pd.concat([df,pd.get_dummies(df['Survived'], prefix='Survived')],axis=1)

df.drop(columns=['Survived'])



#Repeat for test data 

df_test = df_test.drop(columns=['PassengerId','Name', 'Ticket', 'Cabin', 'Fare'])

df_test = df_test.fillna({'Age': df['Age'].mean()})

df_test['Sex'] = df_test["Sex"].astype('category')

df_test['Sex'] = df_test['Sex'].cat.codes

df_test['Embarked'] = df_test['Embarked'].astype('category')

df_test['Embarked'] = df_test["Embarked"].cat.codes
y_data = df[['Survived_0', 'Survived_1']]

X_data = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked']]

X_data_test = df_test



#normalize feature data

X_data_test=X_data_test.apply(lambda x: x/x.max(), axis=0)

X_data=X_data.apply(lambda x: x/x.max(), axis=0)
from sklearn.model_selection import train_test_split

trainX, testX, trainY, testY = train_test_split(X_data, y_data, test_size=0.3)
numFeatures = trainX.shape[1]

numLabels = trainY.shape[1]



X = tf.placeholder(tf.float32, [None, numFeatures])

y = tf.placeholder(tf.float32, [None, numLabels])



weights = tf.Variable(tf.random_normal([numFeatures,numLabels],

                                       mean=0,

                                       stddev=0.01,

                                       name="weights"))



bias = tf.Variable(tf.random_normal([1,numLabels],

                                    mean=0,

                                    stddev=0.01,

                                    name="bias"))
apply_weights_OP = tf.matmul(X, weights, name="apply_weights")

add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias") 

activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")

cost_OP = tf.nn.l2_loss(activation_OP-y, name="squared_error_cost")



learningRate = tf.train.exponential_decay(learning_rate=0.0008,

                                          global_step= 1,

                                          decay_steps=trainX.shape[0],

                                          decay_rate= 0.95,

                                          staircase=True)



training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP)
numEpochs = 50000

with tf.Session() as sess:



    init_OP = tf.global_variables_initializer()

    sess.run(init_OP)

    correct_predictions_OP = tf.equal(tf.argmax(activation_OP,1),tf.argmax(y,1))



    accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))



    # Initialize reporting variables

    cost = 0

    diff = 1

    epoch_values = []

    accuracy_values = []

    cost_values = []



    # Training epochs

    for i in range(numEpochs):

        if i > 1 and diff < .0001:

            print("change in cost %g; convergence."%diff)

            break

        else:

            # Run training step

            step = sess.run(training_OP, feed_dict={X: trainX, y: trainY})

            # Report every 500 epochs

            if i % 500 == 0:

                epoch_values.append(i)

                train_accuracy, newCost = sess.run([accuracy_OP, cost_OP], feed_dict={X: trainX, y: trainY})

                accuracy_values.append(train_accuracy)

                cost_values.append(newCost)

                diff = abs(newCost - cost)

                cost = newCost



                #generate print statements

                print("step %d, training accuracy %g, cost %g, change in cost %g"%(i, train_accuracy, newCost, diff))



    #Final 

    print("final accuracy on test set: %s" %str(sess.run(accuracy_OP, 

                                                         feed_dict={X: testX, 

                                                                    y: testY})))

    #Predictions for submission 

    predict_OP = tf.argmax(activation_OP, axis=1)

    result = sess.run(predict_OP, feed_dict={X: X_data_test})
res_df = pd.DataFrame(result)

res_df.columns = ['Survived']

res_df.index.names = ['PassengerId']

res_df.index +=892
res_df.to_csv('submission.csv')