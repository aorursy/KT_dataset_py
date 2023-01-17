import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from sklearn import preprocessing
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
#data_df = train_df.append(test_df)
print(train_df.columns.values)
print(test_df.columns.values)
train_df.head()
test_df.head()
train_df[['Age']].sort_values(by='Age', ascending=False).head()
train_df[['Age']].sort_values(by='Age', ascending=False).tail()
train_mod_df = train_df.copy()

for i in range(10):
    train_mod_df.loc[ (train_mod_df['Age'] >= i*10) & (train_mod_df['Age'] < (i+1)*10), 'AgeId'] = i
    
for i in range(50):
    train_mod_df.loc[ (train_mod_df['Fare'] >= i*7.91*3) & (train_mod_df['Fare'] < (i+1)*7.91*3), 'FareId'] = i

    train_mod_df['Cabin'] = train_mod_df['Cabin'].fillna('NAN')
    train_mod_df.loc[train_mod_df['Cabin'].str.contains('A'), 'CabinId'] = 1
    train_mod_df.loc[train_mod_df['Cabin'].str.contains('B'), 'CabinId'] = 2
    train_mod_df.loc[train_mod_df['Cabin'].str.contains('C'), 'CabinId'] = 3
    train_mod_df.loc[train_mod_df['Cabin'].str.contains('D'), 'CabinId'] = 4
    train_mod_df.loc[train_mod_df['Cabin'].str.contains('E'), 'CabinId'] = 5
    train_mod_df.loc[train_mod_df['Cabin'].str.contains('F'), 'CabinId'] = 6
    train_mod_df.loc[train_mod_df['Cabin'].str.contains('G'), 'CabinId'] = 7
    train_mod_df.loc[train_mod_df['Cabin'].str.contains('T'), 'CabinId'] = 8
    train_mod_df.loc[train_mod_df['Cabin'].str.contains('NAN'), 'CabinId'] = 0
    
    train_mod_df['Embarked'] = train_mod_df['Embarked'].fillna('NAN')
    train_mod_df.loc[train_mod_df['Embarked'].str.contains('NAN'), 'EmbarkedId'] = 0
    train_mod_df.loc[train_mod_df['Embarked'].str.contains('C'), 'EmbarkedId'] = 1
    train_mod_df.loc[train_mod_df['Embarked'].str.contains('S'), 'EmbarkedId'] = 2
    train_mod_df.loc[train_mod_df['Embarked'].str.contains('Q'), 'EmbarkedId'] = 3
    
    train_mod_df['NotAlone'] = train_mod_df['SibSp'] | train_mod_df['Parch']
    
train_mod_df.tail()
train_mod_df[['Sex', 'Pclass', 'CabinId', 'Survived']].groupby(['Sex', 'Pclass', 'CabinId'], as_index=True).mean()
xtrain = train_mod_df.drop("PassengerId", axis=1).copy()
xtrain = xtrain.drop('Age', axis=1)
xtrain = xtrain.drop('Name', axis=1)
xtrain = xtrain.drop('Ticket', axis=1)
xtrain = xtrain.drop('Fare', axis=1)
xtrain = xtrain.drop('Cabin', axis=1)
xtrain = xtrain.drop('Embarked', axis=1)
xtrain = xtrain.drop('SibSp', axis=1)
xtrain = xtrain.drop('Parch', axis=1)
xtrain['AgeId'] = xtrain['AgeId'].fillna(3.5) #temp
xtrain['FareId'] = xtrain['FareId'].fillna(0) #temp
xtrain.head()
xtrainmale = xtrain[xtrain.Sex == 'male'].copy()
xtrainmale = xtrainmale.drop('Sex', axis=1)
passenger_count = len(xtrainmale.index)
ytrainmale = xtrainmale['Survived'].copy()
xtrainmale = xtrainmale.drop('Survived', axis=1)

xtrainfemale = xtrain[xtrain.Sex == 'female'].copy()
xtrainfemale = xtrainfemale.drop('Sex', axis=1)
ytrainfemale = xtrainfemale['Survived'].copy()
xtrainfemale = xtrainfemale.drop('Survived', axis=1)

xtrainmale.tail()
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(xtrainmale)
xtrainmale = pd.DataFrame(np_scaled)

np_scaled = min_max_scaler.fit_transform(xtrainfemale)
xtrainfemale = pd.DataFrame(np_scaled)
xtrainfemale.tail()
tf.reset_default_graph()
passenger_count = len(xtrainmale.index)
hidden_nodes_count = 5

X = tf.placeholder(shape=(passenger_count,6), dtype=tf.float64, name='X')
Y = tf.placeholder(shape=(passenger_count,), dtype=tf.float64, name='Y')

weights = {
    'hidden1': tf.Variable(tf.random_uniform([6, hidden_nodes_count],minval=-1.0,maxval=1.0,dtype=tf.float64)),
    'hidden2': tf.Variable(tf.random_uniform([hidden_nodes_count, hidden_nodes_count],minval=-1.0,maxval=1.0,dtype=tf.float64)),
    'output': tf.Variable(tf.random_uniform([hidden_nodes_count, 1],minval=-1.0,maxval=1.0,dtype=tf.float64))
}

biases = {
    'hidden1': tf.Variable(tf.zeros([hidden_nodes_count],dtype=tf.float64)),
    'hidden2': tf.Variable(tf.zeros([hidden_nodes_count],dtype=tf.float64)),
    'output': tf.Variable(tf.zeros([1],dtype=tf.float64))
}

A2 = tf.nn.relu(tf.add(tf.matmul(X, weights['hidden1']), biases['hidden1']))
A3 = tf.nn.relu(tf.add(tf.matmul(A2, weights['hidden2']), biases['hidden2']))
Hypothesis = tf.sigmoid(tf.add(tf.matmul(A3, weights['output']), biases['output']))

cost = tf.reduce_sum(tf.square(Hypothesis - Y))

train_step = tf.train.AdamOptimizer(learning_rate = 0.000009).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={X: xtrainmale, Y: ytrainmale})
    weights1male = sess.run(weights['hidden1'])
    weights2male = sess.run(weights['hidden2'])
    weights3male = sess.run(weights['output'])
    biases1male = sess.run(biases['hidden1'])
    biases2male = sess.run(biases['hidden2'])
    biases3male = sess.run(biases['output'])
        
sess.close()

print(weights1male)
tf.reset_default_graph()
passenger_count = len(xtrainfemale.index)
hidden_nodes_count = 5

X = tf.placeholder(shape=(passenger_count,6), dtype=tf.float64, name='X')
Y = tf.placeholder(shape=(passenger_count,), dtype=tf.float64, name='Y')

weights = {
    'hidden1': tf.Variable(tf.random_uniform([6, hidden_nodes_count],minval=-1.0,maxval=1.0,dtype=tf.float64)),
    'hidden2': tf.Variable(tf.random_uniform([hidden_nodes_count, hidden_nodes_count],minval=-1.0,maxval=1.0,dtype=tf.float64)),
    'output': tf.Variable(tf.random_uniform([hidden_nodes_count, 1],minval=-1.0,maxval=1.0,dtype=tf.float64))
}

biases = {
    'hidden1': tf.Variable(tf.zeros([hidden_nodes_count],dtype=tf.float64)),
    'hidden2': tf.Variable(tf.zeros([hidden_nodes_count],dtype=tf.float64)),
    'output': tf.Variable(tf.zeros([1],dtype=tf.float64))
}

A2 = tf.nn.relu(tf.add(tf.matmul(X, weights['hidden1']), biases['hidden1']))
A3 = tf.nn.relu(tf.add(tf.matmul(A2, weights['hidden2']), biases['hidden2']))
Hypothesis = tf.sigmoid(tf.add(tf.matmul(A3, weights['output']), biases['output']))

cost = tf.reduce_sum(tf.square(Hypothesis - Y))

train_step = tf.train.AdamOptimizer(learning_rate = 0.00007).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
    sess.run(train_step, feed_dict={X: xtrainfemale, Y: ytrainfemale})
    weights1female = sess.run(weights['hidden1'])
    weights2female = sess.run(weights['hidden2'])
    weights3female = sess.run(weights['output'])
    biases1female = sess.run(biases['hidden1'])
    biases2female = sess.run(biases['hidden2'])
    biases3female = sess.run(biases['output'])
        
sess.close()

print(weights1female)
test_mod_df = test_df.copy()

for i in range(10):
    test_mod_df.loc[ (test_mod_df['Age'] >= i*10) & (test_mod_df['Age'] < (i+1)*10), 'AgeId'] = i
    
for i in range(50):
    test_mod_df.loc[ (test_mod_df['Fare'] >= i*7.91*3) & (test_mod_df['Fare'] < (i+1)*7.91*3), 'FareId'] = i

    test_mod_df['Cabin'] = test_mod_df['Cabin'].fillna('NAN')
    test_mod_df.loc[test_mod_df['Cabin'].str.contains('A'), 'CabinId'] = 1
    test_mod_df.loc[test_mod_df['Cabin'].str.contains('B'), 'CabinId'] = 2
    test_mod_df.loc[test_mod_df['Cabin'].str.contains('C'), 'CabinId'] = 3
    test_mod_df.loc[test_mod_df['Cabin'].str.contains('D'), 'CabinId'] = 4
    test_mod_df.loc[test_mod_df['Cabin'].str.contains('E'), 'CabinId'] = 5
    test_mod_df.loc[test_mod_df['Cabin'].str.contains('F'), 'CabinId'] = 6
    test_mod_df.loc[test_mod_df['Cabin'].str.contains('G'), 'CabinId'] = 7
    test_mod_df.loc[test_mod_df['Cabin'].str.contains('T'), 'CabinId'] = 8
    test_mod_df.loc[test_mod_df['Cabin'].str.contains('NAN'), 'CabinId'] = 0
    
    test_mod_df['Embarked'] = train_mod_df['Embarked'].fillna('NAN')
    test_mod_df.loc[train_mod_df['Embarked'].str.contains('NAN'), 'EmbarkedId'] = 0
    test_mod_df.loc[train_mod_df['Embarked'].str.contains('C'), 'EmbarkedId'] = 1
    test_mod_df.loc[train_mod_df['Embarked'].str.contains('S'), 'EmbarkedId'] = 2
    test_mod_df.loc[train_mod_df['Embarked'].str.contains('Q'), 'EmbarkedId'] = 3
    
    test_mod_df['NotAlone'] = test_mod_df['SibSp'] | test_mod_df['Parch']
    
test_mod_df.head()
xtestpassenger = test_mod_df.drop("Age", axis=1).copy()
xtestpassenger = xtestpassenger.drop('Name', axis=1)
xtestpassenger = xtestpassenger.drop('Ticket', axis=1)
xtestpassenger = xtestpassenger.drop('Fare', axis=1)
xtestpassenger = xtestpassenger.drop('Cabin', axis=1)
xtestpassenger = xtestpassenger.drop('Embarked', axis=1)
xtestpassenger = xtestpassenger.drop('SibSp', axis=1)
xtestpassenger = xtestpassenger.drop('Parch', axis=1)
xtestpassenger['AgeId'] = xtestpassenger['AgeId'].fillna(3.5) #temp
xtestpassenger['FareId'] = xtestpassenger['FareId'].fillna(0) #temp
xtestpassenger.head()
xtestmalepassenger = xtestpassenger[xtestpassenger.Sex == 'male'].copy()
xtestmalepassenger = xtestmalepassenger.drop('Sex', axis=1)
xtestmale = xtestmalepassenger.drop('PassengerId', axis=1)
xtestmalepassenger = xtestmalepassenger['PassengerId']

xtestfemalepassenger = xtestpassenger[xtestpassenger.Sex == 'female'].copy()
xtestfemalepassenger = xtestfemalepassenger.drop('Sex', axis=1)
xtestfemale = xtestfemalepassenger.drop('PassengerId', axis=1)
xtestfemalepassenger = xtestfemalepassenger['PassengerId']
np_scaled = min_max_scaler.fit_transform(xtestmale)
xtestmale = pd.DataFrame(np_scaled)

np_scaled = min_max_scaler.fit_transform(xtestfemale)
xtestfemale = pd.DataFrame(np_scaled)

xtestmale.tail()
tf.reset_default_graph()
passenger_count = len(xtestmale.index)

X = tf.placeholder(shape=(passenger_count,6), dtype=tf.float64, name='X')
Y = tf.placeholder(shape=(passenger_count,), dtype=tf.float64, name='Y')

W1 = tf.Variable(weights1male)
W2 = tf.Variable(weights2male)
W3 = tf.Variable(weights3male)

A1 = tf.nn.relu(tf.matmul(X, W1)+biases1male)
A2 = tf.nn.relu(tf.matmul(A1, W2)+biases2male)
y_est = tf.sigmoid(tf.matmul(A2, W3)+biases3male)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
y_est_male = sess.run(y_est, feed_dict={X: xtestmale})
        
sess.close()

print(y_est_male)
tf.reset_default_graph()
passenger_count = len(xtestfemale.index)

X = tf.placeholder(shape=(passenger_count,6), dtype=tf.float64, name='X')
Y = tf.placeholder(shape=(passenger_count,), dtype=tf.float64, name='Y')

W1 = tf.Variable(weights1female)
W2 = tf.Variable(weights2female)
W3 = tf.Variable(weights3female)
A1 = tf.nn.relu(tf.matmul(X, W1)+biases1female)
A2 = tf.nn.relu(tf.matmul(A1, W2)+biases2female)
y_est = tf.sigmoid(tf.matmul(A2, W3)+biases3female)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
y_est_female = sess.run(y_est, feed_dict={X: xtestfemale})
        
sess.close()

print(y_est_female)
xtestmalepassenger = pd.DataFrame(xtestmalepassenger)
y_est_male = pd.DataFrame(y_est_male)
outputmale = xtestmalepassenger.copy()
outputmale.index = range(len(outputmale.index)) #Reset indexes in DataFrame - y_est_male was skipped
outputmale['Survived'] = y_est_male

xtestfemalepassenger = pd.DataFrame(xtestfemalepassenger)
y_est_female = pd.DataFrame(y_est_female)
outputfemale = xtestfemalepassenger.copy()
outputfemale.index = range(len(outputfemale.index)) #Reset indexes in DataFrame
outputfemale['Survived'] = y_est_female

output = outputmale.append(outputfemale, ignore_index=True)
output = output.sort_values(by='PassengerId', ascending=True)#Sort data to make it similar to input data.
output.index = range(len(output.index))
output.head() #Print final result
output.loc[output['Survived'] <= 0.5, 'Survived'] = 0
output.loc[output['Survived'] > 0.5, 'Survived'] = 1
output = output.astype(int)
output.head()
outname = 'submission.csv'
output.to_csv(outname, index=False)