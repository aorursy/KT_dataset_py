# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import tensorflow as tf

import time

import math

from sklearn import metrics

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
train_data = pd.read_csv('../input/train.csv')

test_data = pd.read_csv('../input/test.csv')

print(train_data.shape)

print(test_data.shape)
train_data.head(3)
full_data = [train_data, test_data]



# Feature that tells whether a passenger had a cabin on the Titanic

train_data['Has_Cabin'] = train_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)

test_data['Has_Cabin'] = test_data["Cabin"].apply(lambda x: 0 if type(x) == float else 1)



# Feature engineering steps taken from Sina

# Create new feature FamilySize as a combination of SibSp and Parch

for dataset in full_data:

    dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

# Create new feature IsAlone from FamilySize

for dataset in full_data:

    dataset['IsAlone'] = 0

    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

# Remove all NULLS in the Embarked column

for dataset in full_data:

    dataset['Embarked'] = dataset['Embarked'].fillna('S')

# Remove all NULLS in the Fare column and create a new feature CategoricalFare

for dataset in full_data:

    dataset['Fare'] = dataset['Fare'].fillna(train_data['Fare'].median())

train_data['CategoricalFare'] = pd.qcut(train_data['Fare'], 4,duplicates='drop')

# Create a New feature CategoricalAge

for dataset in full_data:

    age_avg = dataset['Age'].mean()

    age_std = dataset['Age'].std()

    age_null_count = dataset['Age'].isnull().sum()

    age_null_random_list = np.random.randint(age_avg - age_std, age_avg + age_std, size=age_null_count)

    dataset['Age'][np.isnan(dataset['Age'])] = age_null_random_list

    dataset['Age'] = dataset['Age'].astype(int)

train_data['CategoricalAge'] = pd.cut(train_data['Age'], 5)

# Define function to extract titles from passenger names

def get_title(name):

    title_search = re.search(' ([A-Za-z]+)\.', name)

    # If the title exists, extract and return it.

    if title_search:

        return title_search.group(1)

    return ""

# Create a new feature Title, containing the titles of passenger names

for dataset in full_data:

    dataset['Title'] = dataset['Name'].apply(get_title)

# Group all non-common titles into one single grouping "Rare"

for dataset in full_data:

    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')

    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')



for dataset in full_data:

    # Mapping Sex

    dataset['Sex'] = dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

    # Mapping titles

    title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}

    dataset['Title'] = dataset['Title'].map(title_mapping)

    dataset['Title'] = dataset['Title'].fillna(0)

    # Mapping Embarked

    dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

    # Mapping Fare

    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare']                               = 0

    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1

    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2

    dataset.loc[ dataset['Fare'] > 31, 'Fare']                                  = 3

    dataset['Fare'] = dataset['Fare'].astype(int)

    

    # Mapping Age

    dataset.loc[ dataset['Age'] <= 16, 'Age']                          = 0

    dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1

    dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2

    dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3

    dataset.loc[ dataset['Age'] > 64, 'Age']                           = 4
# Feature selection

drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp']

train_data = train_data.drop(drop_elements, axis = 1)

train_data = train_data.drop(['CategoricalAge', 'CategoricalFare'], axis = 1)

test_data  = test_data.drop(drop_elements, axis = 1)
train_data.head(3)
colormap = plt.cm.rainbow

plt.figure(figsize=(12,12))

plt.title('Pearson Correlation of Features', y=1.05, size=15)

sns.heatmap(train_data.astype(float).corr(),linewidths=0.1,vmax=1.0, square=True, cmap=colormap, linecolor='white', annot=True)
train_data = train_data.as_matrix()

test_data = test_data.as_matrix()

X_train = train_data[100:,1:]

y_train = train_data[100:,:1]

y_train = np.reshape(y_train,-1)

X_val = train_data[:100,1:]

y_val = train_data[:100,:1]

y_val = np.reshape(y_val,-1)

X_test = test_data

print('Train data shape: ', X_train.shape)

print('Train labels shape: ', y_train.shape)

print('Validation data shape: ', X_val.shape)

print('Validation labels shape: ', y_val.shape)

print('Test data shape: ', X_test.shape)
def run_model(session, predict, loss_val, Xd, yd,

              epochs=1, batch_size=64, print_every=100,

              training=None, plot_losses=False):

    # have tensorflow compute accuracy

    correct_prediction = tf.equal(tf.argmax(predict,1), y)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    

    # shuffle indicies

    train_indicies = np.arange(Xd.shape[0])



    training_now = training is not None

    

    # setting up variables we want to compute (and optimizing)

    # if we have a training function, add that to things we compute

    variables = [cost_op,correct_prediction,accuracy]

    if training_now:

        variables[-1] = training

    

    # counter 

    iter_cnt = 0

    for e in np.arange(epochs):

        # keep track of losses and accuracy

        correct = 0

        losses = []

        # make sure we iterate over the dataset once

        for i in np.arange(int(math.ceil(Xd.shape[0]/batch_size))):

            # generate indicies for the batch

            start_idx = (i*batch_size)%Xd.shape[0]

            idx = train_indicies[start_idx:start_idx+batch_size]

            

            # create a feed dictionary for this batch

            feed_dict = {x: Xd[idx,:],

                         y: yd[idx],

                         is_training: training_now }

            # get batch size

            actual_batch_size = yd[idx].shape[0]

            # have tensorflow compute loss and correct predictions

            # and (if given) perform a training step

            loss, corr, _ = session.run(variables,feed_dict=feed_dict)

            

            # aggregate performance stats

            losses.append(loss*actual_batch_size)

            correct += np.sum(corr)

            

            # print every now and then

            if training_now and (iter_cnt % print_every) == 0:

                print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\

                      .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))

            iter_cnt += 1

        total_correct = correct/Xd.shape[0]

        total_loss = np.sum(losses)/Xd.shape[0]

        print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"

              .format(total_loss,total_correct,e+1))

        if plot_losses and (e == epochs-1):

            plt.plot(losses)

            plt.grid(True)

            plt.title('Epoch {} Loss'.format(e+1))

            plt.xlabel('minibatch number')

            plt.ylabel('minibatch loss')

            plt.show()

    return total_loss,total_correct
numFeatures = X_train.shape[1]

numLabels = 2
# clear old variables

tf.reset_default_graph()



x = tf.placeholder(tf.float32, [None, numFeatures])

y = tf.placeholder(tf.int64, [None])

is_training = tf.placeholder(tf.bool)

Lambda = 0.001 #Regularization Parameter

learningRate = tf.train.exponential_decay(learning_rate=1e-2,

                                          global_step= 1,

                                          decay_steps=X_train.shape[0],

                                          decay_rate= 0.95,

                                          staircase=True)
# Logistic Regression

def Titanicmodel(x,y,is_training):   

    weights=tf.get_variable("weights",shape=[numFeatures,numLabels])

    bias=tf.get_variable("bias",shape=[numLabels])

    y_out = tf.matmul(x,weights)+bias

    return(y_out,weights)

y_out,weights = Titanicmodel(x,y,is_training)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.one_hot(y,2),logits=y_out))

regularizer = tf.nn.l2_loss(weights)

cost_op = tf.reduce_mean(loss + Lambda * regularizer)

optimizer = tf.train.GradientDescentOptimizer(learningRate)

train_step = optimizer.minimize(cost_op)
#Prediction

prediction = tf.argmax(y_out,1)
#Lets strat a session

sess = tf.Session()

sess.run(tf.global_variables_initializer())
print('Training')

run_model(sess,y_out,cost_op,X_train,y_train,100,100,100,train_step,True)

print('Validation')

run_model(sess,y_out,cost_op,X_val,y_val,1,100)
predicted_vallabels = np.zeros(X_val.shape[0])

for i in np.arange(0,X_val.shape[0]/50,dtype=np.int64):

    start = i*50

    end = (i+1)*50

    predicted_vallabels[start:end] = sess.run(prediction,feed_dict={x: X_val[start:end,:],y: predicted_vallabels[start:end],is_training: False})

# Compute ROC curve and ROC area for each class

fpr = dict()

tpr = dict()

roc_auc = dict()

fpr, tpr, _ = metrics.roc_curve(y_val, predicted_vallabels)

roc_auc = metrics.auc(fpr, tpr)

plt.figure()

plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc)

plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

plt.xlim([0.0, 1.0])

plt.ylim([0.0, 1.05])

plt.xlabel('False Positive Rate')

plt.ylabel('True Positive Rate')

plt.title('ROC_Curve')

plt.legend(loc="lower right")

plt.show()
predicted_labels = np.zeros(X_test.shape[0])

for i in np.arange(0,X_test.shape[0]/50,dtype=np.int64):

    start = i*50

    end = (i+1)*50

    predicted_labels[start:end] = sess.run(prediction,feed_dict={x: X_test[start:end,:],y: predicted_labels[start:end],is_training: False})
print('predicted_labels:',predicted_labels[5])
testID=pd.read_csv('../input/gendermodel.csv')

print(testID.shape)

PassengerId = testID['PassengerId']
# save results

np.savetxt('submission.csv', 

           np.c_[PassengerId,predicted_labels], 

           delimiter=',', 

           header = 'PassengerId,Survived', 

           comments = '', 

           fmt='%d')
sess.close()