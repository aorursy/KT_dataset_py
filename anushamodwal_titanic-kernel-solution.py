# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train['Dataset'] = "train"
test['Dataset'] = "test"

all_data = train.append(test, sort='True')
all_data.info()
all_data.head()
to_drop = ['Ticket', 'Cabin']
all_data.drop(columns = to_drop, inplace = True, axis = 1)
all_data.head()
all_data[['Pclass', 'Survived']].groupby(['Pclass'], as_index = False).mean().sort_values(by = ['Survived'], ascending = False)
all_data[['Sex', 'Survived']].groupby(['Sex'], as_index = False).mean().sort_values(by = ['Survived'], ascending = False)
all_data[['SibSp', 'Survived']].groupby(['SibSp'], as_index = False).mean().sort_values(by = ['Survived'], ascending = False)
all_data[['Parch', 'Survived']].groupby(['Parch'], as_index = False).mean().sort_values(by = ['Survived'], ascending = False)
corrMatrix = all_data[['Age', 'Embarked', 'Fare', 'Name', 'Parch', 'Pclass', 'Sex', 'SibSp']].corr()

# Masking upper triangle as values are same as lower traingle
mask = np.zeros_like(corrMatrix, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
mask[np.diag_indices_from(mask)] = False

# Initializing matplotlib figure
fig, ax = plt.subplots(figsize=(10,10))
color = sns.diverging_palette(200, 15, as_cmap = True)
sns.heatmap(corrMatrix, mask = mask, cmap = color, square = True, annot = True, vmax = 0.5)
display(all_data[all_data['Fare'].isnull()])
df_class_emb = all_data['Fare'].loc[(all_data['Embarked'] == 'S') & (all_data['Pclass'] == 3)]
plt.figure(figsize=[10,10])
sns.distplot(df_class_emb.dropna(), color='C0')
plt.plot([df_class_emb.median(), df_class_emb.median()], [0, 0.16], '--', color='C1')

sns.despine(offset = 10)
all_data['Fare'] = all_data['Fare'].fillna(df_class_emb.median())
all_data['FareRange'] = pd.qcut(all_data['Fare'], 4)
all_data[['FareRange', 'Survived']].groupby(['FareRange'], as_index = False).mean().sort_values(by = 'FareRange', ascending = True)
all_data.loc[all_data['Fare'] <= 7.91, 'Fare'] = 0
all_data.loc[(all_data['Fare'] > 7.91) & (all_data['Fare'] <= 14.454), 'Fare'] = 1
all_data.loc[(all_data['Fare'] > 14.454) & (all_data['Fare'] <= 31), 'Fare']   = 2
all_data.loc[ all_data['Fare'] > 31, 'Fare'] = 3
all_data['Fare'] = all_data['Fare'].astype(int)

all_data = all_data.drop(['FareRange'], axis=1)
    
all_data.head(10)
display(all_data[all_data['Embarked'].isnull()])
all_data['Embarked'] = all_data['Embarked'].fillna('C')
display(all_data[all_data['Embarked'].isnull()])
display(all_data[all_data['Age'].isnull()])
plt.figure(figsize=[10,10])
sns.distplot(all_data['Age'].dropna(), color='C0')
plt.plot([all_data['Age'].median(), all_data['Age'].median()], [0, 0.4], '--', color = 'C1')
plt.plot([all_data['Age'].mean(), all_data['Age'].mean()], [0, 0.2], '-', color = 'C2')

sns.despine(offset = 10)
plt.title('Distribution plot of Age data for all passengers')
plt.xlabel('Age')
plt.legend(['median', 'mean'])
plt.show()
g = sns.FacetGrid(all_data, col='Survived')
g.map(plt.hist, 'Age', bins=20)
grid = sns.FacetGrid(all_data, col='Survived', row='Pclass', size=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();
grid = sns.FacetGrid(all_data, row='Embarked', size=2.2, aspect=1.6)
grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', palette='deep')
grid.add_legend()
all_data['Title'] = all_data.Name.str.extract(' ([A-Za-z]+)\.', expand = False)
    
pd.crosstab(all_data['Title'], all_data['Sex'])
# Merging all columns with similar values and grouping rare values as "Other"
all_data['Title'] = all_data['Title'].replace(['Capt', 'Col', 'Countess', 'Don', 'Dona', 'Dr', 'Jonkheer', 'Lady', 'Major', 'Rev', 'Sir'], 'Other')

all_data['Title'] = all_data['Title'].replace(['Ms'], 'Miss')
all_data['Title'] = all_data['Title'].replace(['Mlle'], 'Miss')
all_data['Title'] = all_data['Title'].replace(['Mme'], 'Mrs')

all_data[['Title', 'Survived']].groupby('Title', as_index = False).mean()
# For 'Name' feature

title_map = {'Master':0, 'Miss':1, 'Mr':2, 'Mrs':3, 'Other':4}
all_data['Title'] = all_data['Title'].map(title_map)
all_data['Title'] = all_data['Title'].fillna(-1)

# Dropping "Name" feature after processing
all_data.drop(columns = ['Name'], inplace = True, axis = 1)
# For Sex feature
all_data['Sex'] = all_data['Sex'].map({'male':0, 'female':1}).astype(int)
# For Embarked feature
all_data['Embarked'] = all_data['Embarked'].map({'S':0, 'C':1, 'Q':2}).astype(int)
# Due to correlation between Age and Pclass
guess_ages = np.zeros((2,3))
for i in range(0, 2):
    for j in range(0, 3):
        guess_df = all_data[(all_data['Sex'] == i) & (all_data['Pclass'] == j+1)]['Age'].dropna()
        age_guess = guess_df.median()

        # Convert random age float to nearest .5 age
        guess_ages[i,j] = int(age_guess / 0.5 + 0.5) * 0.5
            
for i in range(0, 2):
    for j in range(0, 3):
        all_data.loc[ (all_data.Age.isnull()) & (all_data.Sex == i) & (all_data.Pclass == j+1),'Age'] = guess_ages[i,j]

all_data['Age'] = all_data['Age'].astype(int)

all_data.head()
all_data['AgeRange'] = pd.cut(all_data['Age'], 5)
all_data[['AgeRange', 'Survived']].groupby(['AgeRange'], as_index=False).mean().sort_values(by='AgeRange', ascending=True)
all_data.loc[all_data['Age'] <= 16, 'Age'] = 0
all_data.loc[(all_data['Age'] > 16) & (all_data['Age'] <= 32), 'Age'] = 1
all_data.loc[(all_data['Age'] > 32) & (all_data['Age'] <= 48), 'Age'] = 2
all_data.loc[(all_data['Age'] > 48) & (all_data['Age'] <= 64), 'Age'] = 3
all_data.loc[ all_data['Age'] > 64, 'Age']
all_data.head()
# Now we remove AgeRange feature
all_data = all_data.drop(['AgeRange'], axis = 1)
all_data.head()
all_data['FamilySize'] = all_data["SibSp"] + all_data["Parch"] + 1

all_data[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index= False).mean().sort_values(by = 'Survived', ascending = False)
all_data['IsAlone'] = 0
all_data.loc[all_data['FamilySize'] == 1, 'IsAlone'] = 1

all_data[['IsAlone', 'Survived']].groupby(['IsAlone'], as_index = False).mean().sort_values(by = 'Survived', ascending = False)
all_data = all_data.drop(['Parch', 'SibSp', 'FamilySize'], axis = 1)
all_data.head()
train_data = all_data[all_data['Dataset'] == 'train']
train_data = train_data.drop(['Dataset', 'PassengerId'], axis = 1)
train_data.head()
test_data = all_data[all_data['Dataset'] == 'test']
test_data = test_data.drop(['Dataset', 'PassengerId', 'Survived'], axis = 1)
test_data.head()
train_size = int(train.shape[0] * 0.85)

train_dataset = train_data[:train_size]
val_dataset = train_data[train_size:]

X_train = (train_dataset.drop(labels=["Survived"], axis=1).values).T
Y_train =  np.reshape(train_dataset["Survived"].values, (1, len(train_dataset)))

X_val = (val_dataset.drop(labels=["Survived"], axis=1).values).T
Y_val = np.reshape(val_dataset["Survived"].values, (1, len(val_dataset)))

X_test = (test_data.values.astype(np.float32)).T

input_size = len(train_dataset.columns) - 1  # number of final features
input_size
print(X_train.shape, X_val.shape)
print(Y_train.shape, Y_val.shape)
print(X_test.shape)
train_data.head()
import math
import tensorflow as tf
from tensorflow.python.framework import ops
def create_placeholders(n_x, n_y):
    # n_x - number of features
    # n_y - number of classes
    X = tf.placeholder(tf.float32, [n_x, None], name = 'X')
    Y = tf.placeholder(tf.float32, [n_y, None], name = 'Y')
    return X, Y
def initialize_parameters():                  
    W1 = tf.get_variable("W1", [7,7], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1", [7,1], initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2", [5,7], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2", [5,1], initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3", [1,5], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3", [1,1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return parameters
def forward_propagation(X, parameters):
    # Retrieve the parameters from the dictionary parameters
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = tf.add(tf.matmul(W1, X), b1)                                   
    A1 = tf.nn.elu(Z1)                                              
    Z2 = tf.add(tf.matmul(W2, A1), b2)                                      
    A2 = tf.nn.elu(Z2)                                         
    Z3 = tf.add(tf.matmul(W3, A2), b3)
    
    return Z3
def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = labels))
    
    return cost
def random_mini_batches(X, Y, mini_batch_size = 32, seed = 0):
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches
def model(X_train, Y_train, X_test, Y_test, learning_rate = 0.003, num_epochs = 1500, minibatch_size = 32, print_cost = True):
    # Implements a three layer layer neural network using tensorflow
    
    ops.reset_default_graph()                         # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)                             # to keep consistent results
    seed = 3                                          # to keep consistent results
    (n_x, m) = X_train.shape                          # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]                            # n_y : output size
    costs = []                                        # To keep track of the cost

    X, Y = create_placeholders(n_x, n_y)

    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0.                       # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size) # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)

            for minibatch in minibatches:
                (minibatch_X, minibatch_Y) = minibatch
                _ , minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})
                epoch_cost += minibatch_cost / num_minibatches
                
            if print_cost == True and epoch % 100 == 0:
                print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
                
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        parameters = sess.run(parameters)
        print ("Parameters have been trained!")
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
        
        return parameters
model_params = model(X_train, Y_train, X_val, Y_val)
def predict(parameters, X):
    predictions = forward_propagation(X, parameters)
    
    return predictions
y_prediction = predict(model_params, X_test)
sess = tf.Session()
with sess.as_default():
    out = np.round((tf.nn.sigmoid(y_prediction)).eval())
df_test = pd.read_csv("../input/test.csv")
output = pd.DataFrame({'PassengerId': df_test['PassengerId'], 'Survived': np.squeeze((out.astype(int)).reshape(-1))})
output.to_csv('submission.csv', index=False)
print(output.groupby('Survived').count())
output
from sklearn.metrics import confusion_matrix

df = pd.read_csv("../input/gender_submission.csv")
survived_actual = df['Survived']
survived_predicted = output['Survived']

conf_mat = confusion_matrix(survived_actual, survived_predicted)
# true positives(tp) : (1,1) --> predicted 1 and actual was 1
# true negatives(tn) : (0,0) --> predicted 0 and actual was 0
# false positives(fp): (0,1) --> predicted 1 and actual was 0
# false negatives(fn): (1,0) --> predicted 0 and actual was 1 
tp = conf_mat[1,1]
tn = conf_mat[0,0]
fp = conf_mat[0,1]
fn = conf_mat[1,0]
precision = tp / (tp + fp)
recall = tp / (tp + fn)
fscore = 2 * (precision * recall)/ (precision + recall)

print('Precision: ', precision)
print('Recall: ', recall)
print('F1 Score: ', fscore)







