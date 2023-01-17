import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from tensorflow.python.saved_model import tag_constants
from sklearn.metrics import confusion_matrix

train = pd.read_csv('../input/train.csv')


train.columns
m = train.shape[0]

catCols = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']

    
test = pd.read_csv('../input/test.csv')
dfs = [train,test]
data = pd.concat(dfs,ignore_index=True)


def getTitles(names):
    titleRegex = re.compile(r',.\w+\.')    
    title = []
    for str in names:
        titlePat = re.search(titleRegex,str)
        if titlePat is None:
            title.append(str)
        else:
            x = titlePat.group()
            x = x[2:len(x)-1]
            title.append(x)
    return title


title = getTitles(data['Name'])
set(title)
def getCleanTitles(title):
    for i in range(len(title)):
        if title[i] in ['Don', 'Sir', 'Jonkheer']:
            title[i] = 'Noble'
        elif title[i] in ['Rothes, the Countess. of (Lucy Noel Martha Dyer-Edwards)', 'Lady', 'Dona']:
            title[i] = 'Noble'
        elif title[i] in ['Mlle', 'Ms']:
            title[i] = 'Miss'
        elif title[i] == 'Mme':
            title[i] = 'Mrs'
        elif title[i] in ['Capt', 'Col', 'Dr', 'Major', 'Rev']:
            title[i] = 'Other'
    return title

data['Title'] = getCleanTitles(title)
data.groupby('Title').Age.mean()
data['Age'].fillna(data.groupby('Title')['Age'].transform("mean"), inplace=True)
data.loc[pd.isnull(data['Embarked'])]
data.loc[61,'Embarked'] = 'S'
data.loc[829,'Embarked'] = 'S'
data['Fare'].fillna(data['Fare'].mean(), inplace = True)
num_family = (data['Parch'] + data['SibSp']).astype(int)
data['num_family'] = num_family
data.columns
catCols.extend(['Title', 'num_family'])
catCols
def convertCatValToNum(catVal):
    le = LabelEncoder()
    le.fit(catVal)
    catVal = le.transform(catVal)
    return catVal


for i in catCols:
    data[i] = convertCatValToNum(data[i])

    
data.columns
Xcols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'num_family']

scaler = MinMaxScaler()
scaler.fit(data[Xcols])
X = scaler.transform(data[Xcols])
# Check if the features have been correctly scaled

X_stats = pd.DataFrame()
X_stats['Min'] = np.min(X, axis = 0)
X_stats['Max'] = np.max(X, axis = 0)
X_stats['Mean'] = np.mean(X, axis = 0)
X_stats
y = np.expand_dims(data[:m].Survived.values,1)
y.shape
# Save preprocessed data to file

X_file = 'X.npy'
#np.save(X_file, X)

y_file = 'y.npy'
#np.save(y_file, y)
# Set random seed

seed = 2
np.random.seed(seed)

# Get random training index

train_index = np.random.choice(m, round(m*0.9), replace=False)
dev_index = np.array(list(set(range(m)) - set(train_index)))

test_index = range(m, data.shape[0])
# Make training and dev


X_train = X[train_index]
X_dev = X[dev_index]
X_test = X[test_index]

y_train = y[train_index]
y_dev = y[dev_index]

#y_data = pd.read_csv('../input/gender_submission.csv')
#y_data = y_data['Survived']
# Initialize placeholders for data
n = X.shape[1]
tf.reset_default_graph()
x = tf.placeholder(dtype=tf.float32, shape=[None, n], name = 'inputs_ph')
y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name = 'labels_ph')
# number of neurons in each layer

input_num_units = n
#hidden_num_units_1 = 224
#hidden_num_units_2 = 120
#hidden_num_units_3 = 56
hidden_num_units_1 = 150
hidden_num_units_2 = 80
hidden_num_units_3 = 50
output_num_units = 1
# Build Neural Network Weights
initializer=tf.contrib.layers.variance_scaling_initializer()

weights = {
    'hidden1': tf.Variable(initializer([input_num_units, hidden_num_units_1])),
    'hidden2': tf.Variable(initializer([hidden_num_units_1, hidden_num_units_2])),
    'hidden3': tf.Variable(initializer([hidden_num_units_2, hidden_num_units_3])),
    'output': tf.Variable(initializer([hidden_num_units_3, output_num_units])),
}

biases = {
    'hidden1': tf.Variable(initializer([hidden_num_units_1])),
    'hidden2': tf.Variable(initializer([hidden_num_units_2])),
    'hidden3': tf.Variable(initializer([hidden_num_units_3])),
    'output': tf.Variable(initializer([output_num_units])),
}
# Build model 

keep_prob_1 = tf.placeholder(dtype=tf.float32, name = 'keep_prob_1')
keep_prob_2 = tf.placeholder(dtype=tf.float32, name = 'keep_prob_2')
keep_prob_3 = tf.placeholder(dtype=tf.float32, name = 'keep_prob_3')

hidden_1_layer = tf.add(tf.matmul(x, weights['hidden1']), biases['hidden1'])
hidden_1_layer = tf.nn.dropout(tf.nn.relu(hidden_1_layer),keep_prob = keep_prob_1)
hidden_2_layer = tf.add(tf.matmul(hidden_1_layer, weights['hidden2']), biases['hidden2'])
hidden_2_layer = tf.nn.dropout(tf.nn.relu(hidden_2_layer),keep_prob = keep_prob_2)
hidden_3_layer = tf.add(tf.matmul(hidden_2_layer, weights['hidden3']), biases['hidden3'])
hidden_3_layer = tf.nn.dropout(tf.nn.relu(hidden_3_layer),keep_prob = keep_prob_3)

output_layer = tf.matmul(hidden_3_layer, weights['output']) + biases['output']
# Set hyperparameters

learning_rate = 0.0001
epochs = 1000

# Set loss function and goal i.e. minimize loss

loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=output_layer, labels=y))
opt = tf.train.AdamOptimizer(learning_rate)
goal = opt.minimize(loss)

prediction = tf.round(tf.nn.sigmoid(output_layer), name = 'prediction')
correct = tf.cast(tf.equal(prediction, y), dtype=tf.float32)
recall = tf.metrics.recall(labels = y, predictions = prediction)
accuracy = tf.reduce_mean(correct)

# Initialize lists to store loss and accuracy while training the model

loss_trace = []
train_acc = []
dev_acc = []
test_acc = []
# Start tensorflow session

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# Train the model

for epoch in range(epochs):
    
    sess.run(goal, feed_dict={x: X_train, y: y_train, keep_prob_1: 0.9, keep_prob_2: 0.8, keep_prob_3: 0.7})

    # calculate results for epoch
    
    temp_loss = sess.run(loss, feed_dict={x: X_train, y: y_train, keep_prob_1: 1, keep_prob_2: 1, keep_prob_3: 1})
    temp_train_acc = sess.run(accuracy, feed_dict={x: X_train, y: y_train, keep_prob_1: 1, keep_prob_2: 1, keep_prob_3: 1})
    temp_dev_acc = sess.run(accuracy, feed_dict={x: X_dev, y: y_dev, keep_prob_1: 1, keep_prob_2: 1, keep_prob_3: 1})
    temp_test_acc = sess.run(prediction, feed_dict={x: X_test, keep_prob_1: 1, keep_prob_2: 1, keep_prob_3: 1})
    #num = 0
    #for i in range(y_data.shape[0]-1):
    #    if y_data[i]==temp_test_acc[i][0]:
    #        num+=1
    #temp_test_acc_mean = np.sum(num)/float(y_data.shape[0])
    # save results in a list
    loss_trace.append(temp_loss)
    train_acc.append(temp_train_acc)
    dev_acc.append(temp_dev_acc)
    #test_acc.append(temp_test_acc_mean)
    # output

    if (epoch + 1) % 100 == 0:   
        print('epoch: {:4d} loss: {:5f} train_acc: {:5f} dev_acc: {:5f}'.format(epoch + 1, temp_loss, temp_train_acc, temp_dev_acc))
x = np.arange(epochs)
plt.plot(x, train_acc,  label='train')
plt.plot(x, dev_acc, label='test')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
test['Survived_nn'] = temp_test_acc.astype(int)
test[['PassengerId', 'Survived_nn']].to_csv('submission.csv', index = False, header = ['PassengerId', 'Survived'])
sess.close()

