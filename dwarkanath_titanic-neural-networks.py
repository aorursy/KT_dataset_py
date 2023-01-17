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
train.head()
train.info()
train.columns
m = train.shape[0]
catCols = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked']
j = 0
for i in catCols:
    plt.figure(j)
    sns.barplot(x = i, y = 'Survived', data = train)
    plt.show()
    j+=1
test = pd.read_csv('../input/test.csv')
dfs = [train,test]
data = pd.concat(dfs,ignore_index=True)
data.head()
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
data.head()
data.info()
data.groupby('Title').Age.mean()
data['Age'].fillna(data.groupby('Title')['Age'].transform("mean"), inplace=True)
data.loc[pd.isnull(data['Embarked'])]
data.loc[61,'Embarked'] = 'S'
data.loc[829,'Embarked'] = 'S'
data['Fare'].fillna(data['Fare'].mean(), inplace = True)
num_family = (data['Parch'] + data['SibSp']).astype(int)
data['num_family'] = num_family
sns.barplot(x = 'num_family', y = 'Survived', data = data[:m])
plt.show()
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
Xcols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Title', 'num_family']
data[Xcols].info()
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
# Load preprocessed data from file


#X = np.load(X_file)

#y = np.load(y_file)
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

y_dev.shape
# Initialize placeholders for data
n = X.shape[1]
tf.reset_default_graph()
x = tf.placeholder(dtype=tf.float32, shape=[None, n], name = 'inputs_ph')
y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name = 'labels_ph')
# number of neurons in each layer

input_num_units = n
hidden_num_units_1 = 224
hidden_num_units_2 = 120
hidden_num_units_3 = 56
output_num_units = 1
# Build Neural Network Weights
initializer = tf.contrib.layers.xavier_initializer()
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

learning_rate = 7e-5
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

    # save results in a list

    loss_trace.append(temp_loss)
    train_acc.append(temp_train_acc)
    dev_acc.append(temp_dev_acc)

    # output

    if (epoch + 1) % 200 == 0:
        print('epoch: {:4d} loss: {:5f} train_acc: {:5f} dev_acc: {:5f}'.format(epoch + 1, temp_loss, temp_train_acc, temp_dev_acc))


plt.plot(loss_trace)
y_train_preds_nn = sess.run(prediction, feed_dict ={x: X_train, keep_prob_1: 1, keep_prob_2: 1, keep_prob_3: 1})
y_dev_preds_nn = sess.run(prediction, feed_dict ={x: X_dev, keep_prob_1: 1, keep_prob_2: 1, keep_prob_3: 1})
y_test_preds_nn = sess.run(prediction, feed_dict ={x: X_test, keep_prob_1: 1, keep_prob_2: 1, keep_prob_3: 1})
sess.close()
def get_recall(labels, preds):
    tp = int(np.dot(labels.T,preds))
    fn = int(np.dot(labels.T,1-preds))
    recall = tp/(tp+fn)
    return recall
recall_nn = get_recall(y_train, y_train_preds_nn)
recall_nn
test['Survived_nn'] = y_test_preds_nn.astype(int)
test[['PassengerId', 'Survived_nn']].to_csv('submission_nn.csv', index = False, header = ['PassengerId', 'Survived'])
svclassifier = SVC()
svclassifier.fit(X_train, y_train) 
y_train_preds_svm = np.expand_dims(svclassifier.predict(X_train),1)
train_acc_svm = np.mean(y_train == y_train_preds_svm)
print('Train Accuracy for SVM: {:5f}'.format(train_acc_svm))
y_test_preds_svm = svclassifier.predict(X_test)
recall_svm = get_recall(y_train, y_train_preds_svm)
recall_svm
test['Survived_svm'] = y_test_preds_svm.astype(int)
test[['PassengerId', 'Survived_svm']].to_csv('submission_svm.csv', index = False, header = ['PassengerId', 'Survived'])
rfclassifier = RandomForestClassifier(n_estimators = 100, max_features = 3)
rfclassifier.fit(X_train, y_train) 
y_train_preds_rf = np.expand_dims(rfclassifier.predict(X_train),1)
train_acc_rf = np.mean(y_train == y_train_preds_rf)
print('Train Accuracy for Random Forest: {:5f}'.format(train_acc_rf))
y_dev_preds_rf = np.expand_dims(rfclassifier.predict(X_dev),1)
dev_acc_rf = np.mean(y_dev == y_dev_preds_rf)
print('Dev Accuracy for Random Forest: {:5f}'.format(dev_acc_rf))
recall_rf = get_recall(y_train, y_train_preds_rf)
recall_rf
y_test_preds_rf = rfclassifier.predict(X_test)
test['Survived_rf'] = y_test_preds_rf.astype(int)
test[['PassengerId', 'Survived_rf']].to_csv('submission_rf.csv', index = False, header = ['PassengerId', 'Survived'])
test.columns
test['Survived_nn_wtd'] = test['Survived_nn']*recall_nn
test['Survived_svm_wtd'] = test['Survived_svm']*recall_svm
test['Survived_rf_wtd'] = test['Survived_nn']*recall_rf
y_test_preds_avg = np.round(np.mean(test[['Survived_nn', 'Survived_svm','Survived_rf']],axis = 1))
y_test_preds_wtd_avg = np.round(np.mean(test[['Survived_nn_wtd', 'Survived_svm_wtd','Survived_rf_wtd']],axis = 1))
test['Survived_avg'] = y_test_preds_avg.astype(int)
test['Survived_wtd_avg'] = y_test_preds_wtd_avg.astype(int)
test[['PassengerId', 'Survived_avg']].to_csv('submission_avg.csv', index = False, header = ['PassengerId', 'Survived'])
test[['PassengerId', 'Survived_wtd_avg']].to_csv('submission_wtd_avg.csv', index = False, header = ['PassengerId', 'Survived'])