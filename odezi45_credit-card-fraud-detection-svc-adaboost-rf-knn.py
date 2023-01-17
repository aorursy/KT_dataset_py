from sklearn import preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import svm, tree
import xgboost
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import pickle
%matplotlib inline
import datetime 
import time 
import tensorflow as tf
# Load the data
data  = pd.read_csv('/kaggle/input/1056lab-credit-card-fraud-detection/train.csv')
test  = pd.read_csv('/kaggle/input/1056lab-credit-card-fraud-detection/test.csv')
submission = test[['ID']]

#display some information about data
data.info()
test.info()
# data.isnull() # shows a df with the information whether a data point is null 
# Since True = the data point is missing, while False = the data point is not missing, we can sum them
# This will give us the total number of missing values feature-wise
data.isnull().sum()
# As we can see there are no missing data

# so let take a look at the spread of each column
# Visulazing the distibution of the data for every feature
data.hist(linewidth=1, histtype='stepfilled', facecolor='g', figsize=(20, 20));
#lets look at the basic description of the data
data.describe()
#test.describe()
# looking at the datset, we can infer that the datsset contains two days of traction or the record spans a  48hrs period and as 
# such, I have generated the hour of transaction by assuming that the first seconds of a day is 0 and the last one 86399, Hence:

data['HourBank'] = ((np.where(data['Time'] > 86399 , data['Time'] - 86399 , data['Time'])) % (24 *3600) // 3600).astype(int)
test['HourBank'] = ((np.where(test['Time'] > 86399 , test['Time'] - 86399 , test['Time'])) % (24 *3600) // 3600).astype(int)   
#temptime = np.where(data['Time'] > 86399 , data['Time'] - 86399 , data['Time'])
#data['HourBank'] = data['HourBank'].astype(int)
#test['HourBank'] = test['HourBank'].astype(int)



data['Class'].groupby(data['Class']).count()
# First we limit the data frame to where a fraudulent activity was identify
d = data[data['Class'] == 1]

# group by HourBank, then count of fradulent transaction
d1 = d[['Class','HourBank']]

d1 = d1.groupby(['HourBank']).count()
d1.reset_index(level=0, inplace=True)

# group by HourBank, then average Amount of fradulent transaction
d2 = d[['Amount','HourBank']]
# group store and dept by average weekly sales
d2 = d2.groupby(['HourBank']).mean()
d2.reset_index(level=0, inplace=True)


#Lets do a quick plot to visualise the data
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(10,7))

ax1.bar(d1.HourBank, d1.Class)
ax1.set_title('Count of Fraudulent transactions by Hour')
ax2.bar(d2.HourBank, d2.Amount)
ax2.set_title('Average Amount Classed as Fraudulent by Hour')

fig.tight_layout()
plt.show()
#lets get dummies
T_dummies = pd.get_dummies(data['HourBank'])

#lets merge it
data = pd.concat([data, T_dummies], axis = 1)

# then drop the redundant column
data = data.drop(['HourBank','ID','Time'], axis = 1)


##### Repeat the same for test data

Tt_dummies = pd.get_dummies(test['HourBank'])
a = list(Tt_dummies.columns.values)
b = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22, 23]
c = [x for x in b if x not in a]
Tt_dummies = pd.concat([Tt_dummies, pd.DataFrame(columns = c)]).fillna(0)  
#lets merge it
test = pd.concat([test, Tt_dummies], axis = 1)
# then drop the redundant column
test = test.drop(['HourBank','ID','Time'], axis = 1)



# Balance the data based on column class
g = data.groupby('Class')
g = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True)))
g = g.reset_index(drop=True)

###shuffle dataset
g = g.sample(frac=1).reset_index(drop=True)


# The inputs are all columns in the csv, except for the first one [:,0]
# (which is just the arbitrary customer IDs that bear no useful information),
# and the last one [:,-1] (which is our targets)
balanced_inputs = g.drop(['Class'],axis=1)
# The targets are in the last column. That's how datasets are conventionally organized.
balanced_targets = g['Class'].astype(np.int)
unbalanced_inputs = data.drop(['Class'],axis=1)
# The targets are in the last column. That's how datasets are conventionally organized.
unbalanced_targets = data['Class'].astype(np.int)

balanced_inputs = preprocessing.scale(balanced_inputs)
scaled_unbalanced_inputs = preprocessing.scale(unbalanced_inputs)

test_inputsx = preprocessing.scale(test)

shuffled_inputs = balanced_inputs
shuffled_targets = balanced_targets
# Count the total number of samples
samples_count = shuffled_inputs.shape[0]

# Count the samples in each subset, assuming we want 80-10-10 distribution of training, validation, and test.
# Naturally, the numbers are integers.
train_samples_count = int(0.8 * samples_count)
validation_samples_count = int(0.1 * samples_count)

# The 'test' dataset contains all remaining data.
test_samples_count = samples_count - train_samples_count - validation_samples_count

# Create variables that record the inputs and targets for training
# In our shuffled dataset, they are the first "train_samples_count" observations
train_inputs = shuffled_inputs[:train_samples_count]
train_targets = shuffled_targets[:train_samples_count]

# Create variables that record the inputs and targets for validation.
# They are the next "validation_samples_count" observations, folllowing the "train_samples_count" we already assigned
validation_inputs = shuffled_inputs[train_samples_count:train_samples_count+validation_samples_count]
validation_targets = shuffled_targets[train_samples_count:train_samples_count+validation_samples_count]

# Create variables that record the inputs and targets for test.
# They are everything that is remaining.
test_inputs = shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets = shuffled_targets[train_samples_count+validation_samples_count:]

# We balanced our dataset to be 50-50 (for targets 0 and 1), but the training, validation, and test were 
# taken from a shuffled dataset. Check if they are balanced, too. Note that each time you rerun this code, 
# you will get different values, as each time they are shuffled randomly.
# Normally you preprocess ONCE, so you need not rerun this code once it is done.
# If you rerun this whole sheet, the npzs will be overwritten with your newly preprocessed data.

# Print the number of targets that are 1s, the total number of samples, and the proportion for training, validation, and test.
print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)
print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)
#Unbalanced datset
train_test_split(scaled_unbalanced_inputs, unbalanced_targets)
# declare 4 variables for the split
x_train1, x_test1, y_train1, y_test1 = train_test_split(scaled_unbalanced_inputs, unbalanced_targets, #train_size = 0.75, 
                                                                            test_size = 0.25, random_state = 20)

#Now, we will create an array of Classifiers and append different classification models to our array
classifiers = [] 

mod1 = xgboost.XGBClassifier()
classifiers.append(mod1)
mod2 = svm.SVC()
classifiers.append(mod2)
mod3 = RandomForestClassifier()
classifiers.append(mod3)
mod4 = LogisticRegression()
classifiers.append(mod4)
mod5 = KNeighborsClassifier(3)
classifiers.append(mod5)
mod6 = AdaBoostClassifier()
classifiers.append(mod6)
mod7= GaussianNB()
classifiers.append(mod7)
#Lets fit the models into anarray

for clf in classifiers:
    clf.fit(train_inputs,train_targets)
    y_pred= clf.predict(test_inputs)
    y_tr = clf.predict(train_inputs)
    acc_tr = accuracy_score(train_targets, y_tr)
    acc = accuracy_score(test_targets, y_pred)
    mn = type(clf).__name__
    
    print(clf)
    print("Accuracy of trainset %s is %s"%(mn, acc_tr))
    print("Accuracy of testset %s is %s"%(mn, acc))
    cm = confusion_matrix(test_targets, y_pred)
    print("Confusion Matrix of testset %s is %s"%(mn, cm))
# convert all value into array
validation_inputs = np.array(validation_inputs)
validation_targets = np.array(validation_targets)
train_targets = np.array(train_targets)
train_inputs = np.array(train_inputs)
# Set the input and output sizes
input_size = 53
output_size = 2
# Use same hidden layer size for both hidden layers. Not a necessity.
hidden_layer_size = 6
    
# define how the model will look like
model = tf.keras.Sequential([
    # tf.keras.layers.Dense is basically implementing: output = activation(dot(input, weight) + bias)
    # it takes several arguments, but the most important ones for us are the hidden_layer_size and the activation function
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 1st hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 2nd hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 2nd hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 2nd hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 2nd hidden layer
    # the final layer is no different, we just make sure to activate it with softmax
    tf.keras.layers.Dense(output_size, activation='softmax') # output layer
])


### Choose the optimizer and the loss function

# we define the optimizer we'd like to use, 
# the loss function, 
# and the metrics we are interested in obtaining at each iteration
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

### Training
# That's where we train the model we have built.

# set the batch size
batch_size = 100

# set a maximum number of training epochs
max_epochs = 100

# set an early stopping mechanism
# let's set patience=2, to be a bit tolerant against random validation loss increases
early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

# fit the model
# note that this time the train, validation and test data are not iterable
history = model.fit(train_inputs, # train inputs
          train_targets, # train targets
          batch_size=batch_size, # batch size
          epochs=max_epochs, # epochs that we will train for (assuming early stopping doesn't kick in)
          # callbacks are functions called by a task when a task is completed
          # task here is to check if val_loss is increasing
          callbacks=[early_stopping], # early stopping
          validation_data=(validation_inputs, validation_targets), # validation data
          verbose = 2 # making sure we get enough information about the training process
          )  

# Plot the train/validation loss values
plt.figure(figsize=(15,10))
_loss = history.history['loss'][1:]
_val_loss = history.history['val_loss'][1:]

train_loss_plot, = plt.plot(range(1, len(_loss)+1), _loss, label='Train Loss')
val_loss_plot, = plt.plot(range(1, len(_val_loss)+1), _val_loss, label='Validation Loss')

_ = plt.legend(handles=[train_loss_plot, val_loss_plot])
test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)
#first lets convert the inputs to array
y_train1 = np.array(y_train1)
y_test1 = np.array(y_test1)
# Set the input and output sizes
input_size = 53
output_size = 2
# Use same hidden layer size for both hidden layers. Not a necessity.
hidden_layer_size = 2
    
# define how the model will look like
tfk = tf.keras.Sequential([
    # tf.keras.layers.Dense is basically implementing: output = activation(dot(input, weight) + bias)
    # it takes several arguments, but the most important ones for us are the hidden_layer_size and the activation function
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 1st hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 2nd hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 2nd hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 2nd hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 2nd hidden layer
    # the final layer is no different, we just make sure to activate it with softmax
    tf.keras.layers.Dense(output_size, activation='softmax') # output layer
])


### Choose the optimizer and the loss function

# we define the optimizer we'd like to use, 
# the loss function, 
# and the metrics we are interested in obtaining at each iteration
tfk.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

### Training
# That's where we train the model we have built.

# set the batch size
batch_size = 100

# set a maximum number of training epochs
max_epochs = 100

# set an early stopping mechanism
# let's set patience=2, to be a bit tolerant against random validation loss increases
early_stopping = tf.keras.callbacks.EarlyStopping(patience=2)

# fit the model
# note that this time the train, validation and test data are not iterable
history = tfk.fit(x_train1, # train inputs
          y_train1, # train targets
          batch_size=batch_size, # batch size
          epochs=max_epochs, # epochs that we will train for (assuming early stopping doesn't kick in)
          # callbacks are functions called by a task when a task is completed
          # task here is to check if val_loss is increasing
          callbacks=[early_stopping], # early stopping
          validation_data=(x_test1, y_test1), # validation data
          verbose = 2 # making sure we get enough information about the training process
          )  

# Plot the train/validation loss values
plt.figure(figsize=(20,10))
_loss = history.history['loss'][1:]
_val_loss = history.history['val_loss'][1:]

train_loss_plot, = plt.plot(range(1, len(_loss)+1), _loss, label='Train Loss')
val_loss_plot, = plt.plot(range(1, len(_val_loss)+1), _val_loss, label='Validation Loss')

_ = plt.legend(handles=[train_loss_plot, val_loss_plot])
tfbalanced = model.predict_classes(test_inputsx) # Deep learning with balanced data
tfunbalanced = tfk.predict_classes(test_inputsx)  # Deep Learning with unbalanced data
logreg = mod4.predict(test_inputsx)              # Logistic regression with balanced data
ada = mod6.predict(test_inputsx)            # Adaboost with balanced data
#df = pd.DataFrame(pred, columns=['Prediction'])
submission['tfb'] = tfbalanced          # the AUC score = 54
submission['tfu'] = tfunbalanced        # the AUC score = 0.89
submission['logr'] = logreg             # the AUC score = 57
submission['ada'] = ada                 # the AUC score = 63
submission.to_csv('submission.csv', index=False)
