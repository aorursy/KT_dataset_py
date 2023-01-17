import pandas as pd 
import numpy as np
import os
print(os.listdir("../input"))
path="../input/"
os.chdir(path)
data = pd.read_csv("Personal Loan Data.csv", low_memory=False)
data.head()
def drop_columns(df,columns):
    df1 = df.drop(columns, axis=1)
    return df1

    
    
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)
    
def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000): 
        display(df)
        
                 

def check_uniques(df,col):
    print(df.dtypes)
    print('Dim Of Data')
    print(df.shape)
    col=np.array(col)
    for y in col:
        col1 = pd.DataFrame(df[y])
        col1 = col1.drop_duplicates()
        print(y)
        print(np.array(col1[y]))
        print("_______________")
        print(col1.shape)
        print("============================================================")
        
def na_value(df,value,col):
    df[col]=df[col].fillna(value) 
    
def na_value_zero(df):
    df=df.fillna(0)
    
def get_success_failure(df,col,outcome):
    print('Dim Of Data')
    print(df.shape)
    col1 = pd.DataFrame(df[col])
    col1 = col1.drop_duplicates()
    for y in col1[col]:
        df2 = df.loc[df[col] == y]
        df3 = df2.loc[df2[outcome] == 1]
        df4 = df2.loc[df2[outcome] == 0]
        r2,c2 = df2.shape
        r3,c3 = df3.shape
        r1,c1 = df4.shape
        print(col+'      '+str(y)+'      No Of data points='+str(r2)+'   no_of_success='+str(r3)+'  no_of_failure='+str(r1))
        
def success_failure(df,outcome):
    print('Dim Of Data')
    r,c= df.shape
    df3 = df.loc[df[outcome] == 1]
    df4 = df.loc[df[outcome] == 0]
    r1,c1 = df3.shape
    r2,c2 = df4.shape
    print('No Of success='+str(r1)+'No Of failure='+str(r2))

def to_float(df,col):
    for y in col:
        df[y] = df[y].astype('float')
        
def to_category(df,col):
    for y in col:
        df[y] = df[y].astype('category')
 
col_float = ['Age', 'Experience', 'Income', 'Family', 'CCAvg',
       'Mortgage']
col_cat = ['Education', 'Personal Loan', 'Securities Account', 'CD Account',
       'Online', 'CreditCard']
to_float(data,col_float)

to_category(data,col_cat)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, auc, roc_curve, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
#import parfit.parfit as pf

import math
from IPython.display import display
from pandas.api.types import is_string_dtype, is_numeric_dtype

import seaborn as sns

from scikitplot.metrics import plot_roc_curve

import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.utils import shuffle
import sklearn
print (sklearn.__version__)
def test_train(data, t, target):  
   # Shuffle the dataset to mix up the rows
    data = shuffle(data, random_state=10)
    train_data, test_data = train_test_split(data, test_size=t, random_state=10)
     
     # Convert into Arrays and Encode
    train_x = train_data.drop(target, axis=1)
    train_y = train_data[target]
    test_x = test_data.drop(target, axis=1)
    test_y = test_data[target]
    print("Train_x Shape :: ", train_x.shape)
    print("Train_y Shape :: ", train_y.shape)
    print("Test_x Shape :: ", test_x.shape)
    print("Test_y Shape :: ", test_y.shape)
    return train_x, train_y, test_x, test_y, train_data, test_data

def random_forest_classifier(features, target):
    """
    To train the random forest classifier with features and target data
    :param features:
    :param target:
    :return: trained random forest classifier
    """
    clf = RandomForestClassifier()
    clf.fit(features, target)
    return clf

def print_score(m, roc_auc=False, clf_report=False, cnf_mat=False):
    print("Training score: {}".format(m.score(train_x, train_y)))
    print("Test score: {}".format(m.score(test_x, test_y)))
    
    pred = m.predict(test_x)
    
    if hasattr(m, 'oob_score_'):
        print("OOB score: {}".format(m.oob_score_))

    if clf_report:
        print(classification_report(test_y, pred))

    if roc_auc:
        fpr, tpr, thresholds = roc_curve(test_y, m.predict_proba(test_x)[:, 1])
        print("ROC AUC score: {}".format(auc(fpr, tpr)))
        plot_roc_curve(test_y, m.predict_proba(test_x))
        plt.show()
        
    if cnf_mat:
        cm = pd.DataFrame(confusion_matrix(test_y, pred))
        sns.heatmap(cm, annot=True)
        plt.show()
        print(pd.crosstab(test_y, pred,
                          rownames=['True'], colnames=['Predicted'], margins=True))

def rf_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}).sort_values('imp', ascending=False)

def plot_fi(fi): 
    return fi.plot('cols', 'imp', 'barh', figsize=(12,7), legend=False)
train_x, train_y, test_x, test_y, train_data, test_data = test_train(data, .2, 'Personal Loan')
# Create random forest classifier instance
trained_model = random_forest_classifier(train_x, train_y)
print("Trained model :: ", trained_model)
predictions = trained_model.predict(test_x)
 
for i in range(0,10):
    print("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i]))
print("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
print("Test Accuracy  :: ", accuracy_score(test_y, predictions))
print(" Confusion matrix ", confusion_matrix(test_y, predictions))
m=trained_model
print_score(m,roc_auc=True)
import seaborn as sns

# Predictions
pred = m.predict(test_x)
# Confidence of predictions
pred_proba = m.predict_proba(test_x)

# Confusion matrix
cm = pd.DataFrame(confusion_matrix(test_y, predictions))

sns.heatmap(cm, annot=True)
plt.show()

print(pd.crosstab(test_y, pred, rownames=['True'], colnames=['Predicted'], margins=True))
m = RandomForestClassifier(n_jobs=-1, class_weight='balanced', random_state=1)
%time m.fit(train_x, train_y);
# Accuracy scores on training and validation data
print_score(m)

fi = rf_feat_importance(m, train_x)
print(fi)
plot_fi(fi);
from sklearn import ensemble

def g_b_classifier(features, target, params):
    """
    To train the gb classifier with features and target data
    :param features:
    :param target:
    :return: trained gb classifier
    """
    clf = ensemble.GradientBoostingClassifier(**params)
    clf.fit(features, target)
    return clf
#define parameters
params = {'n_estimators': 1200, 'max_depth': 3, 'subsample': 0.5,
          'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}

# Create random forest classifier instance
trained_model = g_b_classifier(train_x, train_y, params)
print("Trained model :: ", trained_model)
predictions = trained_model.predict(test_x)
 
for i in range(0, 5):
    print("Actual outcome :: {} and Predicted outcome :: {}".format(list(test_y)[i], predictions[i]))
print("Train Accuracy :: ", accuracy_score(train_y, trained_model.predict(train_x)))
print("Test Accuracy  :: ", accuracy_score(test_y, predictions))
print(" Confusion matrix ", confusion_matrix(test_y, predictions))
def gb_feat_importance(m, df):
    return pd.DataFrame({'cols':df.columns, 'imp':m.feature_importances_}).sort_values('imp', ascending=False)

from sklearn.metrics import accuracy_score, auc, roc_curve, classification_report, confusion_matrix, roc_auc_score
from scikitplot.metrics import plot_roc_curve
m=trained_model
print_score(m,roc_auc=True)
import seaborn as sns

# Predictions
pred = m.predict(test_x)
# Confidence of predictions
pred_proba = m.predict_proba(test_x)

# Confusion matrix
cm = pd.DataFrame(confusion_matrix(test_y, predictions))

sns.heatmap(cm, annot=True)
plt.show()

print(pd.crosstab(test_y, pred, rownames=['True'], colnames=['Predicted'], margins=True))
%time m.fit(train_x, train_y);
# Accuracy scores on training and validation data
print_score(m)
fi = rf_feat_importance(m, train_x)
print(fi)
plot_fi(fi);
# Import Packages
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
# Define the Encoder Function
def one_hot_encode(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return(one_hot_encode)


# Define the Model
def multilayer_perceptron(x, weights, biases):
    
    # Hidden Layer with RELU Activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    
    # Hidden Layer with Sigmoid Activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    
    # Hidden Layer with Sigmoid Activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)
    
    # Hidden Layer with Sigmoid Activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)
    
    # Output Layer with Linear Activation
    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return(out_layer)
# Convert into Arrays and Encode
train_x = train_x.values

test_x = test_x.values


# Encode the Dependent Variable
train_y = one_hot_encode(train_y)
test_y = one_hot_encode(test_y)

# Inspect the Shape of the Training and Testing
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
print(test_y.shape)
# Define the important parameters and variables to work with the tensors
learning_rate = 0.1
training_epochs = 1000
cost_history = np.empty(shape=[1], dtype=float)
n_dim = train_x.shape[1]
print("n_dim", n_dim)
n_class = 2
#|model_path = outpath + '/ann_model'

# Define the number of hidden layers and number of neurons for each layer
n_hidden_1 = 60
n_hidden_2 = 60
n_hidden_3 = 60
n_hidden_4 = 60

graph = tf.get_default_graph()

x = tf.placeholder(tf.float32, [None, n_dim])
W = tf.Variable(tf.zeros([n_dim, n_class]))
b = tf.Variable(tf.zeros([n_class]))
y_ = tf.placeholder(tf.float32, [None, n_class])

# Define the Weights and the Biased for Each Layer
weights = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_4, n_class])),
}

biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden_2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden_3])),
    'b4': tf.Variable(tf.truncated_normal([n_hidden_4])),
    'out': tf.Variable(tf.truncated_normal([n_class])),
}
# Initialize All the Variables
init = tf.global_variables_initializer()
saver = tf.train.Saver(save_relative_paths=True)

# Call the Model
y = multilayer_perceptron(x, weights, biases)

# Define the cost function and optimizer
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

# Open a Session
sess = tf.Session()
sess.run(init)

# Compute the Cost and the Accuracy for each Epoch
mse_history = []
accuracy_history = []

for epoch in range(training_epochs):
    sess.run(training_step, feed_dict={x: train_x, y_: train_y})
    cost = sess.run(cost_function, feed_dict={x: train_x, y_: train_y})
    cost_history = np.append(cost_history, cost)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    pred_y = sess.run(y, feed_dict={x: test_x})
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    mse_ = sess.run(mse)
    mse_history.append(mse_)
    accuracy = (sess.run(accuracy, feed_dict={x: train_x, y_: train_y}))
    accuracy_history.append(accuracy)
    
    print('Epoch:', epoch, '- Cost:', cost, '- MSE:', mse_, '- Train Accuracy:', accuracy)
# Plot MSE and Accuracy Graph
plt.plot(mse_history, 'r')
plt.title('MSE History over Epochs')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.show()

plt.plot(cost_history)
plt.title('Cost History over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Cost')
plt.show()

plt.plot(accuracy_history)
plt.title('Accuracy History over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.show()

# Print the Final Accuracy
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Test Accuracy:", (sess.run(accuracy, feed_dict={x:test_x, y_: test_y})))

# Print the Final Mean SQuare Error
pred_y = sess.run(y, feed_dict={x: test_x})
mse = tf.reduce_mean(tf.square(pred_y - test_y))
print("MSE: %.4f" % sess.run(mse))
#train_x = biased_modeling_data.drop('IsCancelled', axis=1).values
#train_y = biased_modeling_data['IsCancelled']
def read_dataset(df):
    X=df.drop('Personal Loan', axis=1).values
    y1=df['Personal Loan']
    #encode the dependent variables
    encoder = LabelEncoder()
    encoder.fit(y1)
    y = encoder.transform(y1)
    Y = one_hot_encode(y)
    print (X.shape)
    return(X,Y,y1)

#read the dataset
X,Y,y1 = read_dataset(test_data)
#initialize all the variables

init = tf.global_variables_initializer()
saver = tf.train.Saver()
sess = tf.Session()
sess.run(init)
#saver.restore(sess)
prediction = tf.argmax(y,1)
correct_prediction=tf.equal(prediction,tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
actual = []
predicted = []
   
for i in range(1,1000):
    prediction_run=sess.run(prediction,feed_dict={x:X[i].reshape(1,11)})
    print("Original Class:",y1[i],"Pedicted_class:",prediction_run[0])
    actual.append(y1[i])
    predicted.append(prediction_run[0])
    
# Confusion matrix
cm = pd.DataFrame(confusion_matrix(actual, predicted))

sns.heatmap(cm, annot=True)
plt.show()

#print(pd.crosstab(actual, predicted, rownames=['True'], colnames=['Predicted'], margins=True))
