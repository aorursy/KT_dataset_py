import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import average_precision_score,confusion_matrix,precision_recall_curve

from sklearn.model_selection import train_test_split,GridSearchCV

import itertools

### Load and data ###

data=pd.read_csv("../input/creditcard.csv",sep=',')

labels=np.array(data[:]['Class'])

features=np.array(data.iloc[:,1:30]) #ommit columns 'time' and 'Class'



#Print some statistics

num_samples,num_features=features.shape

num_frauds=data[data['Class']==1].shape[0]

print("Data set consists of {0} samples with {1} features. Only {2} of these samples are frauds"

      .format(num_samples,num_features,num_frauds))



# Split data into test and trining data

X_train, X_test, y_train, y_test = train_test_split(features,labels, test_size=0.25,stratify=labels)


# Fit training data with a  Logistic Regression classifier, using several regularization parameters C

# As a scoring metric, the area under the precision recall curve is used (detailed later)



params=[{'C':[1,10,100]}]

clf=GridSearchCV(LogisticRegression(),params,scoring='roc_auc')

clf.fit(X_train, y_train)



print("Best parametrers found: {}".format(clf.best_params_))



## Store the predictions and the probabilities (of the transaction being labeld fraud) of the best model

predictions_logreg=clf.best_estimator_.predict(X_test)

probabilities_logreg=clf.best_estimator_.predict_proba(X_test)
def plot_confusion_matrix(cm, classes, title='Confusion matrix',cmap=plt.cm.Blues):

    """

    This function prints and plots the confusion matrix -> taken from official scikit-learn webpage

    """

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45)

    plt.yticks(tick_marks, classes)



    thresh = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(j, i, cm[i, j],

                 horizontalalignment="center",

                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()

    plt.ylabel('True label')

    plt.xlabel('Predicted label')



cnf_matrix=confusion_matrix(y_test,predictions_logreg)     

fig,ax = plt.subplots()

plot_confusion_matrix(cnf_matrix, classes=["Regular","Fraud"],

                      title='Confusion Matrix for Logistic Regression')

plt.show()
def prc(y_test,probabilities,title=''):

    """ Plots the Recall against Precision curve and outputs the area under the Curve"""

    precision,recall,thresholds = precision_recall_curve(y_test,probabilities)

    plt.title(title)

    ax.plot(recall,precision)

    ax.set_xlabel('Recall')

    ax.set_ylabel('Precision')

    return average_precision_score(y_test,probabilities)

    

fig,ax=plt.subplots()      

area_logreg=prc(y_test,probabilities_logreg[:,1],

               title='Precision Recall Curve Logistic Regression')

plt.show()

print("Area under precision recall graph: {0}".format(area_logreg))



import tensorflow as tf

#Helper Functions

def weight_variable(shape,name=None):

    initial=tf.truncated_normal(shape,stddev=0.1)

    return tf.Variable(initial,name)



def bias_variable(shape,name=None):

    initial=tf.constant(0.1,shape=shape)

    return tf.Variable(initial,name)

# Data Preparation



# One hot encode labels

y_train_hot=np.eye(2)[y_train]

y_test_hot=np.eye(2)[y_test] 

# Filter out only fraudulent cases from training data

X_train_pos=X_train[y_train_hot[:,1]==1] 
### Set up Graph



# Hyperparameters

n_epoch=20

batch_size=96

lr=0.001

num_pos=32



##One hot encode labels



with tf.name_scope("Data"):

    X=tf.placeholder(tf.float32,shape=[None,num_features],name="X-input")

    Y=tf.placeholder(tf.float32,shape=[None,2],name="Y-input")

    

with tf.name_scope("Layer_1"):

    W1=weight_variable([num_features,20],name="Weights")

    b1=bias_variable([20],name="biases")

    h1=tf.nn.sigmoid(tf.matmul(X,W1)+b1)

   

with tf.name_scope("Output_Layer"):

    W2=weight_variable([20,2],name="Weights3")

    b2=bias_variable([2],name="biases3")

    h2=tf.matmul(h1,W2)+b2



with tf.name_scope("Cross_Entropy"):

    logits=tf.nn.softmax(h2)

    cross_entropy=tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=logits)

    

with tf.name_scope("Training"):

    train_op=tf.train.AdamOptimizer(lr).minimize(cross_entropy)



with tf.name_scope("Evaluation"):

    pred=tf.argmax(logits,1)

    correct_prediction = tf.equal(pred, tf.argmax(Y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))





def test_model(X_input,y_input,sess):

    """Tests the current model on a given input (X_input,y_input). Returns the accuracy

    as well as the predictions and probabilities"""

    n_batches=int(y_input.shape[0]/batch_size)

    avg_sum=0

    probabilities=np.zeros((y_input.shape))

    predictions=np.zeros(y_input.shape[0])

    for i in range(n_batches):

        X_batch=X_input[i*batch_size:(i+1)*batch_size]

        Y_batch=y_input[i*batch_size:(i+1)*batch_size]

        avg_sum += sess.run(accuracy,feed_dict={X:X_batch,Y:Y_batch})

        # Calculate and store predictions and probabilities

        preds=sess.run(pred,feed_dict={X:X_batch,Y:Y_batch})

        probs=sess.run(logits,feed_dict={X:X_batch,Y:Y_batch})

        predictions[i*batch_size:(i+1)*batch_size]=preds

        probabilities[i*batch_size:(i+1)*batch_size]=probs

        acc=avg_sum/n_batches

    return (acc,predictions,probabilities)



    

            
### Train the neural network in a session with subsequent testing



#saver = tf.train.Saver()    



with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    n_batches=int(X_train.shape[0]/batch_size)

    for epoch in range(1,n_epoch+1):

        for i in range(n_batches):

            # Add num_pos fraudulent data points to the batch, chosen at random

            np.random.shuffle(X_train_pos)

            X_batch=np.vstack((X_train[i*batch_size:(i+1)*batch_size],X_train_pos[:num_pos]))

            Y_batch=np.vstack((y_train_hot[i*batch_size:(i+1)*batch_size],np.vstack([0,1] for i in range(num_pos))))

            #Training step

            sess.run(train_op,feed_dict={X:X_batch,Y:Y_batch})

        # Reporting after every 10 full iteration

        if epoch%10==0:

            acc,_,_=test_model(X_train,y_train_hot,sess)

            print("Finished epoch {0} with a training accuracy of {1}".format(epoch,acc))

           

    acc,predictions_nn,probabilities_nn=test_model(X_test,y_test_hot,sess)

    print('Accuracy of test set: {0}'.format(acc))

    #save_path=saver.save(sess,".//CREDIT_CARDS.ckpt")

    #print("Model saved in file: %s" % save_path)

    

### Confusion Matrix for NN

cnf_matrix=confusion_matrix(y_test,predictions_nn)     

fig,ax = plt.subplots()

plot_confusion_matrix(cnf_matrix, classes=["Regular","Fraud"],

                     title='Confusion Matrix NN')

plt.show()
### Precision_Recall graph for NN

fig,ax=plt.subplots()      

area_nn=prc(y_test,probabilities_nn[:,1],title='Precison Recall Curve NN')

plt.show()

print("Area under precision recall graph: {0}".format(area_nn))