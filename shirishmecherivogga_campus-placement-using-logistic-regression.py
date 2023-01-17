import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer,make_column_transformer,make_column_selector
from sklearn.model_selection import train_test_split
import math


#Logistic Sigmoid function
def logisticSigmoid(theta,X):
    return (1 / (1 + np.exp(-(np.dot(X,theta)))))


def get_acc(y_hat,y_test):
    count = len(y_hat)
    score = 0
    for i,j in zip(y_hat,y_test):
        if i == j:
            score+=1
    
    return (score/count)*100

#Binary Cross entropy loss for 2 distributions
    
def cross_loss(p, q):
    return -(np.mean([p[i]*math.log(q[i]) for i in range(len(p))]))

#Derivative of the log likelihood
def loss_derivative(y_true,y_hat,X):
    return np.dot(np.transpose(y_hat-y_true),X)

def KfoldValidation(k,X_train,y_train):
    num_samples = len(X_train) // k
    validation_error = []
    training_error = []
    for i in range(k):
        print("Processing Fold #", i)
        val_data = X_train[i * num_samples:(i+1) * num_samples]
        val_test = y_train[i * num_samples:(i+1) * num_samples]
        partial_train_data = np.concatenate([X_train[:i * num_samples], X_train[(i+1) * num_samples:]],axis=0)
        partial_train_label = np.concatenate([y_train[:i * num_samples], y_train[(i+1) * num_samples:]],axis=0)
        theta,training_loss = gradient_descent(partial_train_data,partial_train_label,1000,0.0005,0)
        y_hat_test = logisticSigmoid(theta,val_data)
        y_hat_train = logisticSigmoid(theta,partial_train_data)
        validation_loss = cross_loss(val_test,y_hat_test)
        training_loss = cross_loss(partial_train_label,y_hat_train)
        validation_error.append(validation_loss)
        training_error.append(training_loss)
    
    return training_error,validation_error  

def gradient_descent(X,y,epoch,lr,verbose):
    theta = np.random.randn(X.shape[1], 1) / 100
    num_e = 1
    loss_epoch = []
    
    while num_e <= epoch:
        
        #Get initial predictions and loss
        y_hat = logisticSigmoid(theta,X)
        loss = cross_loss(y,y_hat)
        #Get derivative of loss
        dloss = loss_derivative(y,y_hat,X)
        #Update the parameters
        theta = theta - lr * np.transpose(dloss)
        
        if verbose == 1:
            print("Processing Epoch ########" ,num_e)
            print("Loss at epoch",num_e," is ",loss)

        loss_epoch.append(loss)
        num_e = num_e+1
    return theta,loss_epoch

dataset = pd.read_csv('../input/factors-affecting-campus-placement/Placement_Data_Full_Class.csv')
 
#since serial number and gender dont matter we remove them from the dataset
X = dataset.iloc[:,2:-2]
y = dataset.iloc[:,-2]

#Using a column transformer method to scale the dataset and convert the certain columns to one hot encoded vectors
ct = make_column_transformer(
       (StandardScaler(),
        make_column_selector(dtype_include=np.number)),  # rating
       (OneHotEncoder(),
     make_column_selector(dtype_include=object)))


X = ct.fit_transform(X)

le = LabelEncoder()
y = le.fit_transform(y)

#Splitting the dataset 1/3 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

y_train = y_train.reshape(y_train.shape[0],1)
y_test = y_test.reshape(y_test.shape[0],1)

#performing Kfold validation
training_loss,validation_loss = KfoldValidation(10, X_train, y_train)
training_loss = np.mean(training_loss)
validation_loss = np.mean(validation_loss)

print("Training loss: ",training_loss)
print("Validation loss: ",validation_loss)

#Performing gradient descent using a specified learning rate and number of epochs
theta,loss = gradient_descent(X_train, y_train, 1000, 0.0005,0)

#Evaluating on test set
y_pred_test = logisticSigmoid(theta, X_test)
for i in range(len(y_pred_test)):
    if y_pred_test[i] > 0.5:
        y_pred_test[i] = 1
    else:
        y_pred_test[i] = 0
        

test_acc = get_acc(y_pred_test, y_test)

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred_test)
print("Confusion Matrix:\n",confusion_matrix)
from sklearn.metrics import classification_report
print("Classification Report:\n",classification_report(y_test, y_pred_test))