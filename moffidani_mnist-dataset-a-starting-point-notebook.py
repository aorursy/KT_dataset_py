# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#Importing the data as pandas dataframe. We won't use the test.csv file, as we are
#concerned with supervised learning here. We rather divide train.csv into train and test set
#each example then having its own target value known. 

train = pd.read_csv('../input/train.csv')
#test = pd.read_csv('../input/test.csv')
train.shape
train.head()
# We select 5 random examples (rows) from the train dataset
random_indexes = random.sample(range(train.shape[0]),5)
#The original images are obtained by reshaping the rowos
original_images = [np.array(train.iloc[element,1:]).reshape(28,28) for element in random_indexes]

#The 784-dimensional array form is instead
array_representation = [np.array(train.iloc[element,1:]) for element in random_indexes]
#Visualizing the orginal images
fig, axes = plt.subplots(nrows=1, ncols=5)

i=0
for ax in axes:
    ax.imshow(original_images[i], cmap ='gist_gray')
    i +=1
fig.tight_layout()
#Visualizing their 784-dimensional array form

fig, axes = plt.subplots(nrows=1, ncols=5)

i=0
for ax in axes:
    ax.imshow(array_representation[0].reshape(784,1), aspect = 0.02,  cmap='gist_gray')
    i +=1
fig.tight_layout()
train.describe()
train.describe().loc['mean'].idxmax()
temp_im = np.zeros(shape=(28,28))
temp_im[407//28, 407%28] = 100

sns.heatmap(temp_im, cmap ='gray')
plt.show()
train.groupby('label').pixel0.count().plot.bar()
plt.show()
#Normalizes each pixel column [-1,1], taking into account that some column is completely filed with 0s
def feat_normalize(X):
    M = X.shape[1]
    for i in range(M):
        if np.any(X[:,i]) != 0:
            min_ = X[:,i].min()
            max_ = X[:,i].max()
            X[:,i] =(2*X[:,i]-min_-max_)/(max_-min_)
            

            
def append_ones(X):
    
    s = X.shape[0]
    
    ones = np.ones(shape=(s,1))
    
    return np.concatenate((ones, X), axis=1)

            
#functions to calculate precision, recall and F1 score of a model
            

def prec_rec_F1(class_rep):
    precision = []
    recall = []
    F1 = []

    for i in range(10):
        temp = np.zeros(shape=(2,2))
        temp[0,0] = class_rep.iloc[i,i]
        temp[0,1] = sum(class_rep.iloc[i,:i]) + sum(class_rep.iloc[i,i+1:])
        temp[1,0] = sum(class_rep.iloc[:i,i]) + sum(class_rep.iloc[i+1:, i])
        temp[1,1] = sum(np.diag(class_rep))- class_rep.iloc[i,i]
    
        ptemp = temp[0,0]/(temp[0,0]+ temp[0,1])
        precision.append([i,ptemp])
        rectemp = temp[0,0]/(temp[0,0]+ temp[1,0])
        recall.append([i,rectemp])
        F1.append([i,2 * ptemp * rectemp /(ptemp+rectemp)])
    
    return [precision, recall, F1]

def create_class_rep(prediction, y_test):
    class_rep =np.zeros(shape=(10,10))
    
    for i in range(len(y_test)):
        x = prediction[i]
        y = y_test[i]
        class_rep[x,y] +=1
        
    class_rep = pd.DataFrame(class_rep)
    return class_rep.applymap(int)




#Sigmoid function
def sigmoid(x):
    return 1/(1 + np.exp(-x))

#Cost function of the logistic regression for binary classification, s_i = {0,1} 
def cost(X, y , theta):
    dim = X.shape[0]
    s = sigmoid(np.dot(X,theta))
    tot = -(np.log(s)*y +np.log(1-s)*(1-y))
    return 1/dim *sum(tot)[0]
    
#Gradient of the cost function with respect to the parameters theta. To be used in gradient descent below
def grad_cost(X, y, theta):
    
    dim = X.shape[0]
    pred = sigmoid(np.dot(X,theta))
    c1 = 1/dim * np.transpose(pred-y)
    return np.transpose(np.dot(c1,X))

#Gradient descent to get the parameter theta
def grad_descent(X, y, theta, learning_par, num_iter):

    for i in range(num_iter):
        #print cost(X,y,theta) to check the cost is monotonically decreasing at each iteration
        theta = theta - learning_par*grad_cost(X,y,theta)
        
    return theta




#Dividing the training set in train and test set

y = train.iloc[:,0]
X = train.iloc[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state = 101)

#Normalizing the train and test sets
X_train = np.array(X_train)
feat_normalize(X_train)
X_train = append_ones(X_train)


#Appending the bias column to the train and test matrices

X_test = np.array(X_test)
feat_normalize(X_test)
X_test = append_ones(X_test)





#Create the vector of target lables for each digit 0-9
y_target = []
for i in range(10):
    y_target.append(y_train.apply(lambda x: 1 if x == i else 0))
    
#Initialize the list of training parameters (784+1 (bias) for each digit)
theta=[]

#Gradient descent to train the model
for i in range(10):
    ytemp = np.array(y_target[i])
    ytemp = ytemp.reshape(y_train.shape[0],1)

    thetatemp = np.zeros(shape=(X_train.shape[1],1))

    alpha = 0.03
    n_iter = 100

    thetatemp = grad_descent(X_train,ytemp,thetatemp,alpha,n_iter)
    theta.append(thetatemp)
    print('{}: done!'.format(i))
plt.imshow(theta[0][1:].reshape(28,28), cmap='gist_gray')
plt.show()
plt.imshow(theta[1][1:].reshape(28,28), cmap='gist_gray')
plt.show()
result = [sigmoid(np.dot(X_test,theta[i])) for i in range(10)]
result = np.transpose(np.array(result)).reshape(X_test.shape[0],10)

prediction = (np.array([element.argmax() for element in result])).reshape(X_test.shape[0],1)
#testing accuracy of the prediction
y_test = np.array(y_test)
y_test = y_test.reshape(y_test.shape[0],1)


accuracy = sum(prediction == y_test)[0]/(y_test.shape[0])
print('Accuracy is: {}%'.format(accuracy))
class_rep = create_class_rep(prediction,y_test)
class_rep
precision, recall, F1 = prec_rec_F1(class_rep)
plt.figure(figsize=(8,8))

plt.xticks(range(10))
plt.yticks(1/10*np.array(range(10)))

plt.bar(np.transpose(precision)[0],np.transpose(precision)[1], align='edge', width =-0.25)
plt.bar(np.transpose(recall)[0],np.transpose(recall)[1],align='center',width = 0.25)
plt.bar(np.transpose(F1)[0],np.transpose(F1)[1],align='edge',width =0.25)
plt.legend(labels = ('Precision','Recall','F1'))

plt.tight_layout()
from sklearn.ensemble import RandomForestClassifier
#Create a forest with n=100 trees and fot to the model
forest = RandomForestClassifier(n_estimators=100)
forest.fit(X_train, y_train)
#Predicting new results
prediction = forest.predict(X_test)
prediction = prediction.reshape(prediction.shape[0],1)
#Accuracy
accuracy = sum(prediction == y_test)[0]/(y_test.shape[0])
print('Accuracy is: {}%'.format(accuracy))
#Classification report
class_rep = create_class_rep(prediction,y_test)
class_rep
#And precision, recall, F1

precision, recall, F1 = prec_rec_F1(class_rep)

plt.xticks(range(10))
plt.yticks(1/10*np.array(range(10)))

plt.bar(np.transpose(precision)[0],np.transpose(precision)[1], align='edge', width =-0.25)
plt.bar(np.transpose(recall)[0],np.transpose(recall)[1],align='center',width = 0.25)
plt.bar(np.transpose(F1)[0],np.transpose(F1)[1],align='edge',width =0.25)
plt.legend(labels = ('Precision','Recall','F1'))

plt.tight_layout()


from sklearn.naive_bayes import GaussianNB
#Create an instance of the NB algorothm and fit to the training set
classifier = GaussianNB()
classifier.fit(X_test,y_test)
#Predicting new results
prediction = classifier.predict(X_test)
prediction = prediction.reshape(prediction.shape[0],1)
accuracy = sum(prediction == y_test)[0]/(y_test.shape[0])
print('Accuracy is: {}%'.format(accuracy))
#Classification report
class_rep = create_class_rep(prediction,y_test)
class_rep
#Number of examples for each class
class_rep['sum'] = class_rep.sum()

#correctly classified examples for each class
diag = pd.Series([class_rep.iloc[i,i] for i in range(10)])

#Misclassification rate
class_rep['misclass_rate'] = 1 - diag / class_rep['sum'] 

class_rep.drop('sum', axis=1)
#And precision, recall, F1

precision, recall, F1 = prec_rec_F1(class_rep)

plt.xticks(range(10))
plt.yticks(1/10*np.array(range(10)))

plt.bar(np.transpose(precision)[0],np.transpose(precision)[1], align='edge', width =-0.25)
plt.bar(np.transpose(recall)[0],np.transpose(recall)[1],align='center',width = 0.25)
plt.bar(np.transpose(F1)[0],np.transpose(F1)[1],align='edge',width =0.25)
plt.legend(labels = ('Precision','Recall','F1'))

plt.tight_layout()



from sklearn.svm import LinearSVC
#Create an instance of SVM and cfit to the training set
classifier = LinearSVC()
classifier.fit(X_train, y_train)
#Predicting
prediction = classifier.predict(X_test)
prediction = prediction.reshape(prediction.shape[0],1)
#Accuracy
accuracy = sum(prediction == y_test)[0]/(y_test.shape[0])
print('Accuracy is: {}%'.format(accuracy))
class_rep = create_class_rep(prediction,y_test)
class_rep
#And precision, recall, F1

precision, recall, F1 = prec_rec_F1(class_rep)

plt.xticks(range(10))
plt.yticks(1/10*np.array(range(10)))

plt.bar(np.transpose(precision)[0],np.transpose(precision)[1], align='edge', width =-0.25)
plt.bar(np.transpose(recall)[0],np.transpose(recall)[1],align='center',width = 0.25)
plt.bar(np.transpose(F1)[0],np.transpose(F1)[1],align='edge',width =0.25)
plt.legend(labels = ('Precision','Recall','F1'))

plt.tight_layout()


temp_im = np.zeros(shape=(28,28))
temp_im[407//28, 407%28] = 100
temp_im[380//28, 380%28] = 100

sns.heatmap(temp_im, cmap ='gray')
plt.show()
#Calculating the covariance matrix, its eigenvalues and eigenvectors
X_train_trans = np.transpose(X_train)
covariance = np.dot(X_train_trans,X_train)
eigenval, eigenvec = np.linalg.eig(covariance)
#We pick the first two main eigenvectors. This comes with some advantage concerning visualization.
#(note their are sorted in descnding order with respect to their associated eigenvalues).
#Note the imaginary part of the components of those vector is zero.
#We take the real part just not ot get any error in the following
PCA_mat = np.real(eigenvec[:,0:2])
plt.figure(figsize=(10,10))
sns.heatmap(PCA_mat, cmap = 'inferno')
plt.show()
X_transf = np.dot(X_train, PCA_mat)
plt.scatter(X_transf[:,0],X_transf[:,1], c = y_train, cmap ='plasma')
plt.show()
y_target = []
for i in range(10):
    y_target.append(y_train.apply(lambda x: 1 if x == i else 0))
    
#Initialize the list of training parameters (784+1 (bias) for each digit)
theta=[]

#Gradient descent to train the model
for i in range(10):
    ytemp = np.array(y_target[i])
    ytemp = ytemp.reshape(y_train.shape[0],1)

    thetatemp = np.zeros(shape=(X_transf.shape[1],1))

    alpha = 0.03
    n_iter = 400

    thetatemp = grad_descent(X_transf,ytemp,thetatemp,alpha,n_iter)
    theta.append(thetatemp)
    print('{}: done!'.format(i))
X_test_transf = np.dot(X_test, PCA_mat)
result = [sigmoid(np.dot(X_test_transf,theta[i])) for i in range(10)]
result = np.transpose(np.array(result)).reshape(X_test.shape[0],10)

prediction = (np.array([element.argmax() for element in result])).reshape(X_test_transf.shape[0],1)
y_test = np.array(y_test)
y_test = y_test.reshape(y_test.shape[0],1)


accuracy = sum(prediction == y_test)[0]/(y_test.shape[0])
print('Accuracy is: {}%'.format(accuracy))
for k in range(X_train.shape[1]):
    if 1-sum(eigenval[:k])/sum(eigenval) < 0.01:
        print('{} principal components needed'.format(k))
        break
    