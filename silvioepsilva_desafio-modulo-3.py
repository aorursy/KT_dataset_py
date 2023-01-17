import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import random
%matplotlib inline

import os
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# Importando dados de treino
df_train = pd.read_csv('../input/mnist-in-csv/mnist_test.csv')
train_instancias, train_atributos = df_train.shape
print('O dataset de treino possui {} instâncias e {} atributos.'.format(train_instancias,train_atributos))
# Importando dados de test
df_test  = pd.read_csv('../input/mnist-in-csv/mnist_train.csv')
test_instancias, test_atributos = df_test.shape
print('O dataset de treino possui {} instâncias e {} atributos.'.format(test_instancias,test_atributos))
df_train.head()
# Renomeando colunas
df_train.columns = ['pixel'+ str(i) for i in range(0, 785)]
df_train.rename(columns={'pixel0':'label'}, inplace=True)
df_train.describe()
# Selecionando 5 imagens aleatórias para visualizá-las
random_indexes = random.sample(range(df_train.shape[0]),5)
original_images = [np.array(df_train.iloc[element,1:]).reshape(28,28) for element in random_indexes]
array_representation = [np.array(df_train.iloc[element,1:]) for element in random_indexes]

# Visualizando
fig, axes = plt.subplots(nrows=1, ncols=5)

i=0
for ax in axes:
    ax.imshow(original_images[i], cmap ='gist_gray')
    i +=1

fig.tight_layout()
# Criando algumas funções

# Normalização
def feat_normalize(X):
    
    M = X.shape[1]
    for i in range(M):
        if np.any(X[:,i]) != 0:
            min_ = X[:,i].min()
            max_ = X[:,i].max()
            X[:,i] =(2*X[:,i]-min_-max_)/(max_-min_)
# -----------------------------------------------------            
            
def append_ones(X):
    
    s = X.shape[0]
    ones = np.ones(shape=(s,1))
    return np.concatenate((ones, X), axis=1)

# -----------------------------------------------------
            
# funções para calcular precisão, recall e f1_score
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

# -----------------------------------------------------

def create_class_rep(prediction, y_test):
    
    class_rep =np.zeros(shape=(10,10))
    
    for i in range(len(y_test)):
        x = prediction[i]
        y = y_test[i]
        class_rep[x,y] +=1
        
    class_rep = pd.DataFrame(class_rep)
    return class_rep.applymap(int)

# -----------------------------------------------------

# Sigmoid function
def sigmoid(x):
    
    return 1/(1 + np.exp(-x))

# -----------------------------------------------------

# Cost function of the logistic regression for binary classification, s_i = {0,1} 
def cost(X, y , theta):
    
    dim = X.shape[0]
    s = sigmoid(np.dot(X,theta))
    tot = -(np.log(s)*y +np.log(1-s)*(1-y))
    return 1/dim *sum(tot)[0]

# -----------------------------------------------------

# Gradient of the cost function with respect to the parameters theta. To be used in gradient descent below
def grad_cost(X, y, theta):
    
    dim = X.shape[0]
    pred = sigmoid(np.dot(X,theta))
    c1 = 1/dim * np.transpose(pred-y)
    return np.transpose(np.dot(c1,X))

# -----------------------------------------------------

# Gradient descent to get the parameter theta
def grad_descent(X, y, theta, learning_par, num_iter):

    for i in range(num_iter):
        #print cost(X,y,theta) to check the cost is monotonically decreasing at each iteration
        theta = theta - learning_par*grad_cost(X,y,theta)
        
    return theta
#Dividing the training set in train and test set

y = df_train.iloc[:,0]
X = df_train.iloc[:,1:]

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

# Visualizando os dados classificados

plt.imshow(theta[0][1:].reshape(28,28), cmap='gist_gray')
plt.show()
# Predição
result = [sigmoid(np.dot(X_test,theta[i])) for i in range(10)]
result = np.transpose(np.array(result)).reshape(X_test.shape[0],10)

prediction = (np.array([element.argmax() for element in result])).reshape(X_test.shape[0],1)

#testing accuracy of the prediction
y_test = np.array(y_test)
y_test = y_test.reshape(y_test.shape[0],1)

accuracy = sum(prediction == y_test)[0]/(y_test.shape[0])
print('Accuracy is: {}'.format(accuracy))
# Matrix de classificação
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
print('Accuracy is: {}'.format(accuracy))
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