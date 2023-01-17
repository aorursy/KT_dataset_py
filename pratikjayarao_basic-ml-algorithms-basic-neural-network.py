# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import random as rnd



# visualization

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



# machine learning

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC, LinearSVC

from sklearn.ensemble import RandomForestClassifier

from sklearn.neighbors import KNeighborsClassifier

from sklearn.naive_bayes import GaussianNB

from sklearn.linear_model import Perceptron

from sklearn.linear_model import SGDClassifier

from sklearn.tree import DecisionTreeClassifier
data1 = pd.read_csv("../input/Iris.csv")
data1.head()
data1['Species']= data1['Species'].map({'Iris-setosa': 0, 'Iris-versicolor' : 1,'Iris-virginica' : 2})

data1.head()
data1['Species'].value_counts()
sns.pairplot(data1.drop("Id", axis=1), hue="Species", size=3)
from pandas.tools.plotting import radviz

radviz(data1.drop("Id", axis=1), "Species")
from pandas.tools.plotting import parallel_coordinates

parallel_coordinates(data1.drop("Id", axis=1), "Species")
from sklearn.model_selection import train_test_split



data, data2 = train_test_split(data1, test_size = 0.25)
trainx=data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

testx=data2[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]

trainy=data['Species']

testy=data2['Species']

trainy.head()
from IPython.core.display import display, HTML

display(HTML('<h1>Support Vector Machines</h1>'))



from sklearn import svm

svc = svm.SVC(C=10, gamma=7, probability=True)  

svc.fit(trainx,trainy)  

print("Training Set Accruracy",svc.score(trainx,trainy)*100)

print("Test Set Accuracy",svc.score(testx,testy))

from IPython.core.display import display, HTML

display(HTML('<h1>Logistic Regression</h1>'))

#LOGISTIC

logreg = LogisticRegression()

logreg.fit(trainx,trainy)

ac=logreg.score(trainx,trainy)

Y_pred = logreg.predict(testx)

acc_log = round(logreg.score(testx,testy) * 100, 2)



print("Training Set Accruracy",logreg.score(trainx,trainy)*100)

print("Test Set Accuracy",logreg.score(testx,testy)*100)

from IPython.core.display import display, HTML

display(HTML('<h1>K-Neighbours</h1>'))

knn = KNeighborsClassifier(n_neighbors = 3)

knn.fit(trainx, trainy)

print("Training Set Accruracy",knn.score(trainx,trainy)*100)

print("Test Set Accuracy",knn.score(testx,testy)*100)

display(HTML('<h1>GAUSSIAN</h1>'))
gaussian = GaussianNB()

gaussian.fit(trainx, trainy)

print("Training Set Accruracy",gaussian.score(trainx,trainy)*100)

print("Test Set Accuracy",gaussian.score(testx,testy)*100)

display(HTML('<h1>PERCEPTRON</h1>'))
perceptron = Perceptron()

perceptron.fit(trainx, trainy)

print("Training Set Accruracy",perceptron.score(trainx,trainy)*100)

print("Test Set Accuracy",perceptron.score(testx,testy)*100)

display(HTML('<h1>SGD</h1>'))
sgd = SGDClassifier()

sgd.fit(trainx, trainy)

print("Training Set Accruracy",sgd.score(trainx,trainy)*100)

print("Test Set Accuracy",sgd.score(testx,testy)*100)

display(HTML('<h1>DECISION TREE</h1>'))
decision_tree = DecisionTreeClassifier()

decision_tree.fit(trainx, trainy)

print("Training Set Accruracy",decision_tree.score(trainx,trainy)*100)

print("Test Set Accuracy",decision_tree.score(testx,testy)*100)

display(HTML('<h1>RANDOM FOREST</h1>'))
random_forest = RandomForestClassifier(n_estimators=100)

random_forest.fit(trainx, trainy)

print("Training Set Accruracy",random_forest.score(trainx,trainy)*100)

print("Test Set Accuracy",random_forest.score(testx,testy)*100)

display(HTML('<h1>NEURAL NETWORKS</h1>'))
data=data.dropna()

X = np.array((data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]))

y = np.array(data['Species'])

y = y.reshape(len(data), 1)

X.shape,y.shape



input_layer_size=4

hidden_layer_size=6

num_labels=3







def sigmoid(z):  

    return 1 / (1 + np.exp(-z))





# In[45]:



def sigmoidGradient(z):

    g=np.matrix(np.zeros(z.shape))

    g=np.multiply(sigmoid(z),(1-sigmoid(z)))

    return g

        

    





# In[46]:



def forwardpropagate(X,theta1,theta2):



   # theta1 = np.matrix(np.reshape(params[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1))))  

    #theta2 = np.matrix(np.reshape(params[hidden_layer_size * (input_layer_size + 1):], (num_labels, (hidden_layer_size + 1))))

    a1=np.matrix(np.insert(X,0,1,axis=1))

    #print("a1:\t"+str(a1.shape))

    z2=np.matmul(a1,theta1.T)

    a2=sigmoid(z2)

    a2=np.insert(a2,0,1,axis=1)

    #print("a2:\t"+str(a2.shape))

    z3=np.matmul(a2,theta2.T)

    a3=sigmoid(z3)

    return a3







# In[47]:



def nnCostRegGrad(params,input_layer_size,hidden_layer_size,num_labels,X,y,lam):

    theta1 = np.matrix(np.reshape(params[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1))))  

    theta2 = np.matrix(np.reshape(params[hidden_layer_size * (input_layer_size + 1):], (num_labels, (hidden_layer_size + 1))))



    #print(theta1.shape)

    #print(theta2.shape)

    m=X.shape[0]

    X=np.matrix(X)

    a1=np.matrix(np.insert(X,0,1,axis=1))

    #print("a1:\t"+str(a1.shape))

    z2=np.matmul(a1,theta1.T)

    a2=sigmoid(z2)

    a2=np.insert(a2,0,1,axis=1)

    #print("a2:\t"+str(a2.shape))

    z3=np.matmul(a2,theta2.T)

    a3=sigmoid(z3)

  

                       

    y2=np.copy(y)        

    

    #print("y2:\t"+str(y2.shape))

    y2=np.matrix(y2)  

    a3=np.matrix(a3)

    a2=np.matrix(a2)

    a1=np.matrix(a1)

    theta1=np.matrix(theta1)

    theta2=np.matrix(theta2)

    #print(y2)

    M=np.multiply(-y2.T,np.log(a3))-np.multiply((1-y2.T),np.log(1-a3))

    l=np.sum(np.sum(M))

    J=l/m

    

    t1=np.power(theta1,2)

    t2=np.power(theta2,2)

    reg1=np.sum(np.sum(t1[:,2:theta1.shape[1]]))

    reg2=np.sum(np.sum(t2[:,2:theta1.shape[1]]))

    

    reg=reg1+reg2

    

    finreg=(lam*reg)/(2*m)

    J+=finreg

    

    

    

    #print(J)

    yy=np.matrix(y2.T)

    b=np.matrix(np.insert(X,0,1,axis=1))

    findel1=np.matrix(np.zeros(theta1.shape))

    findel2=np.matrix(np.zeros(theta2.shape))

    #print(findel1.shape)

    #print(findel2.shape)

    print("Iterating")

    for t in range(X.shape[0]):

        aa1=np.matrix(b[t])

        

        zz2=np.matmul(aa1,theta1.T)

        aa2=sigmoid(zz2)

        aa2=np.insert(aa2,0,1,axis=1)

        

        #print("aa2:\t"+str(aa2.shape))

        zz3=np.matmul(aa2,theta2.T)

        aa3=sigmoid(zz3)

        #print(aa3)

        #print(yy[t].shape)

        d3=aa3-yy[t]

        #print("d3:\t"+str(d3.shape))

        d2=np.multiply(np.matmul(d3,theta2),sigmoidGradient(np.insert(zz2,0,1)))

        #print("d2:\t"+str(d2.shape))

        #print(findel1.shape)

        #print("d2 before:\t"+str(d2.shape))

        d2=d2[:,1:]

     #   print("d2:\t"+str(d2.shape))

        findel1=findel1+np.matmul(d2.T,aa1)

        #print(findel1)

        findel2=findel2+np.matmul(d3.T,aa2)

        #print(findel2)

    Theta1_grad=np.matrix(np.zeros(theta1.shape))

    Theta2_grad=np.matrix(np.zeros(theta2.shape))

    Theta1_grad=findel1/m

    Theta2_grad=findel2/m

    

    #print(Theta1_grad.shape,Theta2_grad.shape)

    

    Theta1_grad[:,1:Theta1_grad.shape[1]]=Theta1_grad[:,1:]+((lam*theta1[:,1:]))/m

    Theta2_grad[:,1:Theta2_grad.shape[1]]=Theta2_grad[:,1:]+((lam*theta2[:,1:]))/m

    

    grad = np.concatenate((np.ravel(Theta1_grad), np.ravel(Theta2_grad)))

    

    return J,grad





y2=np.zeros((num_labels,X.shape[0]))

#print(y2.shape,X.shape)

#print(y)    

for i in range(0,X.shape[0]):

    for j in range(0,num_labels):

        if(y[i]==j):

            y2[j][i]=1

print(y2)

params = (np.random.random(size=hidden_layer_size * (input_layer_size + 1) + num_labels * (hidden_layer_size + 1)) - 0.5) * 0.25



# unravel the parameter array into parameter matrices for each layer

J,grad=nnCostRegGrad(params,input_layer_size,hidden_layer_size,num_labels,X,y2,lam=1)

#y





# In[49]:



from scipy.optimize import minimize

lam=1

# minimize the objective function

fmin = minimize(fun=nnCostRegGrad, x0=params, args=(input_layer_size, hidden_layer_size, num_labels, X, y2,lam),  

                method='TNC', jac=True, options={'maxiter': 250})

fmin





# In[50]:



X = np.matrix(X)  

theta1 = np.matrix(np.reshape(fmin.x[:hidden_layer_size * (input_layer_size + 1)], (hidden_layer_size, (input_layer_size + 1))))  

theta2 = np.matrix(np.reshape(fmin.x[hidden_layer_size * (input_layer_size + 1):], (num_labels, (hidden_layer_size + 1))))



h = forwardpropagate(X,theta1,theta2)  

y_pred = np.array(np.argmax(h, axis=1))  

#y,y_pred





# In[51]:



correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]  

accuracy = (sum(map(int, correct)) / float(len(correct)))  

print ('Training Set accuracy = {0}%'.format(accuracy * 100))
fnx2 = np.array((data2[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]))
h2 = forwardpropagate(fnx2,theta1,theta2)  

print(h2.shape)

y_pred2 = np.array(np.argmax(h2, axis=1))  

y2,y_pred2

fny = testy.reshape(len(data2), 1)
correct = [1 if a == b else 0 for (a, b) in zip(y_pred2, fny)]  

accuracy = (sum(map(int, correct)) / float(len(correct)))  

print ('Test Set accuracy = {0}%'.format(accuracy * 100))