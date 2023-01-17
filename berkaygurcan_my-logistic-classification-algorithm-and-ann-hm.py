# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
data=pd.read_csv("../input/voice.csv")
data.head(7)
data.describe()
data.columns
data.info()
data.corr()
#correlation map
f,ax = plt.subplots(figsize=(20, 20))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
data.label=[0 if each=="female" else 1 for each in data.label]
#print(data.info()) #2 adet classimiz oldu 1 erkek 0 bayan

y=data.label.values
x_data=data.drop(["label"],axis=1)

#%% normalization
# (x-max)/(max-min)
x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data)).values
#%% train test and split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

x_train=x_train.T
x_test=x_test.T
y_train=y_train.T
y_test=y_test.T

print("x train : ",x_train.shape)
print("x test : ",x_test.shape)
print("y train : ",y_train.shape)
print("y test : ",y_test.shape)

#%% parameter initialize and sigmoid function
#dimension=20
def initialize_weights_and_bias(dimension):
    w=np.full((dimension,1),0.01)
    b=0.0 
    return w,b

def sigmoid(z):
    y_head=1/(1+np.exp(-z))
    return y_head
#print(sigmoid(0))
#%%
def forward_backward_propagation(w,b,x_train,y_train):
    #forward propagation
    z=np.dot(w.T,x_train)+b
    y_head=sigmoid(z)
    loss=-y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost=(np.sum(loss))/x_train.shape[1]
    #backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling
    gradients={"derivative_weight":derivative_weight,"derivative_bias":derivative_bias}
    return cost,gradients
#%%Updating(Learning) parameters
def update(w,b,x_train,y_train,learning_rate,number_of_iterarion):
    cost_list=[]
    cost_list2=[]
    index=[]
    
    for i in range(number_of_iterarion):
        cost,gradients=forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        #lets update
        w=w-learning_rate*gradients["derivative_weight"]
        b=b-learning_rate*gradients["derivative_bias"]
        if i%10 ==0:
            cost_list2.append(cost)
            index.append(i)#grafik için bunları aldık
            print ("Cost after iteration %i: %f" %(i, cost))
            
    #we update(learn) parameters weights and bias
    parameters={"weight":w,"bias":b}
    plt.plot(index,cost_list2)
    plt.xticks(index,rotation="vertical")
    plt.xlabel("Number of Iterarrion")
    plt.ylabel("Cost")
    plt.show()
    return parameters,gradients,cost_list
#%%
# prediction
def predict(w,b,x_test):
    # x_test is a input for forward propagation
    z = sigmoid(np.dot(w.T,x_test)+b)
    Y_prediction = np.zeros((1,x_test.shape[1]))
    # if z is bigger than 0.5, our prediction is sign one (y_head=1),
    # if z is smaller than 0.5, our prediction is sign zero (y_head=0),
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1

    return Y_prediction
# predict(parameters["weight"],parameters["bias"],x_test) 
#%%
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):
    # initialize
    dimension =  x_train.shape[0]  # that is 20
    w,b = initialize_weights_and_bias(dimension)
    # do not change learning rate
    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)
    
    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)
    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 300)
algo_score_list=[]
#%% sklearn with lr
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train.T,y_train.T)
print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))
algo_score_list.append(["logistic reg",lr.score(x_test.T,y_test.T)])

#%% sklearn with knn
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train.T,y_train.T)
prediction = knn.predict(x_test.T)
#print('Prediction: {}'.format(prediction))
print('With KNN (K=3) accuracy is: ',knn.score(x_test.T,y_test.T)) # accuracy
y_pred=knn.predict(x_test.T)

#find best k value
score_list=[]
for each in range(1,15):
    knn2=KNeighborsClassifier(n_neighbors=each)
    knn2.fit(x_train.T,y_train.T)
    score_list.append(knn2.score(x_test.T,y_test.T))
# plot
plt.plot(range(1,15),score_list)
plt.xlabel("k values")
plt.ylabel("score")
plt.show()

#you will see k=8 best 
algo_score_list.append(["knn",knn2.score(x_test.T,y_test.T)])
from sklearn.svm import SVC

svm=SVC(random_state=1)
svm.fit(x_train.T,y_train.T)
print("pring accuracy of svm algo:",svm.score(x_test.T,y_test.T))
algo_score_list.append(["svm",svm.score(x_test.T,y_test.T)])
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train.T,y_train.T)


print("print accuracy of naive bayes algo:",nb.score(x_test.T,y_test.T))
algo_score_list.append(["naive bayes",nb.score(x_test.T,y_test.T)])

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train.T,y_train.T)

print("Tree score:",dt.score(x_test.T,y_test.T))
algo_score_list.append(["decision tree",dt.score(x_test.T,y_test.T)])
from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier(n_estimators=100,random_state=1)
rf.fit(x_train.T,y_train.T)
print("Random forest score:",rf.score(x_test.T,y_test.T))
algo_score_list.append(["random forest",rf.score(x_test.T,y_test.T)])

algo_score_list
algo_score_List=np.array(algo_score_list)
algo_score_List

algo_score_sorted_list = algo_score_List[algo_score_List[:,1].argsort()]
algoritma_isimleri=algo_score_sorted_list[:,0]
algoritma_skorlari=algo_score_sorted_list[:,1]


# Plot
x=algoritma_isimleri
y=algoritma_skorlari
plt.figure(figsize=(7,7))
plt.scatter(x, y,alpha=0.5)
plt.grid()
plt.title('Alghoritm Performance')
plt.xlabel('Alghoritm')
plt.ylabel('Score')
plt.show()
#%% confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test.T,y_pred)


# %% cm visualization
import seaborn as sns
import matplotlib.pyplot as plt

f, ax = plt.subplots(figsize =(5,5))
sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.show()



x_train=x_train.T
x_train.shape
y_train.reshape(2534,1)
y_train.shape
x_train.shape
# Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential # initialize neural network library
from keras.layers import Dense # build our layers library
def build_classifier():
    classifier = Sequential() # initialize neural network
    classifier.add(Dense(units = 8, kernel_initializer = 'uniform', activation = 'relu', input_dim = 20))
    classifier.add(Dense(units = 4, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier
classifier = KerasClassifier(build_fn = build_classifier, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = x_train, y = y_train.T, cv = 3)
mean = accuracies.mean()
variance = accuracies.std()
print("Accuracy mean: "+ str(mean))
print("Accuracy variance: "+ str(variance))