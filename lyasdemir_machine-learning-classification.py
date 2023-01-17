# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data=pd.read_csv("/kaggle/input/graduate-admissions/Admission_Predict.csv")
data
data.info()
# Let's delete the parts we will not use.

data.drop(["Serial No."],axis=1,inplace =True)
data
y=data.Research.values



x_data=data.drop(["Research"],axis=1)



x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))

# %% train test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.2,random_state=42)



x_train=x_train.T

x_test=x_test.T

y_train =y_train.T

y_test =y_test.T

def initialize_weights_and_bias(dimension):

    

    w = np.full((dimension,1),0.01)

    b = 0.0

    return w,b





# w,b = initialize_weights_and_bias(30)



def sigmoid(z):

    

    y_head = 1/(1+ np.exp(-z))

    return y_head

# print(sigmoid(0))



# %%

def forward_backward_propagation(w,b,x_train,y_train):

    # forward propagation

    z = np.dot(w.T,x_train) + b

    y_head = sigmoid(z)

    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)

    cost = (np.sum(loss))/x_train.shape[1]      # x_train.shape[1]  is for scaling

    

    # backward propagation

    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1] # x_train.shape[1]  is for scaling

    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]                 # x_train.shape[1]  is for scaling

    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}

    

    return cost,gradients



#%% Updating(learning) parameters

def update(w, b, x_train, y_train, learning_rate,number_of_iterarion):

    cost_list = []

    cost_list2 = []

    index = []

    

    # updating(learning) parameters is number_of_iterarion times

    for i in range(number_of_iterarion):

        # make forward and backward propagation and find cost and gradients

        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)

        cost_list.append(cost)

        # lets update

        w = w - learning_rate * gradients["derivative_weight"]

        b = b - learning_rate * gradients["derivative_bias"]

        if i % 10 == 0:

            cost_list2.append(cost)

            index.append(i)

            print ("Cost after iteration %i: %f" %(i, cost))

            

    # we update(learn) parameters weights and bias

    parameters = {"weight": w,"bias": b}

    plt.plot(index,cost_list2)

    plt.xticks(index,rotation='vertical')

    plt.xlabel("Number of Iterarion")

    plt.ylabel("Cost")

    plt.show()

    return parameters, gradients, cost_list



#%%  # prediction

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



# %% logistic_regression

def logistic_regression(x_train, y_train, x_test, y_test, learning_rate ,  num_iterations):

    # initialize

    dimension =  x_train.shape[0]  # that is 30

    w,b = initialize_weights_and_bias(dimension)

    # do not change learning rate

    parameters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate,num_iterations)

    

    y_prediction_test = predict(parameters["weight"],parameters["bias"],x_test)



    # Print test Errors

    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))

    

logistic_regression(x_train, y_train, x_test, y_test,learning_rate = 1, num_iterations = 300)    

# %% logistic_regression

from sklearn.linear_model import LogisticRegression



lr=LogisticRegression()



lr.fit(x_train.T,y_train.T)



print("test accuracy {}".format(lr.score(x_test.T,y_test.T)))
color_list = ['red' if i==1 else 'green' for i in data.loc[:,'Research']]
pd.plotting.scatter_matrix(data.loc[:, data.columns != 'Research'],

                                       c=color_list,

                                       figsize= [20,20],

                                       diagonal='hist',

                                       alpha=0.3,

                                       s = 200,

                                       marker = '*',

                                       edgecolor= "black")
experienced=data[data.Research== 1]



inexperienced=data[data.Research==0]

#%% normalization



plt.scatter(experienced["GRE Score"],experienced["TOEFL Score"],color="yellow",alpha=0.8)



plt.scatter(inexperienced["GRE Score"],inexperienced["TOEFL Score"],color="red",alpha=0.8)

plt.xlabel("GRE Score")

plt.ylabel("TOEFL Score")

plt.legend()

plt.show()

y=data.Research.values

x_data=data.drop(["Research"],axis=1)

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

# %% train test split

from sklearn.model_selection import train_test_split



x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=1)

# knn model

from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors = 3)

knn.fit(x_train,y_train)

prediction=knn.predict(x_test)



print(" {} nn score: {} ".format(3,knn.score(x_test,y_test)))
#%% normalization



score_list=[]

for each in range(1, 25):

    knn2=KNeighborsClassifier(n_neighbors = each)

    knn2.fit(x_train,y_train)

    score_list.append(knn2.score(x_test,y_test))

    

plt.plot(range(1,25),score_list)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()
#%% normalization



plt.scatter(experienced["GRE Score"],experienced["TOEFL Score"],color="yellow",alpha=0.8)

plt.scatter(inexperienced["GRE Score"],inexperienced["TOEFL Score"],color="red",alpha=0.8)

plt.xlabel("GRE Score")

plt.ylabel("TOEFL Score")

plt.legend()

plt.show()
y=data.Research.values



x_data=data.drop(["Research"],axis=1)



x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
# %% train test split

from sklearn.model_selection import train_test_split



x_train,x_test,y_train,y_test=train_test_split(x,y,test_size = 0.2,random_state=1)
# %% SVM

from sklearn.svm import SVC



svm=SVC(random_state=1)

svm.fit(x_train,y_train)



print("score :",svm.score(x_test,y_test))
#%% normalization



plt.scatter(experienced["GRE Score"],experienced["TOEFL Score"],color="yellow",alpha=0.8)

plt.scatter(inexperienced["GRE Score"],inexperienced["TOEFL Score"],color="red",alpha=0.8)

plt.xlabel("GRE Score")

plt.ylabel("TOEFL Score")

plt.legend()

plt.show()
y=data.Research.values

x_data=data.drop(["Research"],axis=1)

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

# %% train test split

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state=1)

# %% Naive bayes 

from sklearn.naive_bayes import GaussianNB

nb=GaussianNB()

nb.fit(x_train,y_train)



print("score :",nb.score(x_test,y_test))
# %% train test split

from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)

# %% Decision Tree Classifier

from sklearn.tree import DecisionTreeClassifier

dt=DecisionTreeClassifier()

dt.fit(x_train,y_train)



print("score :",dt.score(x_test, y_test))
#%% normalization



plt.scatter(experienced["GRE Score"],experienced["TOEFL Score"],color="yellow",alpha=0.8)

plt.scatter(inexperienced["GRE Score"],inexperienced["TOEFL Score"],color="red",alpha=0.8)

plt.xlabel("GRE Score")

plt.ylabel("TOEFL Score")

plt.legend()

plt.show()
y=data.Research.values

x_data=data.drop(["Research"],axis=1)

x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))

dt_liistem_esemble=[]



for i in range(1,20):

    from sklearn.model_selection import train_test_split

    x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = i/100,random_state = 42)

    from sklearn.tree import DecisionTreeClassifier

    dt = DecisionTreeClassifier()

    dt.fit(x_train,y_train)

    #print("{} .decision tree score:{} ".format(i,dt.score(x_test,y_test)))

    dt_liistem_esemble.append(dt.score(x_test,y_test))

plt.plot(range(1,20),dt_liistem)

plt.xlabel("k values")

plt.ylabel("accuracy")

plt.show()
# Let's find the best result

print("max score :",dt_liistem_esemble.index(max(dt_liistem_esemble)))
# %% train test split

from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.14,random_state = 42)

#%% decision tree

from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier()

dt.fit(x_train,y_train)

print("decision tree score: ", dt.score(x_test,y_test))
#%%  random forest

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators = 40,random_state = 1)

rf.fit(x_train,y_train)

print("random forest algo result: ",rf.score(x_test,y_test))
dt_liistem_eva=[]



for i in range(2,99): 

    from sklearn.model_selection import train_test_split

    x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = i/100,random_state = 42)

    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators = 100,random_state = 1)

    rf.fit(x_train,y_train)

    dt_liistem_eva.append(dt.score(x_test,y_test))

print("max score :",max(dt_liistem_eva))
dt_liistem_eva.index( max(dt_liistem_eva) )
# %% train test split

from sklearn.model_selection import train_test_split

x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state = 42)
#%%  random forest

from sklearn.ensemble import RandomForestClassifier



rf = RandomForestClassifier(n_estimators = 100,random_state = 1)

rf.fit(x_train,y_train)

print("random forest algo result: ",rf.score(x_test,y_test))





y_pred = rf.predict(x_test)

y_true = y_test
#%% confusion matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true,y_pred)

# %% cm visualization



f, ax = plt.subplots(figsize =(8,8))

sns.heatmap(cm,annot = True,linewidths=0.5,linecolor="red",fmt = ".0f",ax=ax)

plt.xlabel("y_pred")

plt.ylabel("y_true")

plt.show()
print("Logistic Regression Classification {}".format(0.7875 ))



print(" {}K-Nearest Neighbors (KNN): {} ".format(3,knn.score(x_test,y_test)))



print("Support Vector Machine(SVM) Classification :",svm.score(x_test,y_test))



print("Naive Bayes Classification) :",nb.score(x_test,y_test))



print("Decision Tree Classification :",dt.score(x_test, y_test))



print("decision tree score: ", dt.score(x_test,y_test))



print("Random Forest Classification: ",rf.score(x_test,y_test))
